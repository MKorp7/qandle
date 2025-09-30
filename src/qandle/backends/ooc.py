"""Out-of-core state-vector simulator backend.

This module implements the :class:`OOCStateVectorSimulator`, a backend capable of
streaming gate applications from host memory (or a memory-mapped file) to the
compute device.  The implementation purposefully keeps the GPU streaming path
simple – the data is staged through pinned host buffers before being copied to a
single device buffer – which is sufficient for correctness focused unit tests.

The simulator primarily targets scenarios where the full state vector cannot be
kept resident on a GPU and therefore needs to be processed in micro tiles.  The
CPU path is also implemented in terms of the same tiling logic, ensuring that
both modes share the same numerics.
"""
from __future__ import annotations

import contextlib
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Iterable, Iterator, Literal, Optional, Sequence, Tuple

import numpy as np
import torch

DeviceLike = Literal["cpu", "cuda"]


_TORCH_TO_NUMPY_DTYPE = {
    torch.complex64: np.complex64,
    torch.complex128: np.complex128,
}


@dataclass(slots=True)
class _MicroTileConfig:
    stride: int
    length: int


def _bytes_per_amplitude(dtype: torch.dtype) -> int:
    probe = torch.empty(0, dtype=dtype)
    return probe.element_size()


def choose_micro_tile_len(
    stride_small: int,
    arity: int,
    bytes_per_amp: int,
    host_stage_bytes: int,
    gpu_stage_bytes: Optional[int],
) -> int:
    """Return a micro tile length that satisfies the staging constraints."""

    cap = host_stage_bytes
    if gpu_stage_bytes is not None:
        cap = min(cap, gpu_stage_bytes)
    m = cap // (arity * bytes_per_amp)
    if m <= 0:
        m = 1
    m = min(m, stride_small)
    # round to power of two for better alignment
    m = 1 << (m.bit_length() - 1)
    m = max(1, m)
    return m


def _ensure_numpy_dtype(dtype: torch.dtype) -> np.dtype:
    if dtype not in _TORCH_TO_NUMPY_DTYPE:
        raise ValueError(f"Unsupported dtype for OOC simulator: {dtype}")
    return _TORCH_TO_NUMPY_DTYPE[dtype]


class OOCStateVectorSimulator:
    """State-vector simulator that operates out-of-core.

    Parameters
    ----------
    n_qubits:
        Number of qubits.
    dtype:
        Complex dtype to use for the amplitudes (default: ``torch.complex64``).
    device:
        Compute device.  ``"cpu"`` performs all work on the host, ``"cuda"``
        copies micro tiles to the current CUDA device.
    storage:
        Either ``"mem"`` for an in-memory tensor or ``"memmap"`` to back the
        state vector with a memory mapped file.
    memmap_path:
        Optional location of the memory mapped file.  When ``None`` a temporary
        file is created.
    host_stage_bytes / gpu_stage_bytes:
        Limits for the staging buffers.  Smaller values enforce finer tiling and
        keep the resident working set under control.
    streams:
        Currently unused; the implementation runs on the default stream, but the
        parameter is kept for API compatibility with future streaming support.
    seed:
        Seed controlling the random initialisation when ``init_state`` is called
        with ``kind="random"``.
    """

    def __init__(
        self,
        n_qubits: int,
        dtype: torch.dtype = torch.complex64,
        device: DeviceLike = "cpu",
        storage: Literal["mem", "memmap"] = "mem",
        memmap_path: Optional[str] = None,
        host_stage_bytes: int = 256 * 2**20,
        gpu_stage_bytes: Optional[int] = None,
        streams: int = 2,
        seed: Optional[int] = None,
    ) -> None:
        if n_qubits <= 0:
            raise ValueError("The simulator requires at least one qubit")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but PyTorch CUDA is not available")

        self.n_qubits = n_qubits
        self.dtype = dtype
        self.device: DeviceLike = device
        self.storage = storage
        self.memmap_path = memmap_path
        self.host_stage_bytes = int(host_stage_bytes)
        self.gpu_stage_bytes = int(gpu_stage_bytes) if gpu_stage_bytes is not None else None
        self.streams = streams
        self.seed = seed

        self._num_amps = 1 << n_qubits
        self._bytes_per_amp = _bytes_per_amplitude(dtype)
        self._state_bytes = self._num_amps * self._bytes_per_amp
        self._generator = torch.Generator(device="cpu")
        if seed is not None:
            self._generator.manual_seed(seed)

        self._memmap_array: Optional[np.memmap] = None
        self.state: Optional[torch.Tensor] = None
        self.init_state()

    # ------------------------------------------------------------------
    # Allocation helpers
    def _allocate_tensor(self) -> torch.Tensor:
        if self.storage == "mem":
            state = torch.empty(self._num_amps, dtype=self.dtype)
            return state
        if self.storage == "memmap":
            path = self.memmap_path
            if path is None:
                fd, path = tempfile.mkstemp(prefix="qandle_ooc_", suffix=".mm")
                os.close(fd)
                self.memmap_path = path
            numpy_dtype = _ensure_numpy_dtype(self.dtype)
            with open(path, "wb") as f:
                f.truncate(self._state_bytes)
            mm = np.memmap(path, dtype=numpy_dtype, mode="r+", shape=(self._num_amps,))
            self._memmap_array = mm
            tensor = torch.from_numpy(mm)
            if tensor.dtype != self.dtype:
                raise RuntimeError(
                    "Memmap tensor dtype mismatch: expected "
                    f"{self.dtype}, received {tensor.dtype}"
                )
            return tensor
        raise ValueError(f"Unknown storage kind: {self.storage}")

    # ------------------------------------------------------------------
    # Public API
    def init_state(self, kind: Literal["zero", "random"] = "zero") -> None:
        """Initialise the state vector."""

        state = self._allocate_tensor()
        if kind == "zero":
            state.zero_()
            state[0] = torch.tensor(1.0, dtype=self.dtype)
        elif kind == "random":
            float_dtype = torch.float32 if self.dtype == torch.complex64 else torch.float64
            re = torch.randn(self._num_amps, dtype=float_dtype, generator=self._generator)
            im = torch.randn(self._num_amps, dtype=float_dtype, generator=self._generator)
            psi = torch.complex(re, im).to(dtype=self.dtype)
            psi = psi / psi.norm()
            state.copy_(psi)
        else:
            raise ValueError(f"Unknown initialisation kind: {kind}")
        self.state = state

    # ------------------------------------------------------------------
    def _micro_tile_config(self, stride_small: int, arity: int) -> _MicroTileConfig:
        length = choose_micro_tile_len(
            stride_small=stride_small,
            arity=arity,
            bytes_per_amp=self._bytes_per_amp,
            host_stage_bytes=self.host_stage_bytes,
            gpu_stage_bytes=self.gpu_stage_bytes if self.device == "cuda" else None,
        )
        return _MicroTileConfig(stride=stride_small, length=length)

    # ------------------------------------------------------------------
    def apply_1q(self, q: int, U2: torch.Tensor) -> None:
        if self.state is None:
            raise RuntimeError("State has not been initialised")
        if U2.shape != (2, 2):
            raise ValueError("1-qubit gate must be a 2x2 matrix")
        tile = self._micro_tile_config(1 << q, arity=2)
        if self.device == "cuda":
            self._apply_1q_cuda(q, U2)
        else:
            self._apply_1q_cpu(q, U2.to(dtype=self.dtype, device="cpu"), tile)

    def apply_2q(self, p: int, q: int, U4: torch.Tensor) -> None:
        if self.state is None:
            raise RuntimeError("State has not been initialised")
        if U4.shape != (4, 4):
            raise ValueError("2-qubit gate must be a 4x4 matrix")
        if p == q:
            raise ValueError("Distinct qubits required for a 2-qubit gate")
        p, q = sorted((p, q))
        tile = self._micro_tile_config(1 << p, arity=4)
        if self.device == "cuda":
            self._apply_2q_cuda(p, q, U4)
        else:
            self._apply_2q_cpu(p, q, U4.to(dtype=self.dtype, device="cpu"), tile)

    # ------------------------------------------------------------------
    def run(self, gates: Iterable[Tuple[str, Tuple[int, ...], torch.Tensor]]) -> torch.Tensor:
        """Apply a sequence of gates.

        Parameters
        ----------
        gates:
            Iterable of gate descriptions ``(op_type, qubits, matrix)`` where
            ``op_type`` is ``"1q"`` or ``"2q"``.
        """

        for op_type, qubits, matrix in gates:
            if op_type == "1q":
                if len(qubits) != 1:
                    raise ValueError("1q gate expects a single qubit index")
                self.apply_1q(qubits[0], matrix)
            elif op_type == "2q":
                if len(qubits) != 2:
                    raise ValueError("2q gate expects two qubit indices")
                self.apply_2q(qubits[0], qubits[1], matrix)
            else:
                raise ValueError(f"Unknown gate type: {op_type}")
        if self.state is None:
            raise RuntimeError("State is unavailable after run")
        return self.state

    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def timer(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            print(f"[{name}] {end - start:.3f}s")

    # ------------------------------------------------------------------
    # CPU implementations ------------------------------------------------
    def _apply_1q_cpu(self, q: int, U2: torch.Tensor, tile: _MicroTileConfig) -> None:
        assert self.state is not None
        state = self.state
        N = state.numel()
        s = 1 << q
        period = s << 1
        m = tile.length
        for base in range(0, N, period):
            for t in range(0, s, m):
                length = min(m, s - t)
                a_start = base + t
                b_start = a_start + s
                a = state.narrow(0, a_start, length).clone()
                b = state.narrow(0, b_start, length).clone()
                stacked = torch.stack((a, b), dim=1)
                updated = stacked @ U2.T
                state.narrow(0, a_start, length).copy_(updated[:, 0])
                state.narrow(0, b_start, length).copy_(updated[:, 1])

    def _apply_2q_cpu(self, p: int, q: int, U4: torch.Tensor, tile: _MicroTileConfig) -> None:
        assert self.state is not None
        state = self.state
        N = state.numel()
        s0 = 1 << p
        s1 = 1 << q
        period = 1 << (q + 1)
        m = tile.length
        for base in range(0, N, period):
            for t in range(0, s0, m):
                length = min(m, s0 - t)
                i00 = base + t
                i10 = i00 + s0
                i01 = i00 + s1
                i11 = i10 + s1
                a00 = state.narrow(0, i00, length).clone()
                a10 = state.narrow(0, i10, length).clone()
                a01 = state.narrow(0, i01, length).clone()
                a11 = state.narrow(0, i11, length).clone()
                stacked = torch.stack((a00, a10, a01, a11), dim=1)
                updated = stacked @ U4.T
                state.narrow(0, i00, length).copy_(updated[:, 0])
                state.narrow(0, i10, length).copy_(updated[:, 1])
                state.narrow(0, i01, length).copy_(updated[:, 2])
                state.narrow(0, i11, length).copy_(updated[:, 3])

    # ------------------------------------------------------------------
    # Simplified CUDA implementations -----------------------------------
    def _cuda_device(self) -> torch.device:
        return torch.device("cuda")

    def _allocate_cuda_buffers(self, count: int, length: int) -> Tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]]:
        device = self._cuda_device()
        host_buffers = [torch.empty(length, dtype=self.dtype, pin_memory=True) for _ in range(count)]
        device_buffers = [torch.empty(length, dtype=self.dtype, device=device) for _ in range(count)]
        return host_buffers, device_buffers

    def _apply_1q_cuda(self, q: int, U2: torch.Tensor) -> None:
        assert self.state is not None
        state = self.state
        N = state.numel()
        s = 1 << q
        period = s << 1
        tile = self._micro_tile_config(s, arity=2)
        m = tile.length
        host_buffers, device_buffers = self._allocate_cuda_buffers(2, m)
        U2 = U2.to(device=self._cuda_device(), dtype=self.dtype)
        stream = torch.cuda.current_stream()
        for base in range(0, N, period):
            for t in range(0, s, m):
                length = min(m, s - t)
                a_start = base + t
                b_start = a_start + s
                host_buffers[0][:length].copy_(state.narrow(0, a_start, length))
                host_buffers[1][:length].copy_(state.narrow(0, b_start, length))
                device_buffers[0][:length].copy_(host_buffers[0][:length], non_blocking=True)
                device_buffers[1][:length].copy_(host_buffers[1][:length], non_blocking=True)
                stream.synchronize()
                stacked = torch.stack((device_buffers[0][:length], device_buffers[1][:length]), dim=1)
                updated = stacked @ U2.T
                device_buffers[0][:length].copy_(updated[:, 0])
                device_buffers[1][:length].copy_(updated[:, 1])
                host_buffers[0][:length].copy_(device_buffers[0][:length], non_blocking=True)
                host_buffers[1][:length].copy_(device_buffers[1][:length], non_blocking=True)
                stream.synchronize()
                state.narrow(0, a_start, length).copy_(host_buffers[0][:length])
                state.narrow(0, b_start, length).copy_(host_buffers[1][:length])

    def _apply_2q_cuda(self, p: int, q: int, U4: torch.Tensor) -> None:
        assert self.state is not None
        state = self.state
        N = state.numel()
        s0 = 1 << p
        s1 = 1 << q
        period = 1 << (q + 1)
        tile = self._micro_tile_config(s0, arity=4)
        m = tile.length
        host_buffers, device_buffers = self._allocate_cuda_buffers(4, m)
        U4 = U4.to(device=self._cuda_device(), dtype=self.dtype)
        stream = torch.cuda.current_stream()
        for base in range(0, N, period):
            for t in range(0, s0, m):
                length = min(m, s0 - t)
                i00 = base + t
                i10 = i00 + s0
                i01 = i00 + s1
                i11 = i10 + s1
                host_buffers[0][:length].copy_(state.narrow(0, i00, length))
                host_buffers[1][:length].copy_(state.narrow(0, i10, length))
                host_buffers[2][:length].copy_(state.narrow(0, i01, length))
                host_buffers[3][:length].copy_(state.narrow(0, i11, length))
                for idx in range(4):
                    device_buffers[idx][:length].copy_(host_buffers[idx][:length], non_blocking=True)
                stream.synchronize()
                stacked = torch.stack(
                    (
                        device_buffers[0][:length],
                        device_buffers[1][:length],
                        device_buffers[2][:length],
                        device_buffers[3][:length],
                    ),
                    dim=1,
                )
                updated = stacked @ U4.T
                for idx in range(4):
                    device_buffers[idx][:length].copy_(updated[:, idx])
                for idx in range(4):
                    host_buffers[idx][:length].copy_(device_buffers[idx][:length], non_blocking=True)
                stream.synchronize()
                state.narrow(0, i00, length).copy_(host_buffers[0][:length])
                state.narrow(0, i10, length).copy_(host_buffers[1][:length])
                state.narrow(0, i01, length).copy_(host_buffers[2][:length])
                state.narrow(0, i11, length).copy_(host_buffers[3][:length])
