from typing import Iterable, Tuple

import pytest

torch = pytest.importorskip("torch")

from qandle.backends import OOCStateVectorSimulator, StateVectorBackend


def _random_unitary(dim: int, dtype: torch.dtype, generator: torch.Generator) -> torch.Tensor:
    re_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    mat = torch.randn(dim, dim, dtype=re_dtype, generator=generator)
    mat = mat + 1j * torch.randn(dim, dim, dtype=re_dtype, generator=generator)
    q, r = torch.linalg.qr(mat)
    # normalise determinant to unit modulus
    diag = torch.diagonal(r)
    phase = diag / diag.abs().clamp(min=1e-12)
    q = q * phase
    return q.to(dtype=dtype)


def _build_random_circuit(
    n_qubits: int,
    depth: int,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> Iterable[Tuple[str, Tuple[int, ...], torch.Tensor]]:
    gates = []
    for _ in range(depth):
        # single qubit layer
        for q in range(n_qubits):
            gates.append(("1q", (q,), _random_unitary(2, dtype, generator)))
        # two qubit layer on random neighbours
        for q in range(0, n_qubits - 1, 2):
            gates.append(("2q", (q, q + 1), _random_unitary(4, dtype, generator)))
    return gates


def test_ooc_matches_statevector_small():
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1234)
    n_qubits = 4
    dtype = torch.complex64

    gates = list(_build_random_circuit(n_qubits, depth=2, dtype=dtype, generator=generator))

    ref_backend = StateVectorBackend(n_qubits=n_qubits, dtype=dtype, device="cpu")
    sim = OOCStateVectorSimulator(
        n_qubits=n_qubits,
        dtype=dtype,
        device="cpu",
        host_stage_bytes=256,
    )

    for op, wires, matrix in gates:
        if op == "1q":
            ref_backend.apply_1q(matrix, wires[0])
        else:
            ref_backend.apply_2q(matrix, wires[0], wires[1])
    sim.run(gates)

    assert torch.allclose(sim.state, ref_backend.state, atol=1e-6, rtol=1e-6)


def test_ooc_handles_edge_strides():
    generator = torch.Generator(device="cpu")
    generator.manual_seed(2023)
    n_qubits = 4
    dtype = torch.complex128
    sim = OOCStateVectorSimulator(
        n_qubits=n_qubits,
        dtype=dtype,
        device="cpu",
        host_stage_bytes=128,
    )
    ref = StateVectorBackend(n_qubits=n_qubits, dtype=dtype, device="cpu")

    # apply gate on most significant qubit with very small tiles
    gate_1q = _random_unitary(2, dtype, generator)
    sim.apply_1q(n_qubits - 1, gate_1q)
    ref.apply_1q(gate_1q, n_qubits - 1)

    # two qubit gate on (0, n-1)
    gate_2q = _random_unitary(4, dtype, generator)
    sim.apply_2q(0, n_qubits - 1, gate_2q)
    ref.apply_2q(gate_2q, 0, n_qubits - 1)

    assert torch.allclose(sim.state, ref.state, atol=1e-9, rtol=1e-9)


def test_ooc_memmap_roundtrip(tmp_path):
    path = tmp_path / "state.mm"
    sim = OOCStateVectorSimulator(
        n_qubits=3,
        dtype=torch.complex64,
        device="cpu",
        storage="memmap",
        memmap_path=str(path),
        host_stage_bytes=128,
    )
    sim.apply_1q(0, torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64))
    sim.apply_2q(0, 2, torch.eye(4, dtype=torch.complex64))
    mapped = torch.from_file(str(path), dtype=torch.complex64, size=2**3)
    assert torch.allclose(sim.state, mapped, atol=1e-7, rtol=1e-7)
