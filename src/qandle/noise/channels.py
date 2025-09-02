"""Noise channel definitions using Kraus operators.

This module provides a small collection of common noise channels.  The
channels are implemented via their Kraus representations and can act on
density matrices exactly or on statevectors by stochastic trajectory
sampling.  The API mirrors the gate API from :mod:`qandle.operators` with
unbuilt containers that can be :meth:`build` into modules.

``PhaseFlip`` and ``Dephasing`` both reduce coherence but differ in their
scaling factors: Phase flips scale off-diagonals by ``1 - 2p`` while
dephasing scales by ``1 - gamma``.  Use the former to model discrete ``Z``
flips and the latter for continuous phase damping.

Example
-------

>>> import torch
>>> from qandle.noise.channels import PhaseFlip, Dephasing
>>> plus = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.complex64)
>>> p = 0.2
>>> pf = PhaseFlip(p, 0).build(1)(plus)
>>> torch.allclose(pf[0, 1].real, torch.tensor(0.5 * (1 - 2 * p)))
True
>>> gamma = 0.2
>>> dp = Dephasing(gamma, 0).build(1)(plus)
>>> torch.allclose(dp[0, 1].real, torch.tensor(0.5 * (1 - gamma)))
True

Trajectory reproducibility
--------------------------

>>> g = torch.Generator().manual_seed(0)
>>> ch = PhaseFlip(0.5, 0).build(1)
>>> psi1 = ch(torch.tensor([1.0, 0.0], dtype=torch.complex64), trajectory=True, rng=g)
>>> g = torch.Generator().manual_seed(0)
>>> psi2 = ch(torch.tensor([1.0, 0.0], dtype=torch.complex64), trajectory=True, rng=g)
>>> torch.allclose(psi1, psi2)
True
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Sequence

import torch
import warnings

from .. import qasm
from ..operators import BuiltOperator, UnbuiltOperator

__all__ = [
    "NoiseChannel",
    "BuiltNoiseChannel",
    "PhaseFlip",
    "Depolarizing",
    "Dephasing",
    "PhaseDamping",  # backwards compatibility alias
    "AmplitudeDamping",
    "CorrelatedDepolarizing",
]


######################################################################
# Utilities


# Maximum number of qubits allowed for explicit global matrix embedding.
MAX_GLOBAL_KRAUS_QUBITS = 10

# Axis permutations used when applying Kraus operators to density matrices.
# PERM2_NB groups bra/ket indices as (target, rest, target, rest) and the
# inverse undoes this grouping.
PERM2_NB = [1, 3, 0, 2]
INV_PERM2_NB = [2, 0, 3, 1]


@dataclass(frozen=True)
class _PermCache:
    n: int
    t: int
    rest: int
    perm_nb: List[int]
    inv_perm_nb: List[int]
    perm_state_nb: List[int]
    inv_perm_state_nb: List[int]


@lru_cache(maxsize=256)
def _get_perm_cache(targets: tuple[int, ...], n: int) -> _PermCache:
    t = len(targets)
    rest = n - t
    t_left = list(targets)
    rest_left = [i for i in range(n) if i not in t_left]
    t_right = [n + i for i in targets]
    rest_right = [n + i for i in range(n) if i not in targets]
    perm_nb = t_left + rest_left + t_right + rest_right
    inv_perm_nb = [0] * (2 * n)
    for i, p in enumerate(perm_nb):
        inv_perm_nb[p] = i

    perm_state_nb = list(targets) + [i for i in range(n) if i not in targets]
    inv_perm_state_nb = [0] * n
    for i, p in enumerate(perm_state_nb):
        inv_perm_state_nb[p] = i

    return _PermCache(
        n=n,
        t=t,
        rest=rest,
        perm_nb=perm_nb,
        inv_perm_nb=inv_perm_nb,
        perm_state_nb=perm_state_nb,
        inv_perm_state_nb=inv_perm_state_nb,
    )

def _bits(i: int, n: int) -> List[int]:
    """Return the ``n``-bit binary representation of ``i`` (big endian)."""

    return [int(bool(i & (1 << (n - 1 - k)))) for k in range(n)]


def _bits_to_int(bits: Iterable[int]) -> int:
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def _format_targets(targets: Sequence[int]) -> str:
    """Return formatted string for ``targets`` as ``[q0,q1,...]``."""

    return "[" + ",".join(f"q{t}" for t in targets) + "]"

def _apply_kraus_density(rho: torch.Tensor, k: torch.Tensor, cache: _PermCache) -> torch.Tensor:
    """Apply a local Kraus operator to a density matrix using cached perms."""

    if cache.t == 0:
        return rho

    batch = rho.shape[:-2]
    n = cache.n
    perm = list(range(len(batch))) + [len(batch) + p for p in cache.perm_nb]
    rho_t = rho.reshape(*batch, *(2,) * n, *(2,) * n).permute(perm).contiguous()
    rho_t = rho_t.reshape(*batch, 2 ** cache.t, 2 ** cache.rest, 2 ** cache.t, 2 ** cache.rest)

    perm2 = list(range(len(batch))) + [len(batch) + p for p in PERM2_NB]
    rho_blocks = rho_t.permute(perm2).contiguous()

    # ``rho_blocks`` groups bra/ket indices as
    # (..., i=rest_bra, j=rest_ket, b=target_bra, c=target_ket).
    # The einsum ``ab,...ijbc->...ijac`` multiplies the Kraus operator on the
    # bra ("b") indices, replacing them with "a".
    tmp = torch.einsum("ab,...ijbc->...ijac", k, rho_blocks)
    # ``dc,...ijac->...ijad`` then applies :math:`K^\dagger` on the ket
    # ("c") indices, turning them into "d".
    out_blocks = torch.einsum("dc,...ijac->...ijad", k.conj(), tmp)

    inv_perm2 = list(range(len(batch))) + [len(batch) + p for p in INV_PERM2_NB]
    out = out_blocks.permute(inv_perm2).contiguous()
    out = out.reshape(
        *batch, *(2,) * cache.t, *(2,) * cache.rest, *(2,) * cache.t, *(2,) * cache.rest
    )

    inv_perm = list(range(len(batch))) + [len(batch) + p for p in cache.inv_perm_nb]
    out = out.permute(inv_perm).contiguous()
    return out.reshape(*batch, 1 << n, 1 << n)


def _apply_kraus_density_all(
    rho: torch.Tensor, ks: torch.Tensor, cache: _PermCache
) -> torch.Tensor:
    """Apply stacked Kraus operators in ``ks`` to ``rho`` using cached perms."""

    if cache.t == 0:
        return rho

    batch = rho.shape[:-2]
    n = cache.n
    perm = list(range(len(batch))) + [len(batch) + p for p in cache.perm_nb]
    rho_t = rho.reshape(*batch, *(2,) * n, *(2,) * n).permute(perm).contiguous()
    rho_t = rho_t.reshape(*batch, 2 ** cache.t, 2 ** cache.rest, 2 ** cache.t, 2 ** cache.rest)

    perm2 = list(range(len(batch))) + [len(batch) + p for p in PERM2_NB]
    rho_blocks = rho_t.permute(perm2).contiguous()

    # ``rho_blocks`` has the same (..., i=rest_bra, j=rest_ket, b=target_bra,
    # c=target_ket) structure as above.  ``ks`` stacks Kraus operators along
    # the leading ``m`` axis, which is carried through as part of the batch.
    # ``mab,...ijbc->m...ijac`` applies each Kraus operator on the bra indices.
    tmp = torch.einsum("mab,...ijbc->m...ijac", ks, rho_blocks)
    # Finally ``mdc,m...ijac->...ijad`` contracts the ket indices with the
    # conjugate operators and sums over ``m``.
    out_blocks = torch.einsum("mdc,m...ijac->...ijad", ks.conj(), tmp)

    inv_perm2 = list(range(len(batch))) + [len(batch) + p for p in INV_PERM2_NB]
    out = out_blocks.permute(inv_perm2).contiguous()
    out = out.reshape(
        *batch, *(2,) * cache.t, *(2,) * cache.rest, *(2,) * cache.t, *(2,) * cache.rest
    )

    inv_perm = list(range(len(batch))) + [len(batch) + p for p in cache.inv_perm_nb]
    out = out.permute(inv_perm).contiguous()
    return out.reshape(*batch, 1 << n, 1 << n)

def _apply_kraus_state(psi: torch.Tensor, k: torch.Tensor, cache: _PermCache) -> torch.Tensor:
    """Apply a local Kraus operator to a statevector using cached perms."""

    if cache.t == 0:
        return psi

    batch = psi.shape[:-1]
    n = cache.n
    perm = list(range(len(batch))) + [len(batch) + p for p in cache.perm_state_nb]
    psi_t = psi.reshape(*batch, *(2,) * n).permute(perm).contiguous()
    psi_t = psi_t.reshape(*batch, 2 ** cache.t, 2 ** cache.rest)
    out = torch.einsum("ab,...bc->...ac", k, psi_t)

    inv_perm = list(range(len(batch))) + [len(batch) + p for p in cache.inv_perm_state_nb]
    out = out.reshape(*batch, *(2,) * cache.t, *(2,) * cache.rest)
    out = out.permute(inv_perm).contiguous()
    return out.reshape(*batch, 1 << n)


# Slow fallback embedding for explicit matrix construction ------------------

def _embed_kraus_slow(
    k: torch.Tensor, targets: Sequence[int], n: int, *, dtype, device
) -> torch.Tensor:
    """Embed ``k`` into an ``n``-qubit space (slow; for diagnostics only)."""

    dim = 1 << n
    out = torch.zeros((dim, dim), dtype=dtype, device=device)
    tset = list(targets)
    for i in range(dim):
        ib = _bits(i, n)
        for j in range(dim):
            jb = _bits(j, n)
            if any(ib[q] != jb[q] for q in range(n) if q not in tset):
                continue
            li = _bits_to_int([ib[q] for q in tset])
            lj = _bits_to_int([jb[q] for q in tset])
            out[i, j] = k[li, lj]
    return out


# ---------------------------------------------------------------------------
# Base classes


class NoiseChannel(UnbuiltOperator):
    """Base class for *unbuilt* noise channels.

    Unbuilt channels describe a local noise process acting on a set of target
    qubits but are not yet attached to a concrete register size.  Calling
    :meth:`build` binds the channel to a specific number of qubits and returns a
    :class:`BuiltNoiseChannel` that can be applied to states.  The ``targets``
    are stored in sorted order for convenience; they are validated for
    uniqueness and in-range indices during :meth:`build`.

    """

    targets: Sequence[int]

    def __init__(self, targets: Sequence[int]):
        # Sort targets for ergonomic usage; duplicates are checked during build.
        self.targets = tuple(sorted(targets))

    # ------------------------------------------------------------------
    # Interfaces to override
    def to_kraus(self, *, dtype=torch.complex64, device=None) -> List[torch.Tensor]:
        """Return the local Kraus operators of the channel."""

        raise NotImplementedError

    # Optional helper ---------------------------------------------------
    def to_superoperator(self, *, dtype=torch.complex64, device=None) -> torch.Tensor:
        """Return the local superoperator matrix ``S = sum_i K_i ⊗ K_i*``."""

        kraus = self.to_kraus(dtype=dtype, device=device)
        return sum(torch.kron(k, k.conj()) for k in kraus)

    def to_qasm(self) -> qasm.QasmRepresentation:  # pragma: no cover - trivial
        return qasm.QasmRepresentation(gate_str=f"// noise: {self}")

    # Building ----------------------------------------------------------
    def build(self, num_qubits: int, **kwargs) -> "BuiltNoiseChannel":  # type: ignore[override]
        # validate targets
        if not self.targets:
            raise ValueError("at least one target required")
        if len(set(self.targets)) != len(self.targets):
            raise ValueError(
                f"targets must be unique; got targets={self.targets} with num_qubits={num_qubits}"
            )
        if any(t < 0 or t >= num_qubits for t in self.targets):
            raise ValueError(
                f"target index out of range for num_qubits={num_qubits}; targets={self.targets}"
            )
        return BuiltNoiseChannel(channel=self, num_qubits=num_qubits)


class BuiltNoiseChannel(BuiltOperator):
    """Built version of :class:`NoiseChannel`.

    Once built, the channel knows about the total number of qubits and can be
    executed in two modes:

    ``trajectory=False`` (default)
        Operate on density matrices by applying all Kraus operators and
        summing the results, which is fully differentiable.
    ``trajectory=True``
        Perform a stochastic quantum trajectory simulation on statevectors by
        sampling a single Kraus operator.  This mode is non-differentiable.  A
        :class:`torch.Generator` can be supplied via ``rng`` to control the
        random choices.
    """

    def __init__(self, channel: NoiseChannel, num_qubits: int):
        super().__init__()
        self.channel = channel
        self.targets = channel.targets
        self.num_qubits = num_qubits
        self._kraus_cache: dict[tuple[torch.dtype, torch.device], List[torch.Tensor]] = {}
        self._perm_cache = _get_perm_cache(tuple(self.targets), self.num_qubits)

    def to(self, *args, **kwargs):  # type: ignore[override]
        out = super().to(*args, **kwargs)
        self._kraus_cache.clear()
        return out

    # Helper to obtain local Kraus operators ----------------------------
    def _local_kraus(self, dtype, device):
        key = (dtype, device)
        if key not in self._kraus_cache:
            self._kraus_cache[key] = self.channel.to_kraus(dtype=dtype, device=device)
        return self._kraus_cache[key]

    # Operator interface ------------------------------------------------
    def __str__(self) -> str:  # pragma: no cover - small wrapper
        return str(self.channel)

    def forward(self, state: torch.Tensor, *, trajectory: bool = False, rng=None) -> torch.Tensor:
        """Apply the channel to ``state``.

        Real inputs are promoted to ``complex64``. ``trajectory=True`` performs
        stochastic Kraus sampling and is therefore non-differentiable.
        ``state`` must already live on the desired device.

        Example
        -------
        Re-seeding a :class:`torch.Generator` yields identical trajectory
        outcomes:

        >>> ch = PhaseFlip(p=0.1, qubit=0).build(num_qubits=1)
        >>> psi = torch.tensor([1.0, 0.0], dtype=torch.complex64)
        >>> gen = torch.Generator().manual_seed(0)
        >>> out1 = ch(psi, trajectory=True, rng=gen)
        >>> gen.manual_seed(0)
        >>> out2 = ch(psi, trajectory=True, rng=gen)
        >>> torch.allclose(out1, out2)
        True

        """
        if getattr(self.channel, "is_identity", False):
            return state

        if not torch.is_complex(state):
            # Promote real inputs to an appropriate complex dtype.  ``float64``
            # (and ``double`` aliases) are promoted to ``complex128`` while all
            # lower precision types use ``complex64``.
            if state.dtype == torch.float64:
                state = state.to(torch.complex128)
            else:
                state = state.to(torch.complex64)

        dtype = state.dtype
        device = state.device
        eps = torch.finfo(state.real.dtype).eps
        kraus = self._local_kraus(dtype, device)
        assert all(k.device == device for k in kraus), "Kraus/device mismatch"

        if trajectory:  # statevector
            batched = state.dim() > 1
            if not batched:
                state = state.unsqueeze(0)

            amps = torch.stack(
                [_apply_kraus_state(state, k, self._perm_cache) for k in kraus],
                dim=1,
            )  # (B, M, dim)
            probs = (amps.conj() * amps).sum(dim=-1).real.clamp_min(0)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)

            if rng is None:
                choices = torch.multinomial(probs, 1).squeeze(1)
            else:
                choices = torch.multinomial(probs, 1, generator=rng).squeeze(1)

            out = amps[torch.arange(amps.size(0), device=device), choices]
            norm = torch.linalg.vector_norm(out, dim=-1, keepdim=True).clamp_min(eps ** 0.5)
            out = out / norm
            if not batched:
                out = out.squeeze(0)
            return out

        # density matrix mode
        k_stack = torch.stack(kraus, dim=0)
        return _apply_kraus_density_all(state, k_stack, self._perm_cache)

    def to_kraus(
        self,
        *,
        dtype=torch.complex64,
        device=None,
        local: bool = False,
    ) -> List[torch.Tensor]:
        """Return the channel's Kraus operators.

        Parameters
        ----------
        dtype, device:
            Control the dtype and device of the returned tensors.
        local:
            If ``True``, return the cached, unembedded Kraus operators that act
            only on the target subsystem.  If ``False`` (default), embed the
            operators into the full ``n``-qubit space.  Global embedding
            constructs dense ``2**n`` matrices and is restricted to small
            systems.
        """


        device = device or torch.device("cpu")
        local_kraus = self._local_kraus(dtype, device)
        if local:
            return local_kraus

        if self.num_qubits > MAX_GLOBAL_KRAUS_QUBITS:
            warnings.warn(
                "Embedding Kraus operators globally constructs 2**n matrices and is intended for diagnostics on small systems; "
                f"n = {self.num_qubits} may be prohibitively expensive",
                UserWarning,
            )

        return [
            _embed_kraus_slow(k, self.targets, self.num_qubits, dtype=dtype, device=device)
            for k in local_kraus
        ]

    def to_superoperator(self, *, dtype=torch.complex64, device=None) -> torch.Tensor:
        """Return the local ``4^t x 4^t`` superoperator matrix."""

        return self.channel.to_superoperator(dtype=dtype, device=device)

    def to_global_superoperator(self, *, dtype=torch.complex64, device=None) -> torch.Tensor:
        """Return the full ``4^n x 4^n`` superoperator (diagnostic only)."""
        if self.num_qubits > 10:
            warnings.warn(
                "to_global_superoperator constructs a full 4**n matrix (O(4^n) memory) and is intended for diagnostics on small systems; "
                "n > 10 may be prohibitively expensive",
                UserWarning,
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            kraus = self.to_kraus(dtype=dtype, device=device)
        return sum(torch.kron(k, k.conj()) for k in kraus)

    def to_matrix(self, **kwargs) -> torch.Tensor:  # pragma: no cover - API compatibility
        raise NotImplementedError("Noise channels do not act linearly on statevectors; use to_superoperator().")

    def to_qasm(self) -> qasm.QasmRepresentation:  # pragma: no cover - trivial
        return qasm.QasmRepresentation(gate_str=f"// noise: {self.channel}")


# ---------------------------------------------------------------------------
# Actual Kraus based channels


class PhaseFlip(NoiseChannel):
    """Phase flip channel with probability ``p``.

    Kraus operators:

    ``K0 = sqrt(1-p) * I``
    ``K1 = sqrt(p) * Z``
    """

    def __init__(self, p: float, qubit: int):
        if not 0 <= p <= 1:
            raise ValueError("p must be in [0,1]")
        self.p = float(p)
        super().__init__([qubit])

    @property
    def is_identity(self) -> bool:
        return self.p == 0

    def __str__(self) -> str:  # pragma: no cover - tiny wrapper
        return f"PhaseFlip(p={self.p}){_format_targets(self.targets)}"

    def to_kraus(self, *, dtype=torch.complex64, device=None) -> List[torch.Tensor]:
        device = device or torch.device("cpu")
        eye = torch.eye(2, dtype=dtype, device=device)
        Z = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)
        p = torch.tensor(self.p, dtype=dtype, device=device)
        return [torch.sqrt(1 - p) * eye, torch.sqrt(p) * Z]


class Depolarizing(NoiseChannel):
    """Single-qubit depolarizing channel with error rate ``p``."""

    def __init__(self, p: float, qubit: int):
        if not 0 <= p <= 1:
            raise ValueError("p must be in [0,1]")
        self.p = float(p)
        super().__init__([qubit])

    @property
    def is_identity(self) -> bool:
        return self.p == 0

    def __str__(self) -> str:  # pragma: no cover - tiny wrapper
        return f"Depolarizing(p={self.p}){_format_targets(self.targets)}"

    def to_kraus(self, *, dtype=torch.complex64, device=None) -> List[torch.Tensor]:
        device = device or torch.device("cpu")
        eye = torch.eye(2, dtype=dtype, device=device)
        X = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
        Y = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device)
        Z = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)
        p = torch.tensor(self.p, dtype=dtype, device=device)
        s0 = torch.sqrt(1 - p)
        s = torch.sqrt(p / 3)
        return [s0 * eye, s * X, s * Y, s * Z]


class Dephasing(NoiseChannel):
    """Dephasing channel (phase damping) with rate ``gamma``.

    Kraus operators:
    ``K0 = sqrt(1-gamma) * I``
    ``K1 = sqrt(gamma) * |0><0|``
    ``K2 = sqrt(gamma) * |1><1|``

    Unlike :class:`PhaseFlip`, the off-diagonals are scaled by ``1-gamma``.
    """

    def __init__(self, gamma: float, qubit: int):
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be in [0,1]")
        self.gamma = float(gamma)
        super().__init__([qubit])

    @property
    def is_identity(self) -> bool:
        return self.gamma == 0

    def __str__(self) -> str:  # pragma: no cover - tiny wrapper
        return f"Dephasing(gamma={self.gamma}){_format_targets(self.targets)}"

    def to_kraus(self, *, dtype=torch.complex64, device=None) -> List[torch.Tensor]:
        device = device or torch.device("cpu")
        eye = torch.eye(2, dtype=dtype, device=device)
        P0 = torch.tensor([[1, 0], [0, 0]], dtype=dtype, device=device)
        P1 = torch.tensor([[0, 0], [0, 1]], dtype=dtype, device=device)
        g = torch.tensor(self.gamma, dtype=dtype, device=device)
        return [torch.sqrt(1 - g) * eye, torch.sqrt(g) * P0, torch.sqrt(g) * P1]


class AmplitudeDamping(NoiseChannel):
    """Amplitude damping channel with rate ``gamma``."""

    def __init__(self, gamma: float, qubit: int):
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be in [0,1]")
        self.gamma = float(gamma)
        super().__init__([qubit])

    @property
    def is_identity(self) -> bool:
        return self.gamma == 0

    def __str__(self) -> str:  # pragma: no cover - tiny wrapper
        return f"AmplitudeDamping(gamma={self.gamma}){_format_targets(self.targets)}"

    def to_kraus(self, *, dtype=torch.complex64, device=None) -> List[torch.Tensor]:
        device = device or torch.device("cpu")
        g = torch.tensor(self.gamma, dtype=dtype, device=device)
        k0 = torch.tensor([[1.0, 0.0], [0.0, torch.sqrt(1 - g)]], dtype=dtype, device=device)
        k1 = torch.tensor([[0.0, torch.sqrt(g)], [0.0, 0.0]], dtype=dtype, device=device)
        return [k0, k1]


class CorrelatedDepolarizing(NoiseChannel):
    """Two-qubit correlated depolarizing channel with probability ``p``."""

    def __init__(self, p: float, qubits: Sequence[int]):
        if not 0 <= p <= 1:
            raise ValueError("p must be in [0,1]")
        if len(qubits) != 2 or qubits[0] == qubits[1]:
            raise ValueError("two distinct qubits required")
        self.p = float(p)
        super().__init__(qubits)

    @property
    def is_identity(self) -> bool:
        return self.p == 0

    def __str__(self) -> str:  # pragma: no cover - tiny wrapper
        return f"CorrelatedDepolarizing(p={self.p}){_format_targets(self.targets)}"

    def to_kraus(self, *, dtype=torch.complex64, device=None) -> List[torch.Tensor]:
        device = device or torch.device("cpu")
        eye = torch.eye(2, dtype=dtype, device=device)
        X = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
        Y = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device)
        Z = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)
        p = torch.tensor(self.p, dtype=dtype, device=device)
        s0 = torch.sqrt(1 - p)
        s = torch.sqrt(p / 3)
        return [
            s0 * torch.kron(eye, eye),
            s * torch.kron(X, X),
            s * torch.kron(Y, Y),
            s * torch.kron(Z, Z),
        ]


# Backwards compatibility alias ------------------------------------------------
PhaseDamping = Dephasing


