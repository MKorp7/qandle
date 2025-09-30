import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .. import utils
from . import QuantumBackend


_SWAP_BASE = torch.tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=torch.float32,
)


def svd_truncate(theta: torch.Tensor,
                 max_D: int,
                 eps: float = 1e-10) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """SVD with optional bond truncation.

    Returns
    -------
    U : torch.Tensor
        Left unitary.
    S : torch.Tensor
        Kept singular values.
    Vh : torch.Tensor
        Right unitary.
    discarded : float
        Sum of squared discarded singular values.
    """
    U, S, Vh = torch.linalg.svd(theta, full_matrices=False)
    keep = (S > eps) & (torch.arange(S.numel(), device=S.device) < max_D)
    discarded = (S[~keep] ** 2).sum().real.item() if (~keep).any() else 0.0
    return U[:, keep], S[keep], Vh[keep], discarded


@dataclass
class MPSTensor:
    data: torch.Tensor  # shape (Dl, 2, Dr)


class MPSBackend(QuantumBackend):
    """MPS simulator (canonical right-normal form)."""

    def __init__(self,
                 n_qubits: int,
                 dtype=torch.complex64,
                 device="cpu",
                 max_bond_dim: int = 64,
                 auto_swap: bool = True):
        self.dtype = dtype
        self.device = device
        self.max_D = max_bond_dim
        self.auto_swap = auto_swap
        self.trunc_error = 0.0
        self.max_D_used = 1
        self._swap_cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
        basis = F.one_hot(
            torch.tensor(0, device=device, dtype=torch.long), num_classes=2
        ).to(dtype=dtype).reshape(1, 2, 1)
        self.tensors: list[MPSTensor] = [MPSTensor(basis.clone()) for _ in range(n_qubits)]
        self.perm = list(range(n_qubits))
        self.invperm = list(range(n_qubits))

    def allocate(self, n_qubits: int):  # already done in __init__
        self.perm = list(range(n_qubits))
        self.invperm = list(range(n_qubits))
        return self

    @property
    def n_qubits(self) -> int:
        return len(self.tensors)

    @property
    def state(self) -> torch.Tensor:
        """Materialise the full statevector for interoperability hooks."""

        return self._to_statevector()

    @state.setter
    def state(self, state: torch.Tensor) -> None:
        """Reinitialise the MPS from a statevector after noise updates."""

        self._from_statevector(state)

    def apply_1q(self, gate: torch.Tensor, q: int):
        if gate.shape[0] != 2:
            n = int(math.log2(gate.shape[0]))
            idx0 = 0
            idx1 = 1 << (n - q - 1)
            gate = gate[[idx0, idx1]][:, [idx0, idx1]]
        site = self.perm[q]
        T = self.tensors[site].data  # (Dl,2,Dr)
        gate = gate.to(dtype=self.dtype, device=self.device)
        self.tensors[site].data = torch.einsum("ab, lbr -> lar", gate, T)

    def _get_swap_gate(self, *, dtype: torch.dtype | None = None, device: torch.device | str | None = None) -> torch.Tensor:
        key = (torch.device(device or self.device), dtype or self.dtype)
        if key not in self._swap_cache:
            self._swap_cache[key] = _SWAP_BASE.to(device=key[0], dtype=key[1])
        return self._swap_cache[key]

    def _apply_adjacent_2q(self, gate: torch.Tensor, q1: int, q2: int) -> None:
        if abs(q2 - q1) != 1:
            raise ValueError("_apply_adjacent_2q expects neighbouring qubits")
        left, right = (q1, q2) if q1 < q2 else (q2, q1)
        swap_required = q1 > q2
        A, B = self.tensors[left].data, self.tensors[right].data
        Dl, Dr = A.shape[0], B.shape[2]
        theta = torch.einsum("lab, bcr -> lacr", A, B)  # Dl,2,2,Dr
        if swap_required:
            theta = theta.permute(0, 2, 1, 3)
        theta = theta.reshape(Dl, 4, Dr)  # merge physical legs
        theta = torch.einsum("pq, lqr -> lpr", gate.to(theta), theta)
        theta = theta.reshape(Dl, 2, 2, Dr)
        if swap_required:
            theta = theta.permute(0, 2, 1, 3)
        theta = theta.reshape(Dl * 2, 2 * Dr)  # prepare SVD
        U, S, Vh, disc = svd_truncate(theta, self.max_D)
        D_new = S.numel()
        self.max_D_used = max(self.max_D_used, D_new)
        self.trunc_error += disc
        U = U.reshape(Dl, 2, D_new)
        Vh = (torch.diag(S).to(Vh) @ Vh).reshape(D_new, 2, Dr)
        self.tensors[left].data = U
        self.tensors[right].data = Vh

    def apply_2q(self, gate: torch.Tensor, q1: int, q2: int):
        if q1 == q2:
            raise ValueError("Cannot apply a 2-qubit gate on the same qubit twice")
        if gate.shape[0] != 4:
            n = int(math.log2(gate.shape[0]))
            idx = lambda b1, b2: ((b1 << (n - q1 - 1)) | (b2 << (n - q2 - 1)))
            sel = [idx(0, 0), idx(0, 1), idx(1, 0), idx(1, 1)]
            gate = gate[sel][:, sel]
        gate = gate.to(dtype=self.dtype, device=self.device)
        site1 = self.perm[q1]
        site2 = self.perm[q2]
        if abs(site2 - site1) == 1:
            self._apply_adjacent_2q(gate, site1, site2)
            return
        if not self.auto_swap:
            raise AssertionError("non-adjacent 2-qubit gate not supported (insert SWAPs)")
        swap_gate = self._get_swap_gate(dtype=self.dtype, device=self.device)
        left, right = (q1, q2) if q1 < q2 else (q2, q1)
        swap_positions = list(range(right - 1, left, -1))
        for pos in swap_positions:
            self._apply_adjacent_2q(swap_gate, pos, pos + 1)
        if q1 < q2:
            self._apply_adjacent_2q(gate, q1, q1 + 1)
        else:
            self._apply_adjacent_2q(gate, left + 1, left)
        for pos in reversed(swap_positions):
            self._apply_adjacent_2q(swap_gate, pos, pos + 1)


    def _swap_sites_internally(self, i: int):
        self.tensors[i], self.tensors[i + 1] = self.tensors[i + 1], self.tensors[i]
        li, lj = self.invperm[i], self.invperm[i + 1]
        self.invperm[i], self.invperm[i + 1] = lj, li
        self.perm[li], self.perm[lj] = i + 1, i

    def _left_environments(self) -> list[torch.Tensor]:
        Ls: list[torch.Tensor] = [torch.ones(1, 1, dtype=self.dtype, device=self.device)]
        for i in range(self.n_qubits - 1):
            T = self.tensors[i].data
            L = Ls[-1]
            A = T.permute(1, 0, 2)
            LA = torch.einsum("ij,bjk->bik", L, A)
            accum = torch.einsum("bij,bjk->ik", A.conj().permute(0, 2, 1), LA)
            Ls.append(accum)
        return Ls

    def _right_environments(self) -> list[torch.Tensor]:
        Rs: list[torch.Tensor] = [torch.ones(1, 1, dtype=self.dtype, device=self.device)]
        for i in range(self.n_qubits - 1, 0, -1):
            T = self.tensors[i].data
            R = Rs[-1]
            A = T.permute(1, 0, 2)
            AR = torch.einsum("bij,jk->bik", A, R)
            accum = torch.einsum("bij,bjk->ik", AR, A.conj().permute(0, 2, 1))
            Rs.append(accum)
        Rs.reverse()
        return Rs

    def _measure_single_qubit(self, q: int) -> torch.Tensor:
        site = self.perm[q]
        Ls = self._left_environments()
        Rs = self._right_environments()
        T = self.tensors[site].data
        L = Ls[site]
        R = Rs[site]
        A = T.permute(1, 0, 2)
        LA = torch.einsum("ij,bjk->bik", L, A)
        LAR = torch.einsum("bij,jk->bik", LA, R)
        out = torch.einsum("bij,bij->b", LAR, A.conj()).real
        out = torch.clamp(out, min=0.0)
        norm = out.sum()
        inv_norm = torch.where(norm > 0, norm.reciprocal(), torch.zeros_like(norm))
        return out * inv_norm

    def measure(self, qubits=None):
        if qubits is None:
            requested = list(range(n))
        elif isinstance(qubits, int):
            requested = [qubits]
        else:
            requested = list(qubits)

        if len(requested) != len(set(requested)):
            raise ValueError("Measurement qubits must be unique")

        prob_dtype = torch.empty((), dtype=self.dtype).real.dtype

        if not requested:
            return torch.ones(1, dtype=prob_dtype, device=self.device)

        ordered = sorted(requested)
        meas_index = {q: idx for idx, q in enumerate(ordered)}

        envs: dict[int, torch.Tensor] = {
            0: torch.eye(1, dtype=self.dtype, device=self.device)
        }

        for site, tensor in enumerate(self.tensors):
            bit_pos = meas_index.get(site)
            if bit_pos is not None:  # branching on this qubit
                new_envs: dict[int, torch.Tensor] = {}
                for outcome, env in envs.items():
                    for bit in (0, 1):
                        A = tensor.data[:, bit, :]
                        new_env = A.conj().T @ env @ A
                        key = outcome | (bit << bit_pos)
                        new_envs[key] = new_env
                envs = new_envs
            else:  # trace over physical index
                for outcome, env in list(envs.items()):
                    A0 = tensor.data[:, 0, :]
                    A1 = tensor.data[:, 1, :]
                    envs[outcome] = (
                        A0.conj().T @ env @ A0 + A1.conj().T @ env @ A1
                    )

        probs_sorted = torch.zeros(2 ** len(ordered), dtype=prob_dtype, device=self.device)
        for outcome, env in envs.items():
            probs_sorted[outcome] = torch.trace(env).real

        total = probs_sorted.sum()
        if total != 0:
            probs_sorted = probs_sorted / total  # normalise

        if requested == ordered:
            return probs_sorted

        perm = [requested.index(q) for q in ordered]
        probs = torch.zeros_like(probs_sorted)
        for src_index, value in enumerate(probs_sorted):
            dst_index = 0
            for bit_pos, out_pos in enumerate(perm):
                bit = (src_index >> bit_pos) & 1
                dst_index |= bit << out_pos
            probs[dst_index] = value
        return probs


    def _to_statevector(self) -> torch.Tensor:
        """Exact contraction → state vector (small n for testing)."""
        psi = self.tensors[0].data.squeeze(0)  # (2, D1)
        for t in self.tensors[1:]:
            psi = torch.einsum("aR, RbS -> abS", psi, t.data).reshape(-1, t.data.shape[2])
        psi = psi.squeeze(1)
        if self.perm != list(range(self.n_qubits)):
            n = self.n_qubits
            logical_indices = torch.arange(1 << n, device=psi.device)
            bit_shifts = torch.arange(n, device=psi.device)
            bits = ((logical_indices.unsqueeze(1) >> bit_shifts) & 1)
            invperm = torch.tensor(self.invperm, device=psi.device)
            phys_bits = bits[:, invperm]
            weights = (1 << torch.arange(n - 1, -1, -1, device=psi.device, dtype=torch.int64))
            phys_idx = (phys_bits.to(weights.dtype) * weights).sum(dim=1)
            psi = psi[phys_idx.long()]
        return psi

    def _from_statevector(self, state: torch.Tensor) -> None:
        """Decompose ``state`` into an MPS in right-canonical form."""

        n = self.n_qubits
        dim = 1 << n
        state = torch.as_tensor(state, dtype=self.dtype, device=self.device)
        if state.dim() != 1 or state.numel() != dim:
            raise ValueError(
                f"Statevector must have shape ({dim},), received {tuple(state.shape)}"
            )

        psi = state.reshape(1, dim)
        tensors: list[MPSTensor] = []
        trunc_accum = float(self.trunc_error)
        max_bond = max(1, int(self.max_D_used))

        for site in range(n - 1):
            Dl = psi.shape[0]
            theta = psi.reshape(Dl * 2, -1)
            U, S, Vh, disc = svd_truncate(theta, self.max_D)
            if S.numel() == 0:
                U = U[:, :1]
                S = S.new_zeros(1)
                Vh = Vh.new_zeros(1, Vh.shape[1], dtype=theta.dtype, device=theta.device)
            D_new = S.numel()
            trunc_accum += disc
            max_bond = max(max_bond, D_new)
            data = U.reshape(Dl, 2, D_new).to(dtype=self.dtype, device=self.device)
            tensors.append(MPSTensor(data))
            psi = (S.to(Vh.dtype).unsqueeze(1) * Vh)

        final_tensor = psi.reshape(psi.shape[0], 2, 1).to(dtype=self.dtype, device=self.device)
        tensors.append(MPSTensor(final_tensor))
        max_bond = max(max_bond, final_tensor.shape[0])

        self.tensors = tensors
        self.perm = list(range(n))
        self.invperm = list(range(n))
        self.trunc_error = trunc_accum
        self.max_D_used = max_bond

    @property
    def truncation_error(self) -> float:
        """Cumulative discarded weight from all SVD truncations."""
        return float(self.trunc_error)

    @property
    def max_bond_used(self) -> int:
        """Largest bond dimension encountered during simulation."""
        return int(self.max_D_used)
