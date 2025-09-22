import torch, math
from dataclasses import dataclass
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
        self.tensors: list[MPSTensor] = [
            MPSTensor(torch.zeros(1, 2, 1, dtype=dtype, device=device))
            for _ in range(n_qubits)
        ]
        # |0> initialisation
        for t in self.tensors:
            t.data[0, 0, 0] = 1.0
        self.perm = list(range(n_qubits))
        self.invperm = list(range(n_qubits))

    def allocate(self, n_qubits: int):  # already done in __init__
        self.perm = list(range(n_qubits))
        self.invperm = list(range(n_qubits))
        return self

    @property
    def n_qubits(self) -> int:
        return len(self.tensors)

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
        left_site = min(site1, site2)
        right_site = max(site1, site2)
        for pos in range(right_site - 1, left_site, -1):
            self._swap_sites_internally(pos)
        site1 = self.perm[q1]
        site2 = self.perm[q2]
        self._apply_adjacent_2q(gate, site1, site2)

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
            accum = torch.zeros(T.shape[2], T.shape[2], dtype=self.dtype, device=self.device)
            for p in (0, 1):
                A = T[:, p, :]
                accum += A.conj().transpose(0, 1) @ L @ A
            Ls.append(accum)
        return Ls

    def _right_environments(self) -> list[torch.Tensor]:
        Rs: list[torch.Tensor] = [torch.ones(1, 1, dtype=self.dtype, device=self.device)]
        for i in range(self.n_qubits - 1, 0, -1):
            T = self.tensors[i].data
            R = Rs[-1]
            accum = torch.zeros(T.shape[0], T.shape[0], dtype=self.dtype, device=self.device)
            for p in (0, 1):
                A = T[:, p, :]
                accum += A @ R @ A.conj().transpose(0, 1)
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
        out = torch.zeros(2, dtype=self.dtype, device=self.device)
        for b in (0, 1):
            A = T[:, b, :]
            tmp = L @ A
            tmp = tmp @ R
            out[b] = torch.sum(tmp * A.conj())
        out = out.real
        out = torch.clamp(out, min=0.0)
        s = out.sum()
        if s.abs() > 0:
            out = out / s
        return out

    def _measure_subset_on_sites(self, sites: list[int]) -> torch.Tensor:
        if not sites:
            return torch.ones(1, dtype=self.dtype, device=self.device).real
        site_set = set(sites)
        envs: dict[int, torch.Tensor] = {
            0: torch.eye(1, dtype=self.dtype, device=self.device)
        }
        meas_count = 0
        for idx, tensor in enumerate(self.tensors):
            if idx in site_set:
                new_envs: dict[int, torch.Tensor] = {}
                for outcome, env in envs.items():
                    for bit in (0, 1):
                        A = tensor.data[:, bit, :]
                        next_env = A.conj().transpose(0, 1) @ env @ A
                        key = outcome | (bit << meas_count)
                        new_envs[key] = next_env
                envs = new_envs
                meas_count += 1
            else:
                for outcome, env in list(envs.items()):
                    A0 = tensor.data[:, 0, :]
                    A1 = tensor.data[:, 1, :]
                    envs[outcome] = (
                        A0.conj().transpose(0, 1) @ env @ A0
                        + A1.conj().transpose(0, 1) @ env @ A1
                    )
        probs = torch.zeros(2 ** meas_count, dtype=self.dtype, device=self.device)
        for outcome, env in envs.items():
            probs[outcome] = torch.trace(env)
        probs = probs.real
        probs = torch.clamp(probs, min=0.0)
        s = probs.sum()
        if s > 0:
            probs = probs / s
        return probs

    def _measure_subset(self, qubits: list[int]) -> torch.Tensor:
        pairs = [(q, self.perm[q]) for q in qubits]
        sorted_pairs = sorted(pairs, key=lambda x: x[1])
        sorted_sites = [site for _, site in sorted_pairs]
        probs = self._measure_subset_on_sites(sorted_sites)
        if not sorted_pairs:
            return probs
        pos_map = {logical: idx for idx, (logical, _) in enumerate(sorted_pairs)}
        k = len(qubits)
        permuted = torch.zeros_like(probs)
        for idx in range(1 << k):
            new_idx = 0
            for bit_pos, logical in enumerate(qubits):
                sorted_pos = pos_map[logical]
                bit = (idx >> sorted_pos) & 1
                new_idx |= bit << bit_pos
            permuted[new_idx] = probs[idx]
        return permuted

    def measure(self, qubits=None):
        if qubits is None:
            if self.n_qubits <= 16:
                full = self._to_statevector()
                probs = (full.abs() ** 2).real
                s = probs.sum()
                if s > 0:
                    probs = probs / s
                return probs
            outs = [self._measure_single_qubit(q) for q in range(self.n_qubits)]
            return torch.stack(outs, dim=0)

        if isinstance(qubits, int):
            return self._measure_single_qubit(qubits)

        qubits = list(qubits)
        if len(qubits) == 0:
            return torch.ones(1, dtype=self.dtype, device=self.device).real
        if len(qubits) == 1:
            return self._measure_single_qubit(qubits[0])

        return self._measure_subset(qubits)

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

    @property
    def truncation_error(self) -> float:
        """Cumulative discarded weight from all SVD truncations."""
        return float(self.trunc_error)

    @property
    def max_bond_used(self) -> int:
        """Largest bond dimension encountered during simulation."""
        return int(self.max_D_used)