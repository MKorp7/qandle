import torch, math
from dataclasses import dataclass
from . import QuantumBackend


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
                 max_bond_dim: int = 64):
        self.dtype = dtype
        self.device = device
        self.max_D = max_bond_dim
        self.trunc_error = 0.0
        self.max_D_used = 1
        self.tensors: list[MPSTensor] = [
            MPSTensor(torch.zeros(1, 2, 1, dtype=dtype, device=device))
            for _ in range(n_qubits)
        ]
        # |0> initialisation
        for t in self.tensors:
            t.data[0, 0, 0] = 1.0

    def allocate(self, n_qubits: int):  # already done in __init__
        return self

    def apply_1q(self, gate: torch.Tensor, q: int):
        if gate.shape[0] != 2:
            n = int(math.log2(gate.shape[0]))
            idx0 = 0
            idx1 = 1 << (n - q - 1)
            gate = gate[[idx0, idx1]][:, [idx0, idx1]]
        T = self.tensors[q].data  # (Dl,2,Dr)
        self.tensors[q].data = torch.einsum("ab, lbr -> lar", gate.to(T), T)

    def apply_2q(self, gate: torch.Tensor, q1: int, q2: int):
        assert abs(q2 - q1) == 1, "non-adjacent 2-qubit gate not supported (insert SWAPs)"
        if gate.shape[0] != 4:
            n = int(math.log2(gate.shape[0]))
            idx = lambda b1, b2: ((b1 << (n - q1 - 1)) | (b2 << (n - q2 - 1)))
            sel = [idx(0, 0), idx(0, 1), idx(1, 0), idx(1, 1)]
            gate = gate[sel][:, sel]
        if q2 < q1:
            q1, q2 = q2, q1
        A, B = self.tensors[q1].data, self.tensors[q2].data
        Dl, Dr = A.shape[0], B.shape[2]
        theta = torch.einsum("lab, bcr -> lacr", A, B)  # Dl,2,2,Dr
        theta = theta.reshape(Dl, 4, Dr)  # merge physical legs
        theta = torch.einsum("pq, lqr -> lpr", gate.to(theta), theta)
        theta = theta.reshape(Dl * 2, 2 * Dr)  # prepare SVD
        U, S, Vh, disc = svd_truncate(theta, self.max_D)
        D_new = S.numel()
        self.max_D_used = max(self.max_D_used, D_new)
        self.trunc_error += disc
        norm = torch.sum(S ** 2).sqrt()
        if norm != 0:
            S = S / norm
        U = U.reshape(Dl, 2, D_new)
        Vh = (torch.diag(S).to(Vh) @ Vh).reshape(D_new, 2, Dr)
        self.tensors[q1].data = U
        self.tensors[q2].data = Vh

    def measure(self, qubits=None):
        """Return measurement probabilities for given qubits.
        Parameters
        ----------
        qubits:
            Sequence of qubits to measure. If ``None`` all qubits are
            measured and a full probability distribution is returned.
        """

        n = len(self.tensors)

        if qubits is None:
            qubits = list(range(n))
        elif isinstance(qubits, int):
            qubits = [qubits]

        qubits = list(qubits)
        qubits.sort()
        qset = set(qubits)

        envs: dict[int, torch.Tensor] = {
            0: torch.eye(1, dtype=self.tensors[0].data.dtype, device=self.device)
        }

        meas_count = 0
        for i, tensor in enumerate(self.tensors):
            if i in qset:  # branching on this qubit
                new_envs: dict[int, torch.Tensor] = {}
                for outcome, env in envs.items():
                    for bit in (0, 1):
                        A = tensor.data[:, bit, :]
                        new_env = A.conj().T @ env @ A
                        key = outcome | (bit << meas_count)
                        new_envs[key] = new_env
                envs = new_envs
                meas_count += 1
            else:  # trace over physical index
                for outcome, env in list(envs.items()):
                    A0 = tensor.data[:, 0, :]
                    A1 = tensor.data[:, 1, :]
                    envs[outcome] = (
                        A0.conj().T @ env @ A0 + A1.conj().T @ env @ A1
                    )

        probs = torch.zeros(
            2 ** len(qubits),
            dtype=self.tensors[0].data.real.dtype,
            device=self.device,
        )
        for outcome, env in envs.items():
            probs[outcome] = torch.trace(env).real

        total = probs.sum()
        if total != 0:
            probs = probs / total  # normalise
        return probs

    def _to_statevector(self) -> torch.Tensor:
        """Exact contraction → state vector (small n for testing)."""
        psi = self.tensors[0].data.squeeze(0)  # (2, D1)
        for t in self.tensors[1:]:
            psi = torch.einsum("aR, RbS -> abS", psi, t.data).reshape(-1, t.data.shape[2])
        return psi.squeeze(1)

    @property
    def truncation_error(self) -> float:
        """Cumulative discarded weight from all SVD truncations."""
        return float(self.trunc_error)

    @property
    def max_bond_used(self) -> int:
        """Largest bond dimension encountered during simulation."""
        return int(self.max_D_used)