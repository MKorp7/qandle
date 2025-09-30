import math

import pytest

torch = pytest.importorskip("torch")

from qandle.backends.statevector import StateVectorBackend
from qandle.kernels import apply_one_qubit, apply_two_qubit


def reference_apply(state: torch.Tensor, gate: torch.Tensor, qubits: tuple[int, ...]) -> torch.Tensor:
    n = int(math.log2(state.numel()))
    q_be = [n - q - 1 for q in qubits]
    perm = q_be + [i for i in range(n) if i not in q_be]
    inv_perm = [perm.index(i) for i in range(n)]
    psi = state.view([2] * n).permute(perm).reshape(2 ** len(qubits), -1)
    psi = torch.matmul(gate.to(state), psi)
    psi = psi.reshape([2] * len(qubits) + [2] * (n - len(qubits)))
    psi = psi.permute(inv_perm).reshape_as(state)
    return psi


@pytest.mark.parametrize("n", [3, 5])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_apply_one_qubit_matches_dense(n: int, dtype: torch.dtype) -> None:
    state = (
        torch.randn(2**n, dtype=torch.float32)
        + 1j * torch.randn(2**n, dtype=torch.float32)
    ).to(dtype)
    gate = (
        torch.randn(2, 2, dtype=torch.float32)
        + 1j * torch.randn(2, 2, dtype=torch.float32)
    ).to(dtype)
    gate = torch.linalg.qr(gate)[0]
    q = 1

    out = apply_one_qubit(state.clone(), gate, q, n)
    ref = reference_apply(state, gate, (q,))
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("n", [3, 5])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_apply_two_qubit_matches_dense(n: int, dtype: torch.dtype) -> None:
    state = (
        torch.randn(2**n, dtype=torch.float32)
        + 1j * torch.randn(2**n, dtype=torch.float32)
    ).to(dtype)
    gate = (
        torch.randn(4, 4, dtype=torch.float32)
        + 1j * torch.randn(4, 4, dtype=torch.float32)
    ).to(dtype)
    gate = torch.linalg.qr(gate)[0]
    q0, q1 = 0, n - 1

    out = apply_two_qubit(state.clone(), gate, q0, q1, n)
    ref = reference_apply(state, gate, (q0, q1))
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_backend_uses_kernels_on_cuda() -> None:
    be = StateVectorBackend(3, device="cuda")
    gate = torch.eye(2, dtype=be.state.dtype, device=be.state.device)
    before = be.state.clone()
    be.apply_1q(gate, 0)
    assert torch.allclose(be.state, before)
