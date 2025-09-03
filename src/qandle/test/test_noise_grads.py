import torch
import qandle
from qandle.noise import PhaseFlip, AmplitudeDamping
import pytest


def _finite_diff(forward, param, eps=1e-4):
    """Finite-difference estimate of d forward / d param."""
    orig = param.detach().clone()
    with torch.no_grad():
        param.copy_(orig + eps)
        plus = forward()
        param.copy_(orig - eps)
        minus = forward()
        param.copy_(orig)
    return (plus - minus) / (2 * eps)


def _grad_check(channel):
    """Compare autograd gradient against finite-difference for *channel*."""
    gate = qandle.RY(qubit=0, theta=0.3).build(1)
    ch = channel.build(1)
    rho0 = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.complex64)

    def forward():
        U = gate.to_matrix()
        rho = U @ rho0 @ U.conj().T
        rho = ch(rho)
        return rho[0, 0].real

    out = forward()
    (grad,) = torch.autograd.grad(out, gate.theta)
    fd = _finite_diff(forward, gate.theta)
    assert torch.allclose(grad, fd, rtol=5e-3, atol=1e-5)


@pytest.mark.parametrize("channel", [PhaseFlip(0.2, 0), AmplitudeDamping(0.3, 0)])
def test_noise_gradients_match_finite_diff(channel):
    _grad_check(channel)