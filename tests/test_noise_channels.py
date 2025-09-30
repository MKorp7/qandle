"""Analytic regression tests for Kraus-based noise channels."""

from __future__ import annotations

import math

import pytest
import torch

from qandle.noise.channels import (
    AmplitudeDamping,
    CorrelatedDepolarizing,
    Dephasing,
    Depolarizing,
    PhaseFlip,
)


_COMPLEX = torch.complex128


def _dagger(mat: torch.Tensor) -> torch.Tensor:
    return mat.conj().transpose(-1, -2)


def _random_pure_state(num_qubits: int, *, dtype=_COMPLEX, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    dim = 1 << num_qubits
    real = torch.randn(dim, generator=gen)
    imag = torch.randn(dim, generator=gen)
    vec = (real + 1j * imag).to(dtype)
    vec = vec / torch.linalg.vector_norm(vec)
    return vec


def _density_from_state(psi: torch.Tensor) -> torch.Tensor:
    return torch.outer(psi, psi.conj())


def _assert_cptp(kraus: list[torch.Tensor]) -> None:
    dim = kraus[0].shape[-1]
    ident = torch.eye(dim, dtype=kraus[0].dtype, device=kraus[0].device)
    accum = sum(_dagger(k) @ k for k in kraus)
    assert torch.allclose(accum, ident, atol=1e-12)


@pytest.mark.parametrize(
    "channel",
    [
        PhaseFlip(0.17, 0),
        Depolarizing(0.23, 0),
        Dephasing(0.41, 0),
        AmplitudeDamping(0.37, 0),
    ],
)
def test_single_qubit_kraus_are_cptp(channel) -> None:
    _assert_cptp(channel.to_kraus(dtype=_COMPLEX))


def test_correlated_depolarizing_kraus_are_cptp() -> None:
    channel = CorrelatedDepolarizing(0.29, (0, 1))
    _assert_cptp(channel.to_kraus(dtype=_COMPLEX))


def test_phaseflip_limits() -> None:
    plus = torch.tensor(
        [[0.5, 0.5], [0.5, 0.5]],
        dtype=_COMPLEX,
    )
    identity = PhaseFlip(0.0, 0).build(1)
    flip = PhaseFlip(1.0, 0).build(1)
    assert torch.allclose(identity(plus), plus)

    z = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=_COMPLEX)
    expected = z @ plus @ z
    assert torch.allclose(flip(plus), expected)


def test_dephasing_limits() -> None:
    plus = torch.tensor(
        [[0.5, 0.5], [0.5, 0.5]],
        dtype=_COMPLEX,
    )
    identity = Dephasing(0.0, 0).build(1)
    fully = Dephasing(1.0, 0).build(1)
    assert torch.allclose(identity(plus), plus)

    expected = torch.diag_embed(torch.diagonal(plus))
    assert torch.allclose(fully(plus), expected)


@pytest.mark.parametrize("p", [0.0, 0.42, 1.0])
def test_depolarizing_matches_pauli_average(p: float) -> None:
    rho = torch.tensor(
        [[0.6, 0.2 + 0.1j], [0.2 - 0.1j, 0.4]],
        dtype=_COMPLEX,
    )
    channel = Depolarizing(p, 0).build(1)
    out = channel(rho)

    x = torch.tensor([[0, 1], [1, 0]], dtype=_COMPLEX)
    y = torch.tensor([[0, -1j], [1j, 0]], dtype=_COMPLEX)
    z = torch.tensor([[1, 0], [0, -1]], dtype=_COMPLEX)
    expected = (1 - p) * rho + (p / 3.0) * (x @ rho @ x + y @ rho @ y + z @ rho @ z)
    assert torch.allclose(out, expected)


@pytest.mark.parametrize("gamma", [0.0, 0.53, 1.0])
def test_amplitude_damping_expected_action(gamma: float) -> None:
    rho_excited = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=_COMPLEX)
    channel = AmplitudeDamping(gamma, 0).build(1)
    out = channel(rho_excited)

    expected = torch.tensor([[gamma, 0.0], [0.0, 1 - gamma]], dtype=_COMPLEX)
    assert torch.allclose(out, expected)


@pytest.mark.parametrize("p", [0.0, 0.21, 1.0])
def test_correlated_depolarizing_expected_action(p: float) -> None:
    rho = torch.zeros((4, 4), dtype=_COMPLEX)
    rho[0, 0] = 1.0  # |00><00|
    channel = CorrelatedDepolarizing(p, (0, 1)).build(2)
    out = channel(rho)

    x = torch.tensor([[0, 1], [1, 0]], dtype=_COMPLEX)
    y = torch.tensor([[0, -1j], [1j, 0]], dtype=_COMPLEX)
    z = torch.tensor([[1, 0], [0, -1]], dtype=_COMPLEX)

    expected = (1 - p) * rho
    for op in (x, y, z):
        kron = torch.kron(op, op)
        expected = expected + (p / 3.0) * (kron @ rho @ _dagger(kron))
    assert torch.allclose(out, expected)


@pytest.mark.parametrize(
    "channel_factory, num_qubits, param",
    [
        (lambda p: PhaseFlip(p, 0), 1, 0.18),
        (lambda p: Depolarizing(p, 0), 1, 0.27),
        (lambda p: Dephasing(p, 0), 1, 0.63),
        (lambda p: AmplitudeDamping(p, 0), 1, 0.45),
        (lambda p: CorrelatedDepolarizing(p, (0, 1)), 2, 0.33),
    ],
)
def test_density_trace_preserved(channel_factory, num_qubits, param) -> None:
    psi = _random_pure_state(num_qubits, seed=5)
    rho = _density_from_state(psi)
    batch = torch.stack([rho, rho], dim=0)
    channel = channel_factory(param).build(num_qubits)
    out = channel(batch)
    traces = torch.diagonal(out, dim1=-2, dim2=-1).sum(-1)
    assert torch.allclose(traces, torch.ones_like(traces), atol=1e-7)


def test_trajectory_sampling_reproducible() -> None:
    psi = _random_pure_state(1, seed=123)
    channel = Depolarizing(0.6, 0).build(1)
    gen = torch.Generator().manual_seed(11)
    out1 = channel(psi, trajectory=True, rng=gen)
    gen = torch.Generator().manual_seed(11)
    out2 = channel(psi, trajectory=True, rng=gen)
    assert torch.allclose(out1, out2)
    assert math.isclose(torch.linalg.vector_norm(out1).item(), 1.0, rel_tol=1e-6)
