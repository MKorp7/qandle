import math
import pytest
import torch

from qandle.noise import (
    PhaseFlip,
    Depolarizing,
    Dephasing,
    AmplitudeDamping,
    CorrelatedDepolarizing,
    BitFlip,
)


def rand_state(n: int) -> torch.Tensor:
    dim = 1 << n
    re = torch.randn(dim)
    im = torch.randn(dim)
    vec = torch.complex(re, im)
    vec = vec / torch.linalg.vector_norm(vec)
    return vec.to(torch.complex64)


def density_matrix(psi: torch.Tensor) -> torch.Tensor:
    return torch.outer(psi, psi.conj())


# ---------------------------------------------------------------------------
# Parameter validation


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (PhaseFlip, {"p": -0.1, "qubit": 0}),
        (PhaseFlip, {"p": 1.1, "qubit": 0}),
        (Depolarizing, {"p": -0.1, "qubit": 0}),
        (Depolarizing, {"p": 1.1, "qubit": 0}),
        (Dephasing, {"gamma": -0.1, "qubit": 0}),
        (Dephasing, {"gamma": 1.1, "qubit": 0}),
        (AmplitudeDamping, {"gamma": -0.1, "qubit": 0}),
        (AmplitudeDamping, {"gamma": 1.1, "qubit": 0}),
        (CorrelatedDepolarizing, {"p": -0.1, "qubits": (0, 1)}),
        (CorrelatedDepolarizing, {"p": 1.1, "qubits": (0, 1)}),
    ],
)
def test_parameter_validation(cls, kwargs):
    with pytest.raises(ValueError):
        cls(**kwargs)


def test_target_validation():
    with pytest.raises(ValueError):
        CorrelatedDepolarizing(p=0.1, qubits=(1, 1)).build(num_qubits=2)


def test_unsorted_targets_equivalent():
    psi = rand_state(2)
    rho = density_matrix(psi)
    ch1 = CorrelatedDepolarizing(p=0.2, qubits=(1, 0)).build(num_qubits=2)
    ch2 = CorrelatedDepolarizing(p=0.2, qubits=(0, 1)).build(num_qubits=2)
    assert torch.allclose(ch1(rho), ch2(rho))


# ---------------------------------------------------------------------------
# CPTP sanity


def test_kraus_cptp():
    channels = [
        PhaseFlip(p=0.2, qubit=0),
        Depolarizing(p=0.3, qubit=0),
        Dephasing(gamma=0.25, qubit=0),
        AmplitudeDamping(gamma=0.4, qubit=0),
        CorrelatedDepolarizing(p=0.2, qubits=(0, 1)),
    ]
    for ch in channels:
        ks = ch.to_kraus()
        acc = sum(k.conj().T @ k for k in ks)
        eye = torch.eye(ks[0].shape[1], dtype=torch.complex64)
        assert torch.allclose(acc, eye, atol=1e-6)


# ---------------------------------------------------------------------------
# Single-qubit behaviour


def test_phaseflip_plus_state():
    p = 0.37
    plus = torch.tensor([1.0, 1.0], dtype=torch.complex64) / math.sqrt(2)
    rho = density_matrix(plus)
    ch = PhaseFlip(p=p, qubit=0).build(num_qubits=1)
    out = ch(rho)
    assert torch.allclose(out[0, 1], torch.tensor((1 - 2 * p) / 2, dtype=torch.complex64))


def test_dephasing_plus_state():
    g = 0.4
    plus = torch.tensor([1.0, 1.0], dtype=torch.complex64) / math.sqrt(2)
    rho = density_matrix(plus)
    ch = Dephasing(gamma=g, qubit=0).build(num_qubits=1)
    out = ch(rho)
    assert torch.allclose(out[0, 1], torch.tensor((1 - g) / 2, dtype=torch.complex64))


def test_depolarizing_trace_and_limit():
    rho = torch.tensor([[0.6, 0.2], [0.2, 0.4]], dtype=torch.complex64)
    dep1 = Depolarizing(p=0.75, qubit=0).build(num_qubits=1)
    out1 = dep1(rho)
    assert torch.allclose(out1, torch.eye(2, dtype=torch.complex64) / 2, atol=1e-6)

    dep = Depolarizing(p=0.5, qubit=0).build(num_qubits=1)
    rand_rho = density_matrix(rand_state(1))
    out = dep(rand_rho)
    assert torch.allclose(torch.trace(out), torch.trace(rand_rho))


def test_amplitude_damping_populations():
    g = 0.3
    ch = AmplitudeDamping(gamma=g, qubit=0).build(num_qubits=1)
    rho1 = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.complex64)
    out1 = ch(rho1)
    assert torch.allclose(out1[0, 0].real, torch.tensor(g))
    assert torch.allclose(out1[1, 1].real, torch.tensor(1 - g))

    rho0 = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.complex64)
    out0 = ch(rho0)
    assert torch.allclose(out0, rho0)


@pytest.mark.parametrize(
    "real_dtype,complex_dtype",
    [
        (torch.float32, torch.complex64),
        (torch.float64, torch.complex128),
    ],
)
def test_real_dtype_promotes(real_dtype, complex_dtype):
    rho = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=real_dtype)
    ch = PhaseFlip(p=0.3, qubit=0).build(num_qubits=1)
    out = ch(rho)
    expected = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=complex_dtype)
    assert out.dtype == complex_dtype
    assert torch.allclose(out, expected)


# ---------------------------------------------------------------------------
# Two-qubit tests


def test_dephasing_bell_state():
    g = 0.25
    psi = torch.tensor([1, 0, 0, 1], dtype=torch.complex64) / math.sqrt(2)
    rho = density_matrix(psi)
    ch = Dephasing(gamma=g, qubit=0).build(num_qubits=2)
    out = ch(rho)
    assert torch.allclose(out[0, 3], torch.tensor((1 - g) / 2, dtype=torch.complex64))
    assert torch.allclose(out[3, 0], torch.tensor((1 - g) / 2, dtype=torch.complex64))


def test_correlated_depolarizing_invariance():
    psi = torch.tensor([1, 0, 0, 1], dtype=torch.complex64) / math.sqrt(2)
    rho = density_matrix(psi)
    ch = CorrelatedDepolarizing(p=1.0, qubits=(0, 1)).build(num_qubits=2)
    out = ch(rho)
    assert torch.allclose(out, rho, atol=1e-6)


def test_correlated_depolarizing_nonadjacent():
    p = 0.2
    psi = rand_state(3)
    rho = density_matrix(psi)
    ch = CorrelatedDepolarizing(p=p, qubits=(0, 2)).build(num_qubits=3)
    out = ch(rho)

    device = rho.device
    dtype = rho.dtype
    I = torch.eye(2, dtype=dtype, device=device)
    X = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)
    s0 = math.sqrt(1 - p)
    s = math.sqrt(p / 3)
    Ks = [
        s0 * torch.kron(torch.kron(I, I), I),
        s * torch.kron(torch.kron(X, I), X),
        s * torch.kron(torch.kron(Y, I), Y),
        s * torch.kron(torch.kron(Z, I), Z),
    ]
    expected = sum(K @ rho @ K.conj().T for K in Ks)
    assert torch.allclose(out, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Trajectory vs density matrix


@pytest.mark.parametrize(
    "builder,n,N",
    [
        (lambda: PhaseFlip(p=0.3, qubit=0), 1, 2000),
        (lambda: Depolarizing(p=0.2, qubit=0), 1, 2000),
        (lambda: Dephasing(gamma=0.4, qubit=0), 1, 2000),
        (lambda: AmplitudeDamping(gamma=0.4, qubit=0), 1, 3000),
        (lambda: CorrelatedDepolarizing(p=0.2, qubits=(0, 1)), 2, 3000),
    ],
)
def test_trajectory_matches_density(builder, n, N):
    channel = builder().build(num_qubits=n)
    psi = rand_state(n)
    rho = density_matrix(psi)
    exact = channel(rho)
    acc = torch.zeros_like(exact)
    gen = torch.Generator().manual_seed(0)
    for _ in range(N):
        out_state = channel(psi, trajectory=True, rng=gen)
        acc = acc + density_matrix(out_state)
    approx = acc / N
    err = torch.linalg.norm(approx - exact)
    assert err < 3e-2


def test_bitflip_channel():
    ch = BitFlip(p=0.5, qubit=0).build(num_qubits=1)
    state = torch.tensor([1.0, 0.0], dtype=torch.complex64)
    out = ch(state)
    expected = torch.tensor([math.sqrt(0.5), math.sqrt(0.5)], dtype=torch.complex64)
    assert torch.allclose(out, expected)


# ---------------------------------------------------------------------------
# Batch shape and dtype preservation


def test_batch_shapes_and_dtype():
    rho = torch.eye(2, dtype=torch.complex64).unsqueeze(0).repeat(3, 1, 1)
    ch = PhaseFlip(p=0.2, qubit=0).build(num_qubits=1)
    out = ch(rho)
    assert out.shape == rho.shape
    assert out.dtype == rho.dtype


def test_large_system_scaling():
    """Regression test on a larger system to ensure local application scales."""

    n = 10
    psi = rand_state(n)
    rho = density_matrix(psi)
    ch = PhaseFlip(p=0.3, qubit=3).build(num_qubits=n)
    out = ch(rho)
    assert out.shape == rho.shape
    assert torch.allclose(torch.trace(out), torch.tensor(1.0, dtype=torch.complex64))


def test_trajectory_batch_independence():
    ch = AmplitudeDamping(gamma=0.6, qubit=0).build(num_qubits=1)
    psi = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.complex64)
    gen = torch.Generator().manual_seed(0)
    out = ch(psi, trajectory=True, rng=gen)
    assert torch.allclose(out[0], torch.tensor([1.0, 0.0], dtype=torch.complex64))


def test_trajectory_dtype_device():
    psi = rand_state(1)
    ch = PhaseFlip(p=0.2, qubit=0).build(num_qubits=1)
    out = ch(psi, trajectory=True)
    assert out.dtype == psi.dtype
    assert out.device == psi.device


def test_trajectory_large_n():
    n = 18
    psi = rand_state(n)
    ch = PhaseFlip(p=0.3, qubit=0).build(num_qubits=n)
    out = ch(psi, trajectory=True)
    assert out.shape == psi.shape
    assert torch.allclose(torch.linalg.vector_norm(out), torch.tensor(1.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_channel_cuda():
    psi = rand_state(1).to("cuda")
    rho = density_matrix(psi)
    ch = PhaseFlip(p=0.2, qubit=0).build(num_qubits=1)
    out_rho = ch(rho)
    out_psi = ch(psi, trajectory=True)
    assert out_rho.device.type == "cuda"
    assert out_psi.device.type == "cuda"
