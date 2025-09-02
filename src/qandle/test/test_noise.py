import math
from unittest.mock import patch

import pytest
import torch

from qandle.noise import (
    NoiseChannel,
    PhaseFlip,
    Depolarizing,
    Dephasing,
    AmplitudeDamping,
    CorrelatedDepolarizing,
    BitFlip,
    NoiseChannel,
)


class DummyChannel(NoiseChannel):
    def to_kraus(self, *, dtype=torch.complex64, device=None):
        device = device or torch.device("cpu")
        I = torch.eye(2, dtype=dtype, device=device)
        return [I]

    def __str__(self) -> str:  # pragma: no cover - simple representation
        return "DummyChannel"


def rand_state(n: int) -> torch.Tensor:
    dim = 1 << n
    re = torch.randn(dim)
    im = torch.randn(dim)
    vec = torch.complex(re, im)
    vec = vec / torch.linalg.vector_norm(vec)
    return vec.to(torch.complex64)


def density_matrix(psi: torch.Tensor) -> torch.Tensor:
    return torch.outer(psi, psi.conj())


def is_tp(ks: list[torch.Tensor], atol: float = 1e-6) -> bool:
    """Check that Kraus operators form a CPTP map."""

    eye = torch.eye(ks[0].shape[1], dtype=ks[0].dtype)
    acc = sum(k.conj().T @ k for k in ks)
    if not torch.allclose(acc, eye, atol=atol):
        return False

    vecs = [k.reshape(-1, 1) for k in ks]
    choi = sum(v @ v.conj().T for v in vecs)
    evals = torch.linalg.eigvalsh(choi)
    return bool(torch.all(evals.real >= -atol))


# ---------------------------------------------------------------------------
# String representations


def test_noise_channel_str_and_qasm():
    channels = [
        BitFlip(p=0.1, qubit=0),
        PhaseFlip(p=0.2, qubit=1),
        Depolarizing(p=0.3, qubit=2),
        Dephasing(gamma=0.4, qubit=3),
        AmplitudeDamping(gamma=0.5, qubit=4),
        CorrelatedDepolarizing(p=0.6, qubits=(5, 6)),
    ]
    expected = [
        "BitFlip(p=0.1)[q0]",
        "PhaseFlip(p=0.2)[q1]",
        "Depolarizing(p=0.3)[q2]",
        "Dephasing(gamma=0.4)[q3]",
        "AmplitudeDamping(gamma=0.5)[q4]",
        "CorrelatedDepolarizing(p=0.6)[q5,q6]",
    ]
    for ch, exp in zip(channels, expected):
        assert str(ch) == exp
        assert ch.to_qasm().gate_str == f"// noise: {exp}"

        if isinstance(ch, BitFlip):
            targets = [ch.qubit]
        else:
            targets = ch.targets
        built = ch.build(num_qubits=max(targets) + 1)
        assert str(built) == exp
        assert built.to_qasm().gate_str == f"// noise: {exp}"


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
    with pytest.raises(ValueError) as exc:
        DummyChannel(targets=(1, 1)).build(num_qubits=2)
    msg = str(exc.value)
    assert "targets=(1, 1)" in msg
    assert "num_qubits=2" in msg


def test_target_range_validation_message():
    with pytest.raises(ValueError) as exc:
        DummyChannel(targets=(0, 2)).build(num_qubits=2)
    msg = str(exc.value)
    assert "targets=(0, 2)" in msg
    assert "num_qubits=2" in msg


def test_build_requires_targets():
    class DummyChannel(NoiseChannel):
        def __str__(self) -> str:  # pragma: no cover - trivial
            return "dummy"

        def to_kraus(self, *, dtype=torch.complex64, device=None):  # pragma: no cover - trivial
            device = device or torch.device("cpu")
            return [torch.eye(1, dtype=dtype, device=device)]

    with pytest.raises(ValueError, match="at least one target required"):
        DummyChannel(()).build(num_qubits=1)


def test_unsorted_targets_equivalent():
    psi = rand_state(2)
    rho = density_matrix(psi)
    ch1 = CorrelatedDepolarizing(p=0.2, qubits=(1, 0)).build(num_qubits=2)
    ch2 = CorrelatedDepolarizing(p=0.2, qubits=(0, 1)).build(num_qubits=2)
    assert torch.allclose(ch1(rho), ch2(rho))


def test_get_perm_cache_same_object():
    channels = pytest.importorskip("qandle.noise.channels")
    targets = (0, 2)
    n = 4
    cache1 = channels._get_perm_cache(targets, n)
    cache2 = channels._get_perm_cache(targets, n)
    assert cache1 is cache2


# ---------------------------------------------------------------------------
# CPTP sanity


@pytest.mark.parametrize(
    "channel",
    [
        PhaseFlip(p=0.2, qubit=0),
        Depolarizing(p=0.3, qubit=0),
        Dephasing(gamma=0.25, qubit=0),
        AmplitudeDamping(gamma=0.4, qubit=0),
        CorrelatedDepolarizing(p=0.2, qubits=(0, 1)),
    ],
)
def test_kraus_cptp(channel):
    assert is_tp(channel.to_kraus())


def test_built_kraus_local_and_global():
    unbuilt = PhaseFlip(p=0.2, qubit=0)
    built = unbuilt.build(num_qubits=2)

    # Local Kraus operators match the unbuilt channel
    expected_local = unbuilt.to_kraus()
    local = built.to_kraus(local=True)
    assert len(expected_local) == len(local)
    for a, b in zip(expected_local, local):
        assert torch.allclose(a, b)

    # Global embedding acts on the full space and matches the forward action
    global_kraus = built.to_kraus()
    assert all(k.shape == (4, 4) for k in global_kraus)
    rho = density_matrix(rand_state(2))
    out = sum(k @ rho @ k.conj().T for k in global_kraus)
    assert torch.allclose(out, built(rho))


def test_kraus_global_size_guard():
    ch = PhaseFlip(p=0.1, qubit=0).build(num_qubits=11)
    with pytest.raises(ValueError):
        ch.to_kraus()


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


@pytest.mark.parametrize(
    "builder,state",
    [
        (lambda: PhaseFlip(p=0.0, qubit=0), torch.eye(2, dtype=torch.float32)),
        (lambda: Depolarizing(p=0.0, qubit=0), torch.eye(2, dtype=torch.float32)),
        (lambda: Dephasing(gamma=0.0, qubit=0), torch.eye(2, dtype=torch.float32)),
        (lambda: AmplitudeDamping(gamma=0.0, qubit=0), torch.eye(2, dtype=torch.float32)),
        (lambda: CorrelatedDepolarizing(p=0.0, qubits=(0, 1)), torch.eye(4, dtype=torch.float32)),
    ],
)
def test_identity_channels_skip(builder, state):
    n = int(math.log2(state.shape[-1]))
    ch = builder().build(num_qubits=n)
    with patch.object(ch, "_local_kraus", side_effect=AssertionError("should not be called")) as spy:
        out = ch(state)
    spy.assert_not_called()
    assert out is state
    assert out.dtype == state.dtype


@pytest.mark.parametrize(
    "builder,state",
    [
        (lambda: PhaseFlip(p=0.0, qubit=0), torch.tensor([1.0, 0.0], dtype=torch.float32)),
        (lambda: Depolarizing(p=0.0, qubit=0), torch.tensor([1.0, 0.0], dtype=torch.float32)),
        (lambda: Dephasing(gamma=0.0, qubit=0), torch.tensor([1.0, 0.0], dtype=torch.float32)),
        (lambda: AmplitudeDamping(gamma=0.0, qubit=0), torch.tensor([1.0, 0.0], dtype=torch.float32)),
        (lambda: CorrelatedDepolarizing(p=0.0, qubits=(0, 1)), torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)),
    ],
)
def test_identity_channels_skip_trajectory(builder, state):
    n = int(math.log2(state.shape[-1]))
    ch = builder().build(num_qubits=n)
    with patch.object(ch, "_local_kraus", side_effect=AssertionError("should not be called")) as spy:
        out = ch(state, trajectory=True)
    spy.assert_not_called()
    assert out is state
    assert out.dtype == state.dtype


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
    eye = torch.eye(2, dtype=dtype, device=device)
    X = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)
    s0 = math.sqrt(1 - p)
    s = math.sqrt(p / 3)
    Ks = [
        s0 * torch.kron(torch.kron(eye, eye), eye),
        s * torch.kron(torch.kron(X, eye), X),
        s * torch.kron(torch.kron(Y, eye), Y),
        s * torch.kron(torch.kron(Z, eye), Z),
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


def test_trajectory_reproducible_with_seed():
    ch = PhaseFlip(p=0.3, qubit=0).build(num_qubits=1)
    psi = rand_state(1)
    gen = torch.Generator()
    gen.manual_seed(1234)
    out1 = ch(psi, trajectory=True, rng=gen)
    gen.manual_seed(1234)
    out2 = ch(psi, trajectory=True, rng=gen)
    assert torch.allclose(out1, out2)


def test_trajectory_dtype_device():
    psi = rand_state(1)
    ch = PhaseFlip(p=0.2, qubit=0).build(num_qubits=1)
    out = ch(psi, trajectory=True)
    assert out.dtype == psi.dtype
    assert out.device == psi.device
    assert torch.allclose(
        torch.linalg.vector_norm(out), torch.tensor(1.0, dtype=out.real.dtype)
    )


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


def test_to_kraus_warns_large_system(monkeypatch):
    ch = PhaseFlip(p=0.1, qubit=0).build(num_qubits=11)
    import qandle.noise.channels as channels

    monkeypatch.setattr(
        channels,
        "_embed_kraus_slow",
        lambda *args, **kwargs: torch.ones((1, 1), dtype=torch.complex64),
    )
    with pytest.warns(UserWarning):
        ch.to_kraus()


def test_to_global_superoperator_warns_large_system(monkeypatch):
    ch = PhaseFlip(p=0.1, qubit=0).build(num_qubits=11)
    import qandle.noise.channels as channels

    monkeypatch.setattr(
        channels,
        "_embed_kraus_slow",
        lambda *args, **kwargs: torch.ones((1, 1), dtype=torch.complex64),
    )
    with pytest.warns(UserWarning):
        ch.to_global_superoperator()

