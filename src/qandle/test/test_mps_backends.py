import pytest
import torch
import qandle
from qandle import Circuit
from qandle import MeasureJointProbability
import random


def _assert_backends_agree(gates, num_qubits, backend_kwargs=None):
    c = Circuit(gates, num_qubits=num_qubits)
    vec = c(backend="statevector")
    mps = c(backend="mps", backend_kwargs=backend_kwargs or {})
    torch.testing.assert_close(vec.state, mps._to_statevector())


def _assert_probabilities_agree(gates, num_qubits):
    c = Circuit(gates, num_qubits=num_qubits)
    vec = c(backend="statevector")
    mps = c(backend="mps")
    vec_probs = MeasureJointProbability()(vec.state)
    mps_probs = mps.measure()
    torch.testing.assert_close(vec_probs, mps_probs, atol=1e-6, rtol=1e-6)

def test_ghz_state_and_measure():

    gates = [
        qandle.H(0, num_qubits=3),
        qandle.CNOT(0, 1),
        qandle.CNOT(1, 2)
    ]
    c = Circuit(gates, num_qubits=3)
    vec = c(backend="statevector")
    mps = c(backend="mps")
    torch.testing.assert_close(vec.state, mps._to_statevector())
    torch.testing.assert_close(vec.measure(), mps.measure())

def test_measure_subsets():
    gates = [
        qandle.H(0, num_qubits=2),
        qandle.CNOT(1, 0),
        qandle.CNOT(0, 1),
        qandle.H(1, num_qubits=2)
    ]
    c = Circuit(gates, num_qubits=2)
    vec = c(backend="statevector")
    mps = c(backend="mps")
    for qs in (None, [0], [1], [0,1]):
        torch.testing.assert_close(vec.measure(qs), mps.measure(qs))

def test_state_vector_simulator_correctness():
    gates = [qandle.H(0, num_qubits=2), qandle.CNOT(0, 1)]
    c = Circuit(gates, num_qubits=2)
    direct_state = c()
    backend = c(backend="statevector")
    torch.testing.assert_close(direct_state, backend.state)
    torch.testing.assert_close(qandle.MeasureJointProbability()(direct_state),
                               backend.measure())


def test_measure_normalizes():
    gates = [
        qandle.H(0, num_qubits=3),
        qandle.CNOT(0, 1),
        qandle.CNOT(1, 2),
    ]
    c = Circuit(gates, num_qubits=3)
    mps = c(backend="mps")
    # break canonical normalisation by scaling the first tensor
    mps.tensors[0].data *= 2
    probs = mps.measure()
    assert torch.isclose(probs.sum(), torch.tensor(1.0))
    vec_probs = c(backend="statevector").measure()
    torch.testing.assert_close(probs, vec_probs)

def test_random_circuits_measurements_agree():
    single_gates = [qandle.H, qandle.X]
    for seed in range(5):
        rng = random.Random(seed)
        gates = []
        gate_count = rng.randint(1, 5)
        for _ in range(gate_count):
            gate_fn = rng.choice(single_gates)
            gates.append(gate_fn(0, num_qubits=2))
            gates.append(gate_fn(1, num_qubits=2))
        c = Circuit(gates, num_qubits=2)
        vec = c(backend="statevector")
        mps = c(backend="mps")
        for _ in range(3):
            subset = rng.choice([None, [0], [1]])
            vec_subset = [1, 0] if subset is None else [1 - q for q in subset]
            torch.testing.assert_close(vec.measure(vec_subset), mps.measure(subset))


def test_long_range_cnot_matches_statevector():
    gates = [
        qandle.H(0, num_qubits=5),
        qandle.CNOT(0, 4),
        qandle.CNOT(4, 1),
    ]
    _assert_probabilities_agree(gates, num_qubits=5)


def test_long_range_cz_matches_statevector():
    gates = [
        qandle.H(0, num_qubits=4),
        qandle.CNOT(0, 3),
        qandle.CZ(3, 1),
    ]
    _assert_probabilities_agree(gates, num_qubits=4)


def test_auto_swap_opt_out_requires_manual_swaps():
    gates = [
        qandle.H(0, num_qubits=4),
        qandle.CNOT(0, 3),
    ]
    with pytest.raises(AssertionError):
        _assert_backends_agree(gates, num_qubits=4, backend_kwargs={"auto_swap": False})


def test_mps_single_qubit_marginals_match_statevector_random():
    torch.manual_seed(0)
    n = 8
    gates = []
    for q in range(n):
        gates.append(qandle.H(q, num_qubits=n))
        gates.append(qandle.RX(q, theta=0.2 * (q + 1), num_qubits=n))
        gates.append(qandle.RZ(q, theta=0.15 * (q + 1), num_qubits=n))
    for i in range(0, n - 1, 2):
        gates.append(qandle.CNOT(i, i + 1))
    circuit = Circuit(gates, num_qubits=n)
    vec_backend = circuit(backend="statevector")
    mps_backend = circuit(backend="mps", backend_kwargs={"max_bond_dim": 64})
    amplitudes = vec_backend.state
    probs_full = (amplitudes.conj() * amplitudes).real
    for q in range(n):
        probs = torch.zeros(2, dtype=probs_full.dtype)
        for idx, pv in enumerate(probs_full):
            bit = (idx >> q) & 1
            probs[bit] += pv
        torch.testing.assert_close(mps_backend.measure(q), probs, atol=1e-5, rtol=1e-5)


def test_mps_subset_measurement_noncontiguous():
    n = 7
    gates = [
        qandle.H(0, num_qubits=n),
        qandle.CNOT(0, 3),
        qandle.RY(5, theta=0.4, num_qubits=n),
        qandle.CNOT(5, 2),
        qandle.RX(6, theta=0.7, num_qubits=n),
        qandle.CNOT(6, 1),
    ]
    circuit = Circuit(gates, num_qubits=n)
    vec_backend = circuit(backend="statevector")
    mps_backend = circuit(backend="mps")
    subset = [0, 2, 5, 6]
    full_probs = MeasureJointProbability()(vec_backend.state).reshape(-1)
    target = torch.zeros(2 ** len(subset), dtype=full_probs.dtype)
    for idx, pv in enumerate(full_probs):
        key = 0
        for bit_pos, q in enumerate(subset):
            key |= ((idx >> q) & 1) << bit_pos
        target[key] += pv
    torch.testing.assert_close(mps_backend.measure(subset), target, atol=1e-6, rtol=1e-6)


def test_mps_far_cnot_depth_preserves_state():
    n = 6
    gates = [
        qandle.H(0, num_qubits=n),
        qandle.H(5, num_qubits=n),
        qandle.CNOT(0, 5),
        qandle.RZ(2, theta=0.33, num_qubits=n),
        qandle.CNOT(1, 4),
    ]
    circuit = Circuit(gates, num_qubits=n)
    vec_backend = circuit(backend="statevector")
    mps_backend = circuit(backend="mps", backend_kwargs={"max_bond_dim": 128})
    torch.testing.assert_close(vec_backend.state, mps_backend._to_statevector(), atol=1e-5, rtol=1e-5)


def test_mps_truncation_stats_and_cap():
    n = 8
    gates = [qandle.H(i, num_qubits=n) for i in range(n)]
    gates += [qandle.CNOT(i, (i + 1) % n) for i in range(n - 1)]
    circuit = Circuit(gates, num_qubits=n)
    mps_backend = circuit(backend="mps", backend_kwargs={"max_bond_dim": 8})
    assert mps_backend.max_bond_used <= 8
    assert mps_backend.truncation_error >= 0.0