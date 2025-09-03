import pytest
import torch
import qandle
from qandle import Circuit
import random

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