import torch

from qandle import qcircuit, utils
from qandle import operators
from qandle.noise.channels import PhaseFlip
from qandle import utils_gates


def _zero_state(num_qubits: int) -> torch.Tensor:
    state = torch.zeros(2 ** num_qubits, dtype=torch.complex64)
    state[0] = 1
    return state


def _random_state(num_qubits: int) -> torch.Tensor:
    real = torch.randn(2 ** num_qubits)
    imag = torch.randn(2 ** num_qubits)
    state = torch.complex(real, imag)
    state = state / state.norm()
    return state


def test_merge_consecutive_rx_layers():
    rx1 = operators.RX(0, theta=torch.tensor(0.3))
    rx2 = operators.RX(0, theta=torch.tensor(0.2))
    circuit = qcircuit.Circuit([rx1, rx2], num_qubits=1).circuit

    assert len(circuit.layers) == 1
    fused = circuit.layers[0]
    assert isinstance(fused, qcircuit.FusedSingleQubitRotation)
    assert len(fused.layers) == 2

    state = _zero_state(1)
    ref_rx1 = operators.RX(0, theta=torch.tensor(0.3)).build(num_qubits=1)
    ref_rx2 = operators.RX(0, theta=torch.tensor(0.2)).build(num_qubits=1)

    expected_state = ref_rx2(ref_rx1(state.clone()))
    fused_state = fused(state.clone())
    assert torch.allclose(fused_state, expected_state, atol=1e-6)

    expected_matrix = utils.reduce_dot(ref_rx2.to_matrix(), ref_rx1.to_matrix())
    fused_matrix = fused.to_matrix()
    assert torch.allclose(fused_matrix, expected_matrix, atol=1e-6)


def test_named_rotations_preserve_metadata():
    rz1 = operators.RZ(0, name="alpha")
    rz2 = operators.RZ(0, name="beta")
    circuit = qcircuit.Circuit([rz1, rz2], num_qubits=1).circuit

    assert len(circuit.layers) == 1
    fused = circuit.layers[0]
    assert isinstance(fused, qcircuit.FusedSingleQubitRotation)
    assert fused.named

    kwargs = {"alpha": torch.tensor(0.1), "beta": torch.tensor(0.4)}
    state = _zero_state(1)

    ref_rz1 = operators.RZ(0, name="alpha").build(num_qubits=1)
    ref_rz2 = operators.RZ(0, name="beta").build(num_qubits=1)
    expected_state = ref_rz2(ref_rz1(state.clone(), **kwargs), **kwargs)

    fused_state = fused(state.clone(), **kwargs)
    assert torch.allclose(fused_state, expected_state, atol=1e-6)

    fused_matrix = fused.to_matrix(**kwargs)
    expected_matrix = utils.reduce_dot(ref_rz2.to_matrix(**kwargs), ref_rz1.to_matrix(**kwargs))
    assert torch.allclose(fused_matrix, expected_matrix, atol=1e-6)


def test_double_pauli_is_cancelled():
    x1 = utils_gates.X(0, num_qubits=1)
    x2 = utils_gates.X(0, num_qubits=1)
    circuit = qcircuit.Circuit([x1, x2], num_qubits=1).circuit

    assert len(circuit.layers) == 0

    state = _random_state(1)
    result = circuit.forward(state.clone())
    assert torch.allclose(result, state, atol=1e-6)


def test_hadamard_pair_is_cancelled():
    h1 = utils_gates.H(0, num_qubits=1)
    h2 = utils_gates.H(0, num_qubits=1)
    circuit = qcircuit.Circuit([h1, h2], num_qubits=1).circuit

    assert len(circuit.layers) == 0

    state = _random_state(1)
    result = circuit.forward(state.clone())
    assert torch.allclose(result, state, atol=1e-6)


def test_duplicate_cnot_is_removed():
    cnot1 = operators.CNOT(0, 1)
    cnot2 = operators.CNOT(0, 1)
    circuit = qcircuit.Circuit([cnot1, cnot2], num_qubits=2).circuit

    assert len(circuit.layers) == 0

    state = _random_state(2)
    result = circuit.forward(state.clone())
    assert torch.allclose(result, state, atol=1e-6)


def test_noise_prevents_cross_qubit_fusion():
    rx1 = operators.RX(0, theta=torch.tensor(0.1))
    noise = PhaseFlip(0.2, 0).build(num_qubits=1)
    rx2 = operators.RX(0, theta=torch.tensor(0.3))

    circuit = qcircuit.Circuit([rx1, noise, rx2], num_qubits=1).circuit

    assert len(circuit.layers) == 3
    assert isinstance(circuit.layers[0], operators.BuiltRX)
    assert isinstance(circuit.layers[1], type(noise))
    assert isinstance(circuit.layers[2], operators.BuiltRX)

