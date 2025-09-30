import torch

from qandle import embeddings
from qandle import ansaetze
from qandle import operators as op


def test_zz_feature_map_matches_diagonal():
    x = torch.tensor([0.3, -0.5], dtype=torch.float64)
    feature_map = embeddings.ZZFeatureMap(
        n_qubits=2,
        edges=[(0, 1)],
        lambda_scale=0.7,
        eta_scale=0.2,
    )

    circuit, inputs = feature_map(x)
    matrix = circuit.to_matrix(**inputs).to(dtype=torch.complex128)

    phases = []
    for basis in range(4):
        z0 = 1.0 if (basis >> 1) & 1 == 0 else -1.0
        z1 = 1.0 if basis & 1 == 0 else -1.0
        phase = 0.7 * (x[0] * z0 + x[1] * z1) + 0.2 * x[0] * x[1] * z0 * z1
        phases.append(torch.exp(1j * phase))
    expected = torch.diag(torch.stack(phases)).to(dtype=torch.complex128)

    assert torch.allclose(matrix, expected, atol=1e-6)


def test_angle_embedding_ry_state():
    embedding = embeddings.AngleEmbeddingRY(n_qubits=1)
    x = torch.tensor([0.8])
    circuit, inputs = embedding(x)
    state = circuit(**inputs)

    expected = torch.stack(
        (torch.cos(x / 2), -torch.sin(x / 2)),
        dim=0,
    ).to(dtype=torch.complex64).squeeze()
    assert torch.allclose(state, expected, atol=1e-6)


def test_qaoa_layer_single_edge_structure():
    layer = ansaetze.QAOALayer(
        n_qubits=2,
        p=1,
        edges=[(0, 1)],
        w=torch.tensor([0.5]),
        h=torch.tensor([0.1, -0.3]),
    )
    circuit, inputs = layer()

    built_layers = list(circuit.circuit.layers)
    assert len(built_layers) == 7
    assert isinstance(built_layers[0], op.BuiltCNOT)
    assert isinstance(built_layers[1], op.BuiltRZ)
    assert isinstance(built_layers[2], op.BuiltCNOT)
    assert isinstance(built_layers[3], op.BuiltRZ)
    assert isinstance(built_layers[4], op.BuiltRZ)
    assert isinstance(built_layers[5], op.BuiltRX)
    assert isinstance(built_layers[6], op.BuiltRX)

    key_order = [
        "qaoa_l0_edge0_zz",
        "qaoa_l0_q0_rz",
        "qaoa_l0_q1_rz",
        "qaoa_l0_q0_rx",
        "qaoa_l0_q1_rx",
    ]
    for key in key_order:
        assert key in inputs


def test_hardware_efficient_ansatz_backpropagation():
    ansatz = ansaetze.HardwareEfficientAnsatz(n_qubits=2, layers=2)
    circuit, params = ansatz()
    state = circuit(**params)

    indices = torch.arange(state.shape[0])
    z0 = 1.0 - 2.0 * ((indices >> 1) & 1).to(torch.float32)
    expectation = (state.conj() * (state * z0.to(state.dtype))).sum().real
    loss = 1.0 - expectation
    loss.backward()

    assert ansatz.theta.grad is not None
    assert torch.isfinite(ansatz.theta.grad).all()
