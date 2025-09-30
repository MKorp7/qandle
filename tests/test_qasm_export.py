from __future__ import annotations

import random

import numpy as np
import openqasm3
import pytest
import torch

import qandle
from qandle import measurements, operators


def _build_parametric_circuit() -> qandle.Circuit:
    layers = [
        operators.RX(0, name="theta"),
        operators.RY(1, theta=torch.tensor(0.25)),
        operators.CNOT(0, 1),
        measurements.MeasureProbability(),
    ]
    return qandle.Circuit(layers, num_qubits=2)


def _extract_output_names(qasm: str) -> set[str]:
    outputs: set[str] = set()
    for line in qasm.splitlines():
        line = line.strip()
        if line.startswith("output float"):
            outputs.add(line.split()[2].rstrip(";"))
    return outputs


def test_openqasm3_parametric_export_includes_io_and_registers():
    random.seed(7)
    circuit = _build_parametric_circuit()
    qasm = circuit.to_openqasm3()

    parsed = openqasm3.parser.parse(qasm)
    assert parsed.version == "3.0"

    assert "qubit[2] q;" in qasm
    assert "bit[2] c;" in qasm

    assert "input float theta;" in qasm

    outputs = _extract_output_names(qasm)
    assert len(outputs) == circuit.num_qubits
    assert len(outputs) == len(set(outputs))

    assert "rx(theta) q[0];" in qasm


def test_openqasm3_roundtrip_matches_statevector():
    layers = [
        operators.RX(0, theta=torch.tensor(0.432)),
        operators.RZ(1, theta=torch.tensor(-0.137)),
        operators.CNOT(0, 1),
    ]
    circuit = qandle.Circuit(layers, num_qubits=2)

    qasm = circuit.to_openqasm3()
    parsed = openqasm3.parser.parse(qasm)
    assert parsed.version == "3.0"

    qiskit = pytest.importorskip("qiskit")
    qc = qiskit.qasm3.loads(qasm)
    state_qiskit = qiskit.quantum_info.Statevector.from_instruction(qc).data

    state_qandle = circuit.forward().detach().cpu().numpy()
    state_qandle = state_qandle.reshape(-1)

    overlap = np.vdot(state_qiskit, state_qandle)
    if np.isclose(abs(overlap), 0.0):
        np.testing.assert_allclose(state_qandle, state_qiskit, atol=1e-6)
    else:
        phase = overlap / abs(overlap)
        np.testing.assert_allclose(state_qandle, phase * state_qiskit, atol=1e-6)
