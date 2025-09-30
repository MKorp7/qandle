import textwrap

import importlib.util
import pathlib

_QASM_PATH = pathlib.Path(__file__).resolve().parents[1] / "src" / "qandle" / "qasm.py"
_QASM_SPEC = importlib.util.spec_from_file_location("qandle.qasm", _QASM_PATH)
qasm_module = importlib.util.module_from_spec(_QASM_SPEC)
assert _QASM_SPEC is not None and _QASM_SPEC.loader is not None
_QASM_SPEC.loader.exec_module(qasm_module)

QasmRepresentation = qasm_module.QasmRepresentation
convert_to_qasm = qasm_module.convert_to_qasm


class _DummyCircuit:
    def __init__(self, reps, num_qubits):
        self._reps = reps
        self.num_qubits = num_qubits

    def to_qasm(self):
        return self._reps


def test_convert_to_qasm_deduplicates_io_declarations():
    reps = [
        [
            QasmRepresentation(
                gate_str="rz",
                qubit=0,
                qasm3_inputs="theta",
            ),
            QasmRepresentation(
                gate_str="rx",
                qubit=0,
                qasm3_inputs="theta",
            ),
        ],
        [
            QasmRepresentation(
                gate_str="measure",
                qubit=0,
                qasm3_outputs="result",
            ),
            QasmRepresentation(
                gate_str="measure",
                qubit=0,
                qasm3_outputs="result",
            ),
        ],
    ]

    circuit = _DummyCircuit(reps=reps, num_qubits=1)

    qasm = convert_to_qasm(circuit, qasm_version=3, include_header=True)

    expected_prefix = textwrap.dedent(
        """
        OPENQASM 3.0;
        qubit[1] q;
        bit[1] c;
        input float theta;
        output float result;
        """
    ).lstrip()

    assert qasm.startswith(expected_prefix)
    assert qasm.count("input float theta;") == 1
    assert qasm.count("output float result;") == 1
