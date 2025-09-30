import importlib.util
import pathlib
import pytest


if importlib.util.find_spec("torch") is None:
    pytest.skip("torch is required for operator tests", allow_module_level=True)


OPERATORS_PATH = pathlib.Path(__file__).resolve().parents[1] / "src" / "qandle" / "operators.py"
spec = importlib.util.spec_from_file_location("qandle.operators", OPERATORS_PATH)
operators = importlib.util.module_from_spec(spec)
assert spec and spec.loader  # satisfy mypy/static type checkers if run
spec.loader.exec_module(operators)


def test_named_rx_str_representation():
    gate = operators.RX(qubit=0, name="label")

    assert str(gate) == "RX0 (label)"


def test_unnamed_rx_str_representation_omits_placeholder():
    gate = operators.RX(qubit=0)

    assert str(gate) == "RX0"
