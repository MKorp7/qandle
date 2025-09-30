import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--max-qubits",
        action="store",
        type=int,
        default=None,
        help="Override the max_qubits parameter for splitter acceptance tests.",
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "max_qubits" not in metafunc.fixturenames:
        return
    option = metafunc.config.getoption("--max-qubits")
    values = [int(option)] if option is not None else [4, 6, 8]
    metafunc.parametrize("max_qubits", values)
