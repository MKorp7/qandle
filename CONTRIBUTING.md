# Contributing to QANDLE

This document provides a short overview of the repository to help new contributors get started.

## Repository layout

- **`src/qandle/`** – main Python package. Modules of interest include:
  - `qcircuit.py` – defines the `Circuit` class used to compose and execute quantum circuits.
  - `operators.py` – gate definitions (RX, CNOT, etc.) and their built counterparts.
  - `embeddings.py` – layers that prepare classical data, e.g. `AmplitudeEmbedding` and `AngleEmbedding`.
  - `ansaetze/` – reusable circuit blocks such as `StronglyEntanglingLayer` and `TwoLocal`.
  - `measurements.py` – measurement operators returning probabilities or state vectors.
  - `drawer.py` – ASCII based circuit drawing.
  - `splitter/` – logic for splitting large circuits into subcircuits.
  - `convolution.py` – example quantum convolution layer.
  - `test/` – unit tests executed with `pytest`.
- **`docs/`** – Sphinx documentation and tutorial notebooks.
- **`pyproject.toml`** – dependency list and `poethepoet` tasks for tests and docs.

## Defining and running circuits

Circuits are composed from layers (operators). The README provides a minimal example:

```python
import torch
import qandle

# Create a quantum circuit
circuit = qandle.Circuit(
    layers=[
        qandle.AngleEmbedding(num_qubits=2),
        qandle.RX(qubit=1),
        qandle.RY(qubit=0, theta=torch.tensor(0.2)),
        qandle.RX(qubit=0, name="data_reuploading"),
        qandle.RY(qubit=1, remapping=None),
        qandle.CNOT(control=0, target=1),
        qandle.MeasureProbability(),
    ]
)

input_state = torch.rand(circuit.num_qubits, dtype=torch.float)
data_reuploading = torch.rand(1, dtype=torch.float)

# Run the circuit
circuit(input_state, data_reuploading=data_reuploading)
```

The `Circuit` object internally builds each layer. When called, the current state tensor is passed through every layer in sequence. Parameters of gates are standard PyTorch tensors and support autograd.

## Drawing and exporting

Call `circuit.draw()` to obtain a simple ASCII representation. The drawing behaviour can be customised via the `config` module (for example `DRAW_DASH`, `DRAW_SHIFT_LEFT`, etc.). Circuits can also be exported to OpenQASM 2 or 3 via `circuit.to_openqasm2()` or `to_openqasm3()`.

## Circuit splitting

`qandle.splitter` can partition circuits containing many qubits. It builds a dependency graph (using `networkx`) and rewrites the circuit into subcircuits which can be executed sequentially. The resulting `SplittedCircuit` is still a `torch.nn.Module` and can be run just like a normal circuit.

## Tooling

Development dependencies are listed in `pyproject.toml`. Useful tasks are defined under `[tool.poe.tasks]` and include:

- `poe test` – run the test-suite with coverage.
- `poe doc` – build the Sphinx documentation.
- `poe format` – run `ruff` for linting.

The project relies on PyTorch for tensor operations, qW-Map for parameter remapping, NetworkX for circuit splitting, and Einops for tensor reshaping. Documentation is generated with Sphinx and tests use PyTest.

For detailed installation instructions see the tutorial `docs/tutorial/00intro.rst`.