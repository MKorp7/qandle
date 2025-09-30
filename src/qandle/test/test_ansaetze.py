import pennylane as qml
import torch
import qandle
import qandle.operators as op


def _torch_device(num_qubits: int):
    """Return a PennyLane device with Torch support if available.

    PennyLane 0.39 removed the ``default.qubit.torch`` device in favour of
    importing the ``lightning.qubit`` plugin with the Torch interface.  Older
    versions (and our CI environment) might not have that plugin installed
    either, so we progressively fall back to the plain ``default.qubit``
    simulator which still supports the Torch interface.  Newer PennyLane
    versions raise :class:`qml.DeviceError`, while older ones raised
    ``pennylane.exceptions.DeviceError``.  To keep the test suite independent of
    the exact PennyLane release we simply try a list of candidate device names
    and return the first one that is available.
    """

    device_candidates = ["default.qubit.torch", "lightning.qubit", "default.qubit"]
    last_error = None

    for dev_name in device_candidates:
        try:
            return qml.device(dev_name, wires=num_qubits)
        except Exception as exc:  # pragma: no cover - defensive
            last_error = exc

    if last_error is not None:  # pragma: no cover - should not happen
        raise last_error

    raise RuntimeError("No PennyLane device available")



def test_sel():
    """Test StronglyEntanglingLayers"""
    num_qubits = 4
    depth = 10
    pl_dev = _torch_device(num_qubits)
    inp = torch.rand(2**num_qubits, dtype=torch.cfloat)
    inp = inp / inp.norm()
    weights = torch.rand(depth, num_qubits, 3)

    @qml.qnode(device=pl_dev, interface="torch")
    def pl_circuit():
        qml.StatePrep(inp, wires=range(num_qubits))
        qml.StronglyEntanglingLayers(weights=weights, wires=range(num_qubits))
        return qml.state()

    pl_result = pl_circuit().to(torch.cfloat)
    qandle_sel = qandle.StronglyEntanglingLayer(
        qubits=list(range(num_qubits)), depth=depth, q_params=weights, remapping=None
    ).build(num_qubits=num_qubits)
    qandle_result = qandle_sel(inp)
    assert torch.allclose(pl_result, qandle_result, rtol=1e-6, atol=1e-6)


def test_sel_to_matrix():
    num_qubits = 5
    depth = 11
    pl_dev = _torch_device(num_qubits)
    weights = torch.rand(depth, num_qubits, 3)

    @qml.qnode(device=pl_dev, interface="torch")
    def pl_circuit():
        qml.StronglyEntanglingLayers(weights=weights, wires=range(num_qubits))
        return qml.state()

    pl_matrix = qml.matrix(qml.StronglyEntanglingLayers(weights=weights, wires=range(num_qubits)))
    qandle_sel = qandle.StronglyEntanglingLayer(
        qubits=list(range(num_qubits)), depth=depth, q_params=weights, remapping=None
    ).build(num_qubits=num_qubits)
    qandle_matrix = qandle_sel.to_matrix()
    assert torch.allclose(pl_matrix, qandle_matrix, rtol=1e-6, atol=1e-6)


def test_sel_sub():
    num_qubits = 5
    batch = 17

    qandle_sel1 = qandle.StronglyEntanglingLayer(qubits=[0, 1, 2, 3], depth=7).build(num_qubits=5)
    qandle_sel2 = qandle.StronglyEntanglingLayer(qubits=[1, 2, 3, 4], depth=7).build(num_qubits=5)
    qandle_sel3 = qandle.StronglyEntanglingLayer(qubits=[0, 1, 3, 4], depth=7).build(num_qubits=5)

    inp = torch.rand(batch, 2**num_qubits, dtype=torch.cfloat)
    inp = inp / torch.linalg.norm(inp, dim=1, keepdim=True)
    qandle_sel1(inp)
    qandle_sel2(inp)
    qandle_sel3(inp)


def test_sel_batched():
    num_qubits = 5
    depth = 7
    batch = 17
    pl_dev = _torch_device(num_qubits)
    inp = torch.rand(batch, 2**num_qubits, dtype=torch.cfloat)
    inp = inp / torch.linalg.norm(inp, dim=1, keepdim=True)
    weights = torch.rand(depth, num_qubits, 3)

    @qml.qnode(device=pl_dev, interface="torch")
    def pl_circuit():
        qml.StatePrep(inp, wires=range(num_qubits))
        qml.StronglyEntanglingLayers(weights=weights, wires=range(num_qubits))
        return qml.state()

    pl_result = pl_circuit().to(torch.cfloat)
    qandle_sel = qandle.StronglyEntanglingLayer(
        qubits=list(range(num_qubits)), depth=depth, q_params=weights, remapping=None
    ).build(num_qubits=num_qubits)
    qandle_result = qandle_sel(inp)
    assert torch.allclose(pl_result, qandle_result, rtol=1e-6, atol=1e-6)


def test_sel_budget():
    num_qubits = 5
    depth = 4
    rots = ["rz", "ry", "rz"]
    sel = qandle.StronglyEntanglingLayer(
        qubits=list(range(num_qubits)), depth=depth, rotations=rots, num_qubits_total=num_qubits
    ).build(num_qubits=num_qubits)
    sel_budget = qandle.StronglyEntanglingLayerBudget(
        num_qubits_total=num_qubits,
        rotations=rots,
        param_budget=num_qubits * depth * len(rots),
        qubits=list(range(num_qubits)),
        control_gate_spacing=3,
    )
    sel_dec = sel.decompose()
    sel_budget_dec = sel_budget.layers
    assert len(sel_dec) == len(sel_budget_dec)
    ssel_dec = sorted(sel_dec, key=lambda x: str(x))
    ssel_budget_dec = sorted(sel_budget_dec, key=lambda x: str(x))
    for s, sb in zip(ssel_dec, ssel_budget_dec):
        assert type(s) is type(sb)
    inp = torch.rand(7, 2**num_qubits, dtype=torch.cfloat)
    inp = inp / torch.linalg.norm(inp, dim=1, keepdim=True)
    sel(inp)


def test_sel_to_qasm_sequence():
    qubits = [0, 1]
    depth = 1
    q_params = torch.tensor(
        [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]], dtype=torch.float
    )
    sel = qandle.StronglyEntanglingLayer(
        qubits=qubits,
        depth=depth,
        q_params=q_params,
        remapping=None,
        num_qubits_total=len(qubits),
    )

    expected = [
        qasm.QasmRepresentation("rz", 0.1, 0),
        qasm.QasmRepresentation("ry", 0.2, 0),
        qasm.QasmRepresentation("rz", 0.3, 0),
        qasm.QasmRepresentation("rz", 0.4, 1),
        qasm.QasmRepresentation("ry", 0.5, 1),
        qasm.QasmRepresentation("rz", 0.6, 1),
        qasm.QasmRepresentation("cx q[0], q[1]"),
        qasm.QasmRepresentation("cx q[1], q[0]"),
    ]

    assert sel.to_qasm() == expected


def test_sel_general():
    num_qubits = 5
    qandle_sel_ub = qandle.StronglyEntanglingLayer(qubits=list(range(num_qubits)), depth=10)
    qandle_sel = qandle_sel_ub.build(num_qubits=num_qubits)
    inp = torch.rand(2**num_qubits, dtype=torch.cfloat)
    assert isinstance(qandle_sel(inp), torch.Tensor)
    assert isinstance(qandle_sel.decompose(), list)
    assert isinstance(qandle_sel.__str__(), str)


def test_sel_unbuilt_decompose_without_total_qubits():
    sel = qandle.StronglyEntanglingLayer(qubits=[0, 2, 4], depth=3)
    layers = sel.decompose()

    assert layers, "Decomposition should return at least one operator"
    assert all(isinstance(layer, op.UnbuiltOperator) for layer in layers)
    assert any(isinstance(layer, op.CNOT) for layer in layers), "Expected unbuilt CNOT gates"


def test_twolocal():
    utwo = qandle.TwoLocal(qubits=list(range(4)))
    assert isinstance(utwo.decompose(), list)
    assert isinstance(utwo.__str__(), str)

    inp = torch.rand(2**4, dtype=torch.cfloat)
    inp = inp / inp.norm()
    two = utwo.build(num_qubits=4)
    assert isinstance(two(inp), torch.Tensor)
    assert isinstance(two.decompose(), list)
    assert isinstance(two.__str__(), str)
    two.to_qasm()

    inp2 = torch.rand(2**5, dtype=torch.cfloat)
    inp2 = inp2 / inp2.norm()
    two2 = utwo.build(num_qubits=5)
    assert isinstance(two2(inp2), torch.Tensor)


def test_su():
    for num_w in [2, 3, 6]:
        inp = torch.rand(2**num_w, dtype=torch.cfloat)
        inp = inp / inp.norm()
        for reps in [0, 1, 2, 10]:
            for rots in [["ry"], ["rz"], ["ry", "rx"]]:
                su = qandle.SU(reps=reps, rotations=rots).build(num_qubits=num_w)
                assert isinstance(su(inp), torch.Tensor)
                assert isinstance(su.decompose(), list)
                assert isinstance(su.__str__(), str)
                assert len(su.decompose()) == len(rots) * num_w * (1 + reps) + reps * (num_w - 1)
        for additional in [0, 1, 2]:
            su1 = qandle.SU(qubits=list(range(additional, num_w + additional)))
            inp = torch.rand(2 ** (num_w + additional * 2), dtype=torch.cfloat)
            inp = inp / inp.norm()
            su1.build(num_qubits=num_w + additional * 2)(inp)
