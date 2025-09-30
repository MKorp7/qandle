import pytest

try:  # pragma: no cover - optional torch dependency guard
    import torch

    from qandle import operators
    from qandle.noise import load_presets
    from qandle.noise.channels import BuiltNoiseChannel, CorrelatedDepolarizing, Depolarizing, PhaseFlip
    from qandle.qcircuit import Circuit
except ModuleNotFoundError:  # pragma: no cover - torch missing in environment
    pytestmark = pytest.mark.skip(reason="torch is required for noise preset tests")
else:

    SAMPLE_PRESET = {
        "toy": {
            "global_noise": {
                "before": [
                    {
                        "channel": "Depolarizing",
                        "broadcast": "per_qubit",
                        "params": {"p": 0.0},
                    }
                ]
            },
            "channels": {
                "default_1q": {
                    "after": [
                        {
                            "channel": "PhaseFlip",
                            "broadcast": "per_target",
                            "params": {"p": 1.0},
                        }
                    ]
                },
                "entangling": {
                    "after": [
                        {
                            "channel": "CorrelatedDepolarizing",
                            "broadcast": "per_pair",
                            "params": {"p": 0.2},
                        }
                    ]
                },
            },
        }
    }


    def make_identity() -> torch.Tensor:
        return torch.eye(2, dtype=torch.complex64)


    def test_per_target_noise_expansion():
        presets = load_presets(SAMPLE_PRESET)
        preset = presets["toy"]
        model = preset.to_noise_model(
            num_qubits=2, gate_mapping={"U": "default_1q"}
        )

        circuit = Circuit([operators.U(1, make_identity())], num_qubits=2)
        applied = model.apply(circuit)

        layers = list(applied.circuit.layers)
        noise_layers = [layer for layer in layers if isinstance(layer, BuiltNoiseChannel)]
        assert any(
            isinstance(layer.channel, PhaseFlip) and layer.targets == (1,)
            for layer in noise_layers
        )


    def test_per_pair_noise_targets_two_qubit_gate():
        presets = load_presets(SAMPLE_PRESET)
        preset = presets["toy"]
        model = preset.to_noise_model(
            num_qubits=2, gate_mapping={"CNOT": "entangling"}
        )

        circuit = Circuit([operators.CNOT(0, 1)], num_qubits=2)
        applied = model.apply(circuit)

        layers = list(applied.circuit.layers)
        noise_layers = [layer for layer in layers if isinstance(layer, BuiltNoiseChannel)]
        assert any(
            isinstance(layer.channel, CorrelatedDepolarizing) and layer.targets == (0, 1)
            for layer in noise_layers
        )


    def test_global_per_qubit_noise_wraps_every_gate():
        presets = load_presets(SAMPLE_PRESET)
        preset = presets["toy"]
        model = preset.to_noise_model(
            num_qubits=2,
            gate_mapping={"U": "default_1q"},
        )

        circuit = Circuit([operators.U(0, make_identity())], num_qubits=2)
        applied = model.apply(circuit)

        layers = list(applied.circuit.layers)
        dep_layers = [
            layer
            for layer in layers
            if isinstance(layer, BuiltNoiseChannel)
            and isinstance(layer.channel, Depolarizing)
        ]
        assert sorted(layer.targets for layer in dep_layers[:2]) == [(0,), (1,)]


    def test_unknown_macro_raises_helpful_error():
        presets = load_presets(SAMPLE_PRESET)
        preset = presets["toy"]
        with pytest.raises(KeyError):
            preset.to_noise_model(num_qubits=1, gate_mapping={"U": "missing"})
