from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, Union

import torch
from collections.abc import (
    Iterable as IterableABC,
    Mapping as MappingABC,
    Sequence as SequenceABC,
)

from .channels import BuiltNoiseChannel, NoiseChannel
from .presets import BroadcastedNoise


NoiseLike = Union[NoiseChannel, BuiltNoiseChannel, BroadcastedNoise]


@dataclass
class NoiseModel:
    """Simple noise model mapping gate names or qubits to channels."""

    channels: Mapping[str, NoiseLike] | Sequence[NoiseLike] | None = None
    global_noise: Sequence[NoiseLike] = field(default_factory=list)

    def apply(self, circuit: "Circuit") -> "Circuit":
        if (not self.channels) and (not self.global_noise):
            return circuit

        from qandle.operators import BuiltOperator

        working = circuit.decompose()
        base = working.circuit

        # Normalise global noise specification once; reused for every gate.
        global_before, global_after = self._split_noise_sequences(self.global_noise)

        mapping_channels: Mapping[str, object] | None = None
        per_target_entries: list[tuple[set[int], tuple[NoiseChannel, ...]]] = []
        broadcasted_per_gate: list[BroadcastedNoise] = []

        if self.channels:
            if isinstance(self.channels, MappingABC):
                mapping_channels = self.channels
            else:
                flattened = self._flatten_noise(self.channels)
                grouped: dict[tuple[int, ...], list[NoiseChannel]] = {}
                for item in flattened:
                    if isinstance(item, BroadcastedNoise):
                        broadcasted_per_gate.append(item)
                        continue

                    targets = self._noise_like_targets(item)
                    if not targets:
                        continue

                    if isinstance(item, BuiltNoiseChannel):
                        channel = item.channel
                    else:
                        channel = item
                    grouped.setdefault(tuple(targets), []).append(channel)

                per_target_entries = [
                    (set(targets), tuple(channels))
                    for targets, channels in grouped.items()
                ]

        def build_sequence(
            seq: Sequence[NoiseLike], *, layer=None
        ) -> list[BuiltNoiseChannel]:
            built: list[BuiltNoiseChannel] = []
            for noise in seq:
                if isinstance(noise, BroadcastedNoise):
                    channels = noise.instantiate(num_qubits=working.num_qubits, layer=layer)
                    built.extend(channel.build(working.num_qubits) for channel in channels)
                elif isinstance(noise, BuiltNoiseChannel):
                    built.append(noise.channel.build(working.num_qubits))
                elif isinstance(noise, NoiseChannel):
                    built.append(noise.build(working.num_qubits))
                else:  # pragma: no cover - defensive programming
                    raise TypeError(
                        "Noise specifications must be NoiseChannel, BuiltNoiseChannel, or BroadcastedNoise instances"
                    )
            return built

        new_layers: list[torch.nn.Module] = []
        for layer in base.layers:
            if isinstance(layer, BuiltNoiseChannel):
                new_layers.append(layer)
                continue

            if not isinstance(layer, BuiltOperator):
                new_layers.append(layer)
                continue

            local_before: Sequence[NoiseLike] = []
            local_after: Sequence[NoiseLike] = []
            if mapping_channels:
                spec = self._lookup_noise_spec(layer, mapping_channels)
                if spec is not None:
                    local_before_seq, local_after_seq = self._split_noise_sequences(spec)
                else:
                    local_before_seq, local_after_seq = [], []
            else:
                local_before_seq, local_after_seq = [], []

            local_before = list(local_before_seq)
            local_after = list(local_after_seq)

            if per_target_entries or broadcasted_per_gate:
                targets = self._layer_qubits(layer)
                if targets:
                    target_set = set(targets)
                    for target_group, noises in per_target_entries:
                        if target_group.issubset(target_set):
                            local_after.extend(noises)

                    for spec in broadcasted_per_gate:
                        instantiated = spec.instantiate(
                            num_qubits=working.num_qubits, layer=layer
                        )
                        local_after.extend(instantiated)

            combined_before = [*global_before, *local_before]
            combined_after = [*local_after, *global_after]

            new_layers.extend(build_sequence(combined_before, layer=layer))
            new_layers.append(layer)
            new_layers.extend(build_sequence(combined_after, layer=layer))

        base.layers = torch.nn.ModuleList(new_layers)
        return working

    def sample(
        self,
        circuit: "Circuit",
        n_shots: int,
        *,
        backend: str | "QuantumBackend" = "statevector",
        backend_kwargs: Mapping[str, Any] | None = None,
        initial_state: torch.Tensor | None = None,
        measured_qubits: Sequence[int] | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample ``n_shots`` noisy trajectories of ``circuit``.

        The circuit is executed gate-by-gate with stochastic noise applied
        after every layer according to this noise model.  Each trajectory is
        collapsed into a computational basis sample which is returned together
        with aggregated counts and empirical probabilities.
        """

        if n_shots < 0:
            raise ValueError("n_shots must be non-negative")

        from qandle.qcircuit import Circuit as CircuitCls  # local import
        from qandle.qcircuit import UnsplittedCircuit, _apply_backend_layer, _make_backend
        from qandle.measurements import BuiltMeasurement

        if not isinstance(circuit, CircuitCls):
            raise TypeError("circuit must be a qandle.qcircuit.Circuit instance")

        noise_applied = self.apply(circuit)
        impl = noise_applied.circuit
        if not isinstance(impl, UnsplittedCircuit):
            impl = impl.decompose()

        layers = list(impl.layers)
        if any(isinstance(layer, BuiltMeasurement) for layer in layers):
            raise NotImplementedError(
                "NoiseModel.sample does not support circuits with measurement operations."
            )

        num_qubits = noise_applied.num_qubits
        backend_kwargs = dict(backend_kwargs or {})

        if generator is None:
            generator = torch.Generator(device=backend_kwargs.get("device", "cpu"))
            if seed is not None:
                generator.manual_seed(seed)
        elif seed is not None:
            raise ValueError("Provide either 'seed' or 'generator', not both")

        def make_backend() -> "QuantumBackend":
            be = _make_backend(backend, num_qubits, **backend_kwargs)
            # ``_make_backend`` returns the provided backend unchanged if it's
            # already an instance.  Ensure we start from the |0...0> state by
            # reallocating explicitly.
            be.allocate(num_qubits)
            if not hasattr(be, "state"):
                raise NotImplementedError(
                    "NoiseModel.sample currently supports backends exposing a 'state' attribute."
                )
            return be

        measured_qubits = list(range(num_qubits)) if measured_qubits is None else list(measured_qubits)
        if sorted(measured_qubits) != list(measured_qubits):
            raise ValueError("measured_qubits must be sorted in ascending order")
        if len(set(measured_qubits)) != len(measured_qubits):
            raise ValueError("measured_qubits must be unique")
        if not all(0 <= q < num_qubits for q in measured_qubits):
            raise ValueError("measured_qubits indices must be within the circuit range")

        n_meas = len(measured_qubits)
        samples = torch.zeros(n_shots, dtype=torch.int64)
        bitstrings = torch.zeros(n_shots, n_meas, dtype=torch.int64)

        noise_kwargs = {"trajectory": True, "rng": generator}

        for shot in range(n_shots):
            be = make_backend()
            if initial_state is not None:
                state = torch.as_tensor(initial_state, dtype=be.state.dtype, device=be.state.device)
                if state.numel() != be.state.numel():
                    raise ValueError("initial_state has incompatible size")
                be.state = state.clone()

            for layer in layers:
                _apply_backend_layer(layer, be, noise_model=self, noise_kwargs=noise_kwargs)

            probs = be.measure()
            if probs.dim() != 1:
                raise RuntimeError("Backend measurement must return a 1D probability vector")
            if probs.numel() != (1 << num_qubits):
                raise RuntimeError("Measurement probabilities have unexpected length")

            choice = torch.multinomial(probs, num_samples=1, generator=generator).item()

            if n_meas == num_qubits:
                samples[shot] = choice
                for idx, qubit in enumerate(measured_qubits):
                    bitstrings[shot, idx] = (choice >> (num_qubits - qubit - 1)) & 1
            else:
                value = 0
                for bit_index, qubit in enumerate(measured_qubits):
                    bit = (choice >> (num_qubits - qubit - 1)) & 1
                    value |= bit << (n_meas - bit_index - 1)
                    bitstrings[shot, bit_index] = bit
                samples[shot] = value

        if n_meas != num_qubits:
            counts = torch.bincount(samples, minlength=1 << n_meas)
        else:
            counts = torch.bincount(samples, minlength=1 << num_qubits)

        if n_shots == 0:
            probabilities = counts.to(dtype=torch.float32)
        else:
            probabilities = counts.to(dtype=torch.float32) / float(n_shots)

        return {
            "bitstrings": bitstrings,
            "samples": samples,
            "counts": counts,
            "probabilities": probabilities,
        }

    def _split_noise_sequences(
        self, spec: Sequence[NoiseLike] | Mapping[str, Sequence[NoiseLike]] | NoiseLike | None
    ) -> tuple[list[NoiseLike], list[NoiseLike]]:
        if spec is None:
            return [], []

        if isinstance(spec, MappingABC):
            before = spec.get("before", [])
            after = spec.get("after", [])
        else:
            before = []
            after = spec

        return self._flatten_noise(before), self._flatten_noise(after)

    def _flatten_noise(self, spec: Sequence[NoiseLike] | NoiseLike) -> list[NoiseLike]:
        if isinstance(spec, (NoiseChannel, BuiltNoiseChannel, BroadcastedNoise)):
            return [spec]

        if isinstance(spec, SequenceABC) and not isinstance(spec, (str, bytes)):
            flattened: list[NoiseLike] = []
            for item in spec:
                flattened.extend(self._flatten_noise(item))
            return flattened

        return [spec]

    def _lookup_noise_spec(
        self, layer: torch.nn.Module, channels: Mapping[str, object]
    ) -> Sequence[NoiseLike] | Mapping[str, Sequence[NoiseLike]] | NoiseLike | None:
        keys = self._layer_keys(layer)
        for key in keys:
            if key in channels:
                return channels[key]
        return None

    @staticmethod
    def _layer_keys(layer: torch.nn.Module) -> list[str]:
        name = layer.__class__.__name__
        keys = [name]
        if name.startswith("Built"):
            keys.append(name[5:])
        unbuilt = getattr(layer, "unbuilt_class", None)
        if unbuilt is not None:
            keys.append(unbuilt.__name__)
        description = getattr(layer, "description", None)
        if description:
            keys.append(str(description))
        return list(dict.fromkeys(keys))

    @staticmethod
    def _noise_like_targets(noise: NoiseLike) -> tuple[int, ...]:
        if isinstance(noise, BuiltNoiseChannel):
            return tuple(noise.targets)
        if isinstance(noise, NoiseChannel):
            return tuple(noise.targets)
        raise TypeError(
            "Noise specifications must expose target qubits via 'targets'"
        )

    @staticmethod
    def _iter_qubits_from_value(value) -> IterableABC[int]:
        if isinstance(value, int):
            yield value
            return

        if isinstance(value, (list, tuple, set)):
            for elem in value:
                yield from NoiseModel._iter_qubits_from_value(elem)
            return

        if isinstance(value, MappingABC):
            for elem in value.values():
                yield from NoiseModel._iter_qubits_from_value(elem)
            return

        if hasattr(value, "qubit"):
            yield from NoiseModel._iter_qubits_from_value(getattr(value, "qubit"))

        if hasattr(value, "qubits"):
            yield from NoiseModel._iter_qubits_from_value(getattr(value, "qubits"))

    @staticmethod
    def _layer_qubits(layer: torch.nn.Module) -> tuple[int, ...]:
        seen: set[int] = set()
        ordered: list[int] = []

        attrs = (
            "qubits",
            "targets",
            "qubit",
            "control",
            "target",
            "c",
            "t",
            "a",
            "b",
            "c1",
            "c2",
            "control1",
            "control2",
        )

        for attr in attrs:
            if hasattr(layer, attr):
                for qubit in NoiseModel._iter_qubits_from_value(getattr(layer, attr)):
                    if qubit not in seen:
                        seen.add(qubit)
                        ordered.append(qubit)

        return tuple(ordered)
