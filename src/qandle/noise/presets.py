"""Helpers for loading hardware-inspired noise presets.

This module turns the YAML schema documented in ``docs/tutorial/noise_presets.rst``
into :class:`qandle.noise.model.NoiseModel` instances.  The presets specify a
``broadcast`` directive that controls how each noise channel should be expanded
for a given gate:

``per_target``
    Instantiate one channel per qubit that the gate touches.

``per_pair``
    Instantiate a multi-qubit channel for the ordered pair of qubits the gate
    touches.  This is primarily intended for correlated depolarising noise on
    two-qubit entangling gates.

``per_qubit``
    Instantiate one channel per qubit in the circuit.  This is typically used
    for global decoherence that wraps every gate.

The :func:`load_presets` function accepts a parsed YAML dictionary and returns
rich :class:`NoisePreset` objects.  Each preset can then be converted into a
fully fledged :class:`~qandle.noise.model.NoiseModel` by mapping the logical
gate names in your circuit to the channel macros defined in the YAML file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, Mapping, MutableMapping

from .channels import (
    AmplitudeDamping,
    CorrelatedDepolarizing,
    Dephasing,
    Depolarizing,
    NoiseChannel,
    PhaseFlip,
)
if TYPE_CHECKING:  # pragma: no cover - circular import guard for typing only
    from .model import NoiseModel

__all__ = [
    "BroadcastedNoise",
    "NoiseMacro",
    "NoisePreset",
    "load_presets",
]


_CHANNEL_REGISTRY: dict[str, type[NoiseChannel]] = {
    "AmplitudeDamping": AmplitudeDamping,
    "CorrelatedDepolarizing": CorrelatedDepolarizing,
    "Depolarizing": Depolarizing,
    "Dephasing": Dephasing,
    "PhaseFlip": PhaseFlip,
}


def _normalise_params(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a defensive copy of ``params`` with stringified keys."""

    return {str(k): v for k, v in dict(params).items()}


def _iter_qubits_from_value(value: Any) -> Iterator[int]:
    if isinstance(value, int):
        yield value
        return

    if isinstance(value, (list, tuple, set)):
        for elem in value:
            yield from _iter_qubits_from_value(elem)
        return

    if isinstance(value, Mapping):
        for elem in value.values():
            yield from _iter_qubits_from_value(elem)
        return

    if hasattr(value, "qubit"):
        yield from _iter_qubits_from_value(getattr(value, "qubit"))

    if hasattr(value, "qubits"):
        yield from _iter_qubits_from_value(getattr(value, "qubits"))


def _extract_gate_qubits(layer) -> tuple[int, ...]:
    """Return the ordered tuple of qubits acted on by ``layer``.

    The helper mirrors the gate-introspection logic used by the splitter to
    identify which qubits a built operator touches.  We avoid importing the
    splitter directly to keep the noise module self contained.
    """

    seen: set[int] = set()
    ordered: list[int] = []

    def add_qubits(values: Iterable[int]) -> None:
        for qubit in values:
            if qubit not in seen:
                seen.add(qubit)
                ordered.append(qubit)

    primary_attrs = ("qubits",)
    secondary_attrs = (
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

    for attr in primary_attrs + secondary_attrs:
        if hasattr(layer, attr):
            add_qubits(_iter_qubits_from_value(getattr(layer, attr)))

    return tuple(ordered)


@dataclass(frozen=True)
class BroadcastedNoise:
    """Specification of a noise channel with a broadcast directive."""

    channel: str
    params: Mapping[str, Any]
    broadcast: str = "per_target"

    def instantiate(self, *, num_qubits: int, layer=None) -> list[NoiseChannel]:
        """Instantiate concrete :class:`NoiseChannel` objects."""

        cls = _CHANNEL_REGISTRY.get(self.channel)
        if cls is None:
            raise KeyError(f"Unknown noise channel '{self.channel}' in preset")

        params = _normalise_params(self.params)

        if self.broadcast == "per_qubit":
            return [cls(qubit=idx, **params) for idx in range(num_qubits)]

        if layer is None:
            raise ValueError(
                "per_target and per_pair broadcasts require a gate layer during instantiation"
            )

        targets = _extract_gate_qubits(layer)
        if not targets:
            raise ValueError(f"Unable to determine qubits for layer {layer!r}")

        if self.broadcast == "per_target":
            return [cls(qubit=idx, **params) for idx in targets]

        if self.broadcast == "per_pair":
            if len(targets) != 2:
                raise ValueError(
                    "per_pair broadcasts expect gates acting on exactly two qubits"
                )
            return [cls(qubits=tuple(targets), **params)]

        raise ValueError(f"Unsupported broadcast directive '{self.broadcast}'")


@dataclass(frozen=True)
class NoiseMacro:
    """Container for before/after noise specifications."""

    before: tuple[BroadcastedNoise, ...] = field(default_factory=tuple)
    after: tuple[BroadcastedNoise, ...] = field(default_factory=tuple)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "NoiseMacro":
        before = tuple(
            BroadcastedNoise(
                channel=item["channel"],
                params=_normalise_params(item.get("params", {})),
                broadcast=item.get("broadcast", "per_target"),
            )
            for item in mapping.get("before", [])
        )
        after = tuple(
            BroadcastedNoise(
                channel=item["channel"],
                params=_normalise_params(item.get("params", {})),
                broadcast=item.get("broadcast", "per_target"),
            )
            for item in mapping.get("after", [])
        )
        return cls(before=before, after=after)


@dataclass
class NoisePreset:
    """Represents a noise preset loaded from ``noise_presets.yaml``."""

    name: str
    family: str | None = None
    description: str | None = None
    references: tuple[str, ...] = ()
    global_noise: NoiseMacro = field(default_factory=NoiseMacro)
    channels: MutableMapping[str, NoiseMacro] = field(default_factory=dict)
    measurement: Mapping[str, Any] = field(default_factory=dict)
    initialization: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, name: str, mapping: Mapping[str, Any]) -> "NoisePreset":
        global_noise = NoiseMacro.from_mapping(mapping.get("global_noise", {}))
        channels = {
            str(key): NoiseMacro.from_mapping(value)
            for key, value in mapping.get("channels", {}).items()
        }
        references = tuple(mapping.get("references", ()))
        return cls(
            name=name,
            family=mapping.get("family"),
            description=mapping.get("description"),
            references=references,
            global_noise=global_noise,
            channels=channels,
            measurement=dict(mapping.get("measurement", {})),
            initialization=dict(mapping.get("initialization", {})),
        )

    @staticmethod
    def _macro_to_spec(macro: NoiseMacro) -> dict[str, list[BroadcastedNoise]]:
        spec: dict[str, list[BroadcastedNoise]] = {}
        if macro.before:
            spec["before"] = list(macro.before)
        if macro.after:
            spec.setdefault("after", [])
            spec["after"].extend(macro.after)
        return spec

    def to_noise_model(
        self,
        *,
        num_qubits: int,
        gate_mapping: Mapping[str, str] | None = None,
    ) -> NoiseModel:
        """Convert the preset into a :class:`NoiseModel`.

        Parameters
        ----------
        num_qubits:
            Number of qubits in the circuit that the noise model will decorate.
        gate_mapping:
            Optional mapping that associates gate names (as seen by
            :class:`~qandle.noise.model.NoiseModel`) with the macro names stored
            in the preset.  When omitted, the keys defined in ``channels`` are
            treated as the gate identifiers directly.
        """

        if gate_mapping is None:
            items = list(self.channels.items())
        else:
            items = []
            for gate, macro_name in gate_mapping.items():
                if macro_name not in self.channels:
                    raise KeyError(
                        f"Gate '{gate}' references unknown noise macro '{macro_name}'"
                    )
                items.append((gate, self.channels[macro_name]))

        global_spec = self._macro_to_spec(self.global_noise)
        global_noise: Any = global_spec if global_spec else []

        channel_specs: dict[str, dict[str, list[BroadcastedNoise]]] = {}
        for gate, macro in items:
            built = self._macro_to_spec(macro)
            if built:
                channel_specs[gate] = built

        from .model import NoiseModel

        return NoiseModel(
            channels=channel_specs or None,
            global_noise=global_noise,
        )


def load_presets(preset_data: Mapping[str, Any]) -> dict[str, NoisePreset]:
    """Load presets from a parsed YAML dictionary."""

    presets: dict[str, NoisePreset] = {}
    for name, mapping in preset_data.items():
        presets[str(name)] = NoisePreset.from_mapping(str(name), mapping)
    return presets

