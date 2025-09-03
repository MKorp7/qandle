from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence, Union

from .channels import NoiseChannel, BuiltNoiseChannel


NoiseLike = Union[NoiseChannel, BuiltNoiseChannel]


@dataclass
class NoiseModel:
    """Simple noise model mapping gate names to channels."""

    channels: Mapping[str, NoiseLike] | None = None
    global_noise: Sequence[NoiseLike] = field(default_factory=list)

    def apply(self, circuit: "Circuit") -> "Circuit":
        # keep as stub for now; you can wire this later
        return circuit