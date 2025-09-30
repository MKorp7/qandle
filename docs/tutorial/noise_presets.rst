Noise model presets for hardware-like simulations
=================================================

The YAML file :download:`noise_presets.yaml <../_static/noise_presets.yaml>`
collects the headline numbers that the leading hardware vendors publish for
their trapped-ion and superconducting systems.  Each preset condenses the
numbers into a compact schema that can be expanded into
:class:`qandle.noise.NoiseModel` instances or used as a starting point for your
own calibrations.

Schema overview
---------------

The YAML is organised by preset name.  Under each preset the following
sections appear:

``family`` and ``description``
    Human-readable metadata that summarises the hardware target.

``references``
    URLs pointing at the vendor documentation that exposes the calibration
    numbers.

``global_noise``
    Optional set of channels that should wrap every gate.  The entries are
    organised under ``before`` and ``after`` keys and use a ``broadcast`` flag
    to indicate how the channels should be expanded (see below).

``channels``
    Default noise macros for classes of gates (for example ``default_1q`` or
    ``default_2q``).  When you map a gate to one of these buckets you can use
    the same ``before``/``after`` structure as ``global_noise``.

``measurement`` / ``initialization``
    Convenience containers for SPAM parameters when you want to extend the
    simulator to cover non-unitary readout or imperfect resets.

Each noise entry provides the channel ``name`` and the keyword arguments in a
``params`` mapping.  The ``broadcast`` field specifies how the entry should be
expanded into concrete :mod:`qandle.noise.channels` objects:

``per_qubit``
    Duplicate the channel once per qubit in the register (useful for global
    T1/T2 or single-qubit depolarising rates).

``per_target``
    Apply the channel independently to every qubit the gate touches.

``per_pair``
    Apply the channel to the ordered pairs of qubits acted on by the gate.

Loading a preset
----------------

The helper :func:`qandle.noise.load_presets` understands the schema above and
returns :class:`~qandle.noise.presets.NoisePreset` instances.  Each preset can be
turned into a fully fledged :class:`~qandle.noise.model.NoiseModel` by providing
the number of qubits and a mapping from your gate names to the macros defined in
the YAML file.

.. code-block:: python

    import yaml
    from qandle import operators
    from qandle.noise import load_presets
    from qandle.qcircuit import Circuit

    with open("noise_presets.yaml", "r", encoding="utf8") as handle:
        presets = load_presets(yaml.safe_load(handle))

    # Map gate identifiers to the macros defined in the preset.
    gate_mapping = {
        "U": "default_1q",
        "RX": "default_1q",
        "RY": "default_1q",
        "RZ": "default_1q",
        "CNOT": "default_2q",
    }

    preset = presets["quantinuum-h1"]
    noise_model = preset.to_noise_model(num_qubits=4, gate_mapping=gate_mapping)

    circuit = Circuit([
        operators.RX(0),
        operators.CNOT(0, 1),
    ], num_qubits=4)

    # The model can be supplied directly to Circuit.forward(..., noise_model=noise_model).

The resulting :class:`NoiseModel` can be passed directly to
:meth:`qandle.qcircuit.Circuit.forward`.  ``NoisePreset`` also exposes the
``measurement`` and ``initialization`` dictionaries from the YAML block so that
readout models can be layered on top if needed.

Hardware-focused presets
------------------------

The YAML file currently ships with the following presets:

* ``quantinuum-h1`` – mirrors the public noise model used by the H1 emulator,
  including the 2.1×10⁻⁵ single-qubit and 8.8×10⁻⁴ two-qubit depolarising
  rates, slow dephasing, and SPAM figures.
* ``ionq-aria-1`` and ``ionq-forte-1`` – expose the depolarising rates that the
  IonQ noisy simulators use (Aria-1: p₁ = 5.0×10⁻⁴, p₂ = 1.33×10⁻²; Forte-1:
  p₁ = 2.67×10⁻⁴, p₂ = 4.949×10⁻³) together with a 0.4% SPAM envelope.
* ``ibm-generic-127q`` – translates representative T₁/T₂ values (0.35 ms / 0.15
  ms) and gate durations (35 ns / 300 ns) into amplitude- and phase-damping
  probabilities and records a 2% assignment error for readout.

You can duplicate one of the blocks in :download:`noise_presets.yaml
<../_static/noise_presets.yaml>` and tweak the numbers to build custom profiles
for other devices or to reflect the latest calibration pull from the hardware
provider.
