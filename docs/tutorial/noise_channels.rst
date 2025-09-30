Noise channels
==============

``qandle.noise.channels`` implements a small catalogue of Kraus maps that can
be composed with the unitary layers exposed in :mod:`qandle.operators`.  The
channels mirror the stochastic ``BitFlip`` interface used throughout the code
base and therefore interoperate with :class:`~qandle.qcircuit.Circuit`, the
matrix backends and the splitter/MPS machinery.  Deterministic state-vector
executions use the affine mixing rule documented in
:class:`qandle.noise.channels.BuiltNoiseChannel`; stochastic trajectories remain
available via ``trajectory=True`` and are validated in
``tests/test_noise_channels.py``.

The following maps are provided.  All symbols are standard Pauli operators.

* :class:`~qandle.noise.channels.PhaseFlip` implements the discrete phase flip
  channel with Kraus operators ``K_0 = sqrt(1-p) I`` and ``K_1 = sqrt(p) Z``.
  Off-diagonal elements of a one-qubit density matrix are scaled by ``1-2p``.
* :class:`~qandle.noise.channels.Depolarizing` averages over random Pauli errors
  according to ``(1-p)\rho + p/3 \sum_{P \in \{X,Y,Z\}} P\rho P``.
* :class:`~qandle.noise.channels.Dephasing` (``PhaseDamping`` alias) removes
  coherence as ``(1-\gamma)\rho + \gamma\sum_j |j\rangle\langle j|\rho|j\rangle\langle j|``.
* :class:`~qandle.noise.channels.AmplitudeDamping` models energy relaxation
  with the usual ``T_1`` Kraus pair ``[[1,0],[0,\sqrt{1-\gamma}]]`` and
  ``[[0,\sqrt{\gamma}],[0,0]]``.
* :class:`~qandle.noise.channels.CorrelatedDepolarizing` applies the same Pauli
  error to a pair of qubits, ``(1-p)\rho + p/3 \sum_{P} (P\otimes P) \rho (P\otimes P)``.

When working with the MPS backend, the deterministic mixing path avoids large
intermediate density matrices and keeps the channel compatible with state-vector
splitting.  Density-matrix backends act directly on Kraus-expanded tensors and
preserve the trace for batched inputs.

For a hands-on illustration see ``examples/noise_channels_demo.ipynb`` which
compares an ideal Bell-state preparation with its noisy counterpart.
