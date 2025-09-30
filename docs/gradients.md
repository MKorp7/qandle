# Gradient evaluation

The ``parameter_shift_forward`` helper now evaluates all ``±π/2`` shifts in
parallel using :func:`torch.vmap`.  Instead of mutating a single parameter at a
time, the circuit is executed in two large batches (one for the ``+`` shifts and
one for the ``-`` shifts).  This reduces the number of Python round-trips to two
per backward pass and makes the implementation GPU friendly while keeping the
exact same gradient formula as the previous scalar loop.

The operator works with both :class:`~qandle.qcircuit.UnsplittedCircuit` and
split circuits because the batched execution is implemented through
``torch.nn.utils.stateless.functional_call``.  No state is mutated on the module
instance, so multiple differentiable evaluations can run safely in parallel.

```python
from qandle.gradients import parameter_shift_forward

circuit = qandle.qcircuit.Circuit(...)
loss = parameter_shift_forward(circuit.circuit)
loss.backward()
```

``diff_method="parameter_shift"`` continues to work with circuits that contain
noise channels or custom PyTorch modules.  The two evaluations inside each pair
of shifts must remain deterministic – when simulating noisy channels, seed the
noise model or average over batches before computing the gradient.  Tensor-network
backends such as the MPS simulator are supported because the vectorized
implementation does not depend on a fixed state representation and only calls
``forward`` on the supplied circuit module.

An explicit ``adjoint`` differentiation mode is not available yet.  Requests for
``diff_method="adjoint"`` will raise a clear ``NotImplementedError``; the
recommended workaround is to keep using the parameter-shift rule or PyTorch's
native autograd for custom observables.

## Benchmarks

The benchmarking helper ``benchmarks/workloads/parameter_shift.py`` compares the
vectorized parameter-shift implementation, a reference scalar version, and the
adjoint differentiation routine for ``P ∈ {8, 32, 128}`` parameters.  Both
shallow (two rotation layers per qubit and fewer circuit layers) and deep
circuits (single rotation per qubit with many layers) are provided, and the
benchmark can optionally exercise split circuits via ``--split-max-qubits``.

```bash
python benchmarks/workloads/parameter_shift.py --device cuda --repeats 10
python benchmarks/workloads/parameter_shift.py --device cuda --split-max-qubits 1
```

The script prints the wall-clock time for each gradient style together with the
speedup of the vectorized and adjoint methods relative to the scalar
implementation, plus the adjoint versus vectorized ratio.  When running on GPUs
the large batched launches typically reduce the parameter-shift wall time by
more than ``2×`` once the number of parameters reaches 32 or more, while the
adjoint method often offers a further improvement for deeper circuits.
