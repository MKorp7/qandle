# Benchmark Suite Overview

The benchmarking utilities in this directory provide a unified way to
compare multiple quantum simulation backends across a consistent set of
workloads.  The suite focuses on the following families of devices:

* **QANDLE** – Statevector and MPS simulators (CPU/GPU variants).
* **PennyLane** – ``default.qubit``, ``default.tensor``, ``lightning.qubit``
  and ``lightning.gpu`` devices.

Each backend is registered via :mod:`benchmarks.backends` and can be
activated from the command line by using its registry key.  All benchmark
scripts share a common set of flags for selecting devices, hardware knobs,
and output options.

Common registry keys:

* ``qandle_sv_cpu`` / ``qandle_sv_gpu`` / ``qandle_mps_cpu`` – QANDLE 2.x
  backends that require ``QANDLE_NEW_PATH`` to point at the checkout.
* ``qandle_old_sv_cpu`` / ``qandle_old_sv_gpu`` /
  ``qandle_old_mps_cpu`` – legacy QANDLE 1.x simulators that look for the
  ``QANDLE_PATH`` environment variable.
* ``pl_default_qubit`` / ``pl_lightning_qubit`` / ``pl_lightning_gpu`` /
  ``pl_default_tensor`` – PennyLane reference and Lightning devices.

When both environment variables are configured you can benchmark the new
and legacy QANDLE stacks side-by-side in a single run.



## Installation

Create a virtual environment and install the benchmark dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r benchmarks/requirements-bench.txt
```

Backends such as PennyLane Lightning or QANDLE require their respective
packages to be available in the environment.  GPU devices additionally
require a CUDA-enabled PyTorch installation.


## Listing available devices

Backends are exposed through the registry defined in
``benchmarks/backends/registry.py``.  Use the helper below to list the devices that
can be instantiated on the current machine:

```bash
python - <<'PY'
from benchmarks.backends import available_backends, get_backend

for key in available_backends():
    spec = get_backend(key)
    print(f"{key:20s} | {spec.label:35s} | hardware={spec.hardware} | dtype={spec.defaults.get('dtype')}")
PY
```

GPU entries are included only when CUDA is available.  When a backend is
missing a required dependency its availability check reports the reason,
and the benchmark scripts skip the device gracefully.


## Common CLI flags

All benchmark entry points accept a shared set of command line options:

* ``--devices`` – comma-separated list of backend keys.  Defaults to all
  available devices.
* ``--dtype`` – preferred simulation dtype
  (``float32``/``float64``/``complex64``/``complex128``).  Unsupported
  requests fall back to backend defaults.
* ``--threads`` – CPU thread override for compatible devices.
* ``--bond-dim`` – MPS bond dimension knob for tensor/MPS backends.
* ``--splitting`` – ``on``/``off`` toggle for memory splitting (where
  supported by QANDLE backends).
* ``--shots`` – shot count (``0`` enables analytic mode when possible).
* ``--reps`` – number of timing repetitions after an implicit warm-up run.
* ``--seed`` – base RNG seed applied to Python, NumPy, and PyTorch.
* ``--out`` – path to the CSV file where results are appended.
* ``--baseline`` – optional baseline device key used later by
  ``benchmarks/compare.py``.
* ``--tags`` – free-form string recorded in the CSV (useful for machine
  identifiers).

Every script prints CUDA availability before running and records the applied
configuration in the output rows.


## Benchmarks

The suite ships several scripts that focus on different workload aspects:

* ``bench_qubit_scaling.py`` – sweeps the number of qubits for a fixed
  depth circuit to compare runtime scaling across devices.
* ``bench_depth_scaling.py`` – fixes the number of qubits and varies the
  circuit depth.
* ``bench_gradient_perf.py`` – measures parameter-shift style gradient
  evaluations on variational circuits.  Devices without gradient support
  are recorded as skipped.
* ``bench_accuracy_check.py`` – runs backend-agnostic accuracy checks
  (GHZ fidelity, single-qubit ``RY`` expectation errors, probability TV
  distance) and reports pass/fail status per device.
* ``bench_vqe_qaoa.py`` – benchmarks a 4-qubit VQE ansatz and a ring QAOA
  workload.

Example invocation (qubit scaling across several devices):

```bash
python benchmarks/bench_qubit_scaling.py \
  --devices qandle_sv_cpu,pl_default_qubit,pl_lightning_qubit \
  --min-qubits 4 --max-qubits 16 --step 2 --depth 5 \
  --dtype complex64 --reps 5 --out benchmarks/out/qubit_scaling.csv \
  --tags "workstationA"
```


## CSV schema

All benchmark scripts append rows using a unified schema:

| Column | Description |
| --- | --- |
| ``timestamp_iso`` | UTC timestamp of the run. |
| ``host`` | Hostname of the executing machine. |
| ``framework`` / ``family`` | Backend family (e.g., ``qandle`` or ``pennylane``). |
| ``device_key`` / ``label`` | Registry key and human-readable label. |
| ``sim_type`` | Simulator type (statevector, tensor, mps, …). |
| ``hardware`` | ``CPU`` or ``GPU``. |
| ``dtype`` / ``threads`` / ``bond_dim`` / ``splitting`` | Applied configuration knobs. |
| ``shots`` | Shot count (0 for analytic mode). |
| ``algorithm`` / ``scenario`` | Benchmark grouping identifiers. |
| ``n_qubits`` / ``depth`` / ``problem_id`` | Workload metadata. |
| ``seed`` / ``run_id`` | Random seed and sequential run identifier. |
| ``execution_time_s`` / ``execution_time_std_s`` | Mean wall time and standard deviation over ``reps`` repetitions. |
| ``peak_cpu_mb`` / ``peak_gpu_mb`` | Peak CPU/GPU memory usage (MB). |
| ``metric_name`` / ``metric_value`` | Algorithm-specific metric (e.g., fidelity, gradient norm). |
| ``success`` / ``error`` | Execution status and optional error message. |
| ``tags`` | User-supplied context string. |

GPU timings are synchronised via ``torch.cuda.synchronize`` where available.
CPU memory is sampled using ``tracemalloc`` and GPU peak memory comes from
PyTorch when a CUDA device is active.


## Comparing results

Use ``benchmarks/compare.py`` to compute pairwise speedups and memory
offsets relative to a baseline device:

```bash
python benchmarks/compare.py \
  --input benchmarks/out/qubit_scaling.csv \
  --group-by algorithm,n_qubits,depth \
  --baseline pl_lightning_qubit \
  --out benchmarks/out/qubit_scaling_vs_lightning.csv
```

The comparison CSV lists the speedup versus the baseline, CPU/GPU memory
deltas, and accuracy differences for every metric.  A short summary of the
fastest device per group is printed to stdout.


## Plotting

``benchmarks/plot_results.py`` renders quick visual summaries (requires
``matplotlib``):

```bash
python benchmarks/plot_results.py \
  --input benchmarks/out/qubit_scaling.csv \
  --output-prefix benchmarks/out/qubit_scaling_plot \
  --group-by algorithm,scenario \
  --x-axis n_qubits --metric execution_time_s
```

Plots are grouped by ``algorithm``/``scenario`` and annotate the backend
hardware and dtype in the legend, making it easy to spot CPU/GPU or dtype
trends at a glance.


## Reproducibility tips

* Always set ``--seed`` when comparing across machines.  The helper
  ``benchmarks.utils.stable_seed`` seeds Python ``random``, NumPy, and
  PyTorch.
* Record ``--tags`` to capture host information (driver versions, CUDA
  revisions, etc.).
* Use the same ``--devices`` ordering and ``--baseline`` argument when
  generating comparison reports to ensure consistent column grouping.

