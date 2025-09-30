=============================
Benchmarking and Comparisons
=============================

The benchmarking suite measures the runtime and peak memory usage of
representative circuits across multiple simulators:

* the current :mod:`qandle` statevector backend,
* the :mod:`qandle` matrix-product-state (MPS) backend,
* the ``default.qubit.torch`` device provided by PennyLane, and
* an optional legacy :mod:`qandle` release installed from PyPI.

Circuit families
================

The suite covers three styles of workload to highlight different stress
points:

``small_verification``
    4 qubits with a moderately deep strongly entangling layer that is useful for
    regression-style verification of algorithmic changes.

``deep_narrow``
    5 qubits with 32 layers of strongly entangling gates to probe depth scaling
    on narrow systems.

``wide_low_entanglement``
    12 qubits with a shallow two-local ansatz that keeps entanglement low while
    emphasising width-related overhead.

Each configuration is deterministic: gate parameters are sampled using a
repeatable seed so that different backends receive identical workloads.

Dependencies
============

The benchmarking scripts live under ``scripts/benchmarks/`` and rely on the
standard project dependencies plus a small set of optional packages:

``PennyLane``
    Required to run the reference ``default.qubit.torch`` simulator. Install it
    via ``pip install pennylane`` or by using ``pip install 'qandle[dev]'``.

``matplotlib`` *(optional)*
    Enables automatic plot generation. Without it, CSV reports are still
    produced.

``psutil`` *(optional, recommended on Windows)*
    Provides access to peak process memory information on platforms where the
    standard ``resource`` module is unavailable (such as Windows). Without it,
    benchmark runs still succeed but memory figures will be reported as ``NaN``.


``legacy qandle`` *(optional)*
    To benchmark the most recent PyPI release, install it into a separate
    directory and point the suite to that location. For example::

        python -m pip install "qandle==0.1.12" --target .benchmarks/legacy

    Then pass ``--legacy-site-packages .benchmarks/legacy`` when invoking the
    benchmarks. Alternatively, add ``--legacy-auto-install`` to let the script
    download and cache the requested release under ``~/.cache/qandle-benchmarks``.

Running the suite
=================

The main entry point is ``scripts/benchmarks/run.py``. Example invocations::

    # Run the lightweight smoke configuration used in CI
    python scripts/benchmarks/run.py --preset smoke --no-plots

    # Execute the full suite with plots and cached legacy installation
    python scripts/benchmarks/run.py --preset full --legacy-auto-install

Results are written to ``scripts/benchmarks/results`` by default. Two CSV files
are produced:

``benchmark_results.csv``
    Raw per-run measurements with metadata, status and diagnostic details.

``benchmark_summary.csv``
    Aggregated averages across successful runs for each circuit/backend pair.

When ``matplotlib`` is available, bar charts for runtime and peak memory are
saved in the same directory. Use ``--output-dir`` to collect the artefacts in a
custom location.

Continuous integration
======================

The GitHub Actions workflow ``python-test.yml`` executes
``python scripts/benchmarks/run.py --preset smoke --no-plots`` after the test
suite. This smoke check keeps runtime minimal while ensuring the benchmarking
entry point continues to work.
