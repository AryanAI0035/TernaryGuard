# Completed-core packaging validation — 2026-10-05

The main deliverable is now the frozen model, C engine and physically validated Nano research prototype. FPGA and dashboard are optional future work. This packaging pass does not retrain the model or change measured historical results.

## Regressions after relocation

Command from the reorganized repository:

```sh
python3 -m pytest model/tests/ future-scope/fpga/tests/ -v --tb=short
```

Observed: **107 passed in 66.92s**. [Full raw output](benchmarks/core_release/combined-tests.txt).

The eight FPGA tests were moved to `future-scope/fpga/tests/`. The remaining 99 core tests are selected by the root pytest configuration. Tests were segregated, not removed.

## Fresh-clone content check

A snapshot was extracted from the staged tracked files with `git checkout-index --all --prefix=/private/tmp/tg-scope-clean/`. It contained no raw dataset, ignored builds or historical checkpoints. In that directory:

```sh
python3 scripts/demo.py
python3 -m pytest -v --tb=short
```

Observed: demo passed; **99 passed in 56.65s**. [Full core-test output](benchmarks/core_release/clean-core-tests.txt). This uses the workstation's installed Python dependencies/compiler; it verifies tracked-file completeness, not a separately provisioned operating system. Linux CI runs separately on GitHub.

The newly versioned reference consists only of the unchanged final checkpoint (30,867 bytes), data manifest and frozen split identities. Other checkpoints and raw data remain excluded. [Bundled replay output](benchmarks/core_release/demo.txt) compares 264 preprocessed engineering vectors against the live frozen checkpoint; it is not a dataset-accuracy rerun.

## Exact active artifacts

```sh
python3 model/validate_export.py --verify-active
```

Observed: **69,040 samples, 69,040 matches, zero disagreements**, maximum absolute logit error **1.1444091796875e-05**, `artifacts_rewritten: false`. [Actual JSON output](benchmarks/core_release/active-verification.json).

Checkpoint, active identity, existing header, preprocessing, export manifest, C engine and AVR inference source had identical before/after byte hashes. [Protected hashes](benchmarks/core_release/protected-hashes.json) record byte hashes; the export manifest's preprocessing identity uses canonical JSON instead.

## Optional FPGA handoff

The relocated package generator was exercised on all frozen vectors and its independent extractor/verifier reported `HANDOFF_PASS files=38`. The archive still contains `engine-fpga/`, so existing borrowed-laptop commands work. Original transfer archives remain intact locally; the current repository keeps extension sources under `future-scope/fpga/`. This is packaging verification, not Vivado acceptance.

The known BASHLITE TCP limitation stays **1/5,555 (0.018%)**, at test index 59782; no training, inference or hardware result was changed. Physical coverage and timing remain those in [the Nano acceptance record](phase5_hardware_validation.md).
