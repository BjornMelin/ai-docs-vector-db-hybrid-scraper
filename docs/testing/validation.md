# Validation Harnesses

The validation harnesses provide deterministic smoke tests for environments that
run the upgraded scientific and GPU stacks. They are designed to fail fast when
incompatible versions of NumPy, SciPy, scikit-learn, PyTorch, or xFormers are
present.

## CPU Validation

Run the CPU harness locally to confirm BLAS and high-level algorithms still
behave as expected after dependency upgrades:

```bash
uv run python scripts/validation/cpu_validation.py --output artifacts/cpu-validation-report.json
```

The script executes the following checks:

- NumPy linear algebra (Gram matrix, eigenvalues, BLAS backend introspection)
- SciPy Hilbert matrix inversion
- scikit-learn PCA, spectral embedding, and KMeans clustering on deterministic input

The JSON report includes the interpreter version, libc build, the detected BLAS
provider, and pass/fail states for every sub-check. The CLI exits with status 1
if any check fails unless `--allow-failures` is supplied for diagnostic runs.

## GPU Validation

Run the GPU harness on CUDA/ROCm-capable hosts to verify deep learning
extensions:

```bash
uv run python scripts/validation/gpu_validation.py --output artifacts/gpu-validation-report.json
```

The GPU job performs:

- PyTorch CUDA availability checks, matrix multiplication smoke test, and memory
  telemetry capture
- xFormers `memory_efficient_attention` execution
- FlashAttention import verification (treated as a warning if unavailable)

By default the harness fails (exit code 1) when no GPU is detected. CI pipelines
should invoke the CLI without flags so missing devices block the pipeline. Local
developers on CPU-only hardware may pass `--allow-missing-gpu` to downgrade the
error to a warning for observational runs.

## CI Integration

`.github/workflows/validation.yml` runs both harnesses. The CPU job executes on
GitHub-hosted Ubuntu runners while the GPU job targets self-hosted runners tagged
with `gpu`. Both steps upload their JSON output as build artefacts for auditing.
