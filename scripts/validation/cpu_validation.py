"""CPU validation harness for scientific stack upgrades.

The harness verifies that NumPy, SciPy, and scikit-learn operate correctly on the
current platform by running a series of deterministic smoke tests. The script
exits with status code 0 when all checks pass and 1 when any check fails unless
``--allow-failures`` is provided.

Usage
-----
    uv run python scripts/validation/cpu_validation.py --output artifacts/cpu.json
"""

# pylint: disable=duplicate-code,import-outside-toplevel
from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class CheckResult:
    """Outcome of an individual validation step."""

    name: str
    status: str
    detail: str | None = None


@dataclass(slots=True)
class ValidationReport:
    """Structured report emitted by the CPU validation harness."""

    # pylint: disable=too-many-instance-attributes

    status: str
    python_version: str
    platform: str
    libc: tuple[str, str]
    numpy_version: str | None = None
    scipy_version: str | None = None
    sklearn_version: str | None = None
    blas_info: dict[str, Any] | None = None
    checks: list[CheckResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass to a JSON-serialisable dictionary."""
        payload = asdict(self)
        payload["checks"] = [asdict(check) for check in self.checks]
        return payload


def _numpy_checks() -> tuple[CheckResult, str, dict[str, Any]]:
    """Perform numpy checks and return result, version, and blas info."""
    import numpy as np

    rng = np.random.default_rng(42)
    matrix = rng.random((128, 64))
    gram = matrix @ matrix.T
    spectral_radius = float(np.linalg.eigvals(gram)[0].real)
    config_summary = np.__config__.show(mode="dicts")
    build_deps = config_summary.get("Build Dependencies", {})
    raw_blas_info = build_deps.get("blas") or {}
    blas_info: dict[str, Any]
    if isinstance(raw_blas_info, Mapping):
        blas_info = dict(raw_blas_info)
    else:
        blas_info = {"value": raw_blas_info}
    detail = (
        f"Gram matrix computed with shape {gram.shape}; "
        f"spectral radius {spectral_radius:.4f}"
    )
    return (
        CheckResult("numpy-linear-algebra", "passed", detail),
        np.__version__,
        blas_info,
    )


def _scipy_checks() -> tuple[CheckResult, str]:
    """Perform scipy checks and return result and version."""
    import scipy  # pyright: ignore[reportMissingTypeStubs]
    from scipy import linalg  # pyright: ignore[reportMissingTypeStubs]

    hilbert = linalg.hilbert(16)
    inv_norm = float(linalg.norm(linalg.inv(hilbert)))
    detail = f"Inverted Hilbert matrix (order 16) with norm {inv_norm:.4f}"
    return CheckResult("scipy-linalg", "passed", detail), scipy.__version__


def _sklearn_checks() -> tuple[list[CheckResult], str]:
    """Perform sklearn checks and return results and version."""
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.manifold import SpectralEmbedding

    rng = np.random.default_rng(42)
    data = rng.normal(size=(256, 16))

    pca = PCA(n_components=8, random_state=42)
    components = pca.fit_transform(data)

    embedder = SpectralEmbedding(n_components=2, random_state=42)
    embedding = embedder.fit_transform(components)

    kmeans = KMeans(n_clusters=4, n_init="auto", random_state=42)
    clusters = kmeans.fit_predict(components)

    results = [
        CheckResult(
            "sklearn-pca",
            "passed",
            f"PCA explained variance sum {pca.explained_variance_ratio_.sum():.4f}",
        ),
        CheckResult(
            "sklearn-spectral-embedding",
            "passed",
            f"Spectral embedding shape {embedding.shape}",
        ),
        CheckResult(
            "sklearn-kmeans",
            "passed",
            f"Cluster distribution: {np.bincount(clusters).tolist()}",
        ),
    ]
    import sklearn

    return results, sklearn.__version__


def run_cpu_validation() -> ValidationReport:
    """Execute the CPU validation harness and return a structured report."""
    checks: list[CheckResult] = []
    numpy_version: str | None = None
    scipy_version: str | None = None
    sklearn_version: str | None = None
    blas_info: dict[str, Any] | None = None

    def _append_linalg_failure(check_id: str, exc: Exception) -> None:
        """Append a failure result while attempting to classify LinAlg errors."""
        detail = str(exc)
        try:
            import numpy as _np
        except Exception:  # pragma: no cover - import edge case
            checks.append(CheckResult(check_id, "failed", detail))
            return

        lin_alg_error = getattr(_np.linalg, "LinAlgError", None)
        if lin_alg_error and isinstance(exc, lin_alg_error):
            detail = f"LinAlgError: {exc}"
        checks.append(CheckResult(check_id, "failed", detail))

    try:
        numpy_check, numpy_version, blas_info = _numpy_checks()
        checks.append(numpy_check)
    except ImportError as exc:  # pragma: no cover - exercised in tests
        checks.append(
            CheckResult("numpy-linear-algebra", "failed", f"ImportError: {exc}")
        )
    except ValueError as exc:  # pragma: no cover - exercised in tests
        checks.append(
            CheckResult("numpy-linear-algebra", "failed", f"ValueError: {exc}")
        )
    except Exception as exc:  # pragma: no cover - exercised in tests
        _append_linalg_failure("numpy-linear-algebra", exc)

    try:
        scipy_check, scipy_version = _scipy_checks()
        checks.append(scipy_check)
    except ImportError as exc:  # pragma: no cover - exercised in tests
        checks.append(CheckResult("scipy-linalg", "failed", f"ImportError: {exc}"))
    except ValueError as exc:  # pragma: no cover - exercised in tests
        checks.append(CheckResult("scipy-linalg", "failed", f"ValueError: {exc}"))
    except Exception as exc:  # pragma: no cover - exercised in tests
        _append_linalg_failure("scipy-linalg", exc)

    try:
        sklearn_checks, sklearn_version = _sklearn_checks()
        checks.extend(sklearn_checks)
    except ImportError as exc:  # pragma: no cover - exercised in tests
        checks.append(CheckResult("sklearn-suite", "failed", f"ImportError: {exc}"))
    except ValueError as exc:  # pragma: no cover - exercised in tests
        checks.append(CheckResult("sklearn-suite", "failed", f"ValueError: {exc}"))
    except Exception as exc:  # pragma: no cover - exercised in tests
        checks.append(CheckResult("sklearn-suite", "failed", str(exc)))

    status = "passed" if all(check.status == "passed" for check in checks) else "failed"

    return ValidationReport(
        status=status,
        python_version=platform.python_version(),
        platform=platform.platform(),
        libc=platform.libc_ver(),
        numpy_version=numpy_version,
        scipy_version=scipy_version,
        sklearn_version=sklearn_version,
        blas_info=blas_info,
        checks=checks,
    )


def main() -> int:
    """Main entry point for the CPU validation script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the validation report as JSON.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentation level for JSON output.",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Return exit code 0 even if checks fail (useful for diagnostics).",
    )
    args = parser.parse_args()

    report = run_cpu_validation()
    payload = report.to_dict()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=args.indent), encoding="utf-8"
        )
    else:
        print(json.dumps(payload, indent=args.indent))

    if report.status == "failed" and not args.allow_failures:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
