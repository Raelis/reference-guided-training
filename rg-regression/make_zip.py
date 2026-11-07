"""Package M4 source and compact experiment summaries without heavy predictions.

Examples:
    python make_zip.py --dry-run
    python make_zip.py
    python make_zip.py --include-predictions --include-baselines
    python make_zip.py --max-mb 10
    python make_zip.py --out ./m4_release.zip
"""

from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import tempfile
import zipfile


# Keep code, documentation, plots, Optuna databases, aggregated CSVs and small
# experiment_data.parquet files; exclude raw prediction arrays by default.
IGNORE_DIRS = {
    ".git", ".hg", ".svn", ".idea", ".vscode", ".venv", "venv", "env",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".tox", ".nox", "node_modules", "wandb", "lightning_logs",
    "tensorboard", "checkpoints", "checkpoint", "trained_models",
    "model_checkpoints", "saved_models", "weights", "tmp", "temp",
    ".ssh", ".aws", ".cache",
}

IGNORE_EXTS = {
    ".pt", ".pth", ".ckpt", ".safetensors", ".onnx",
    ".bin", ".idx", ".ubyte", ".pkl", ".pickle", ".joblib",
    ".h5", ".hdf5",
    ".zip", ".7z", ".rar", ".tar", ".gz", ".tgz", ".bz2", ".xz",
    ".pyc", ".pyo", ".log", ".tmp", ".bak", ".swp",
    ".pem", ".key", ".p12", ".pfx",
}

IGNORE_FILES = {
    ".env", ".env.local", ".env.production", "credentials.json",
    "service-account.json", "id_rsa", "id_ed25519", ".ds_store",
    "thumbs.db", ".netrc", ".pypirc", ".npmrc",
}

README_NAMES = ("README.md", "README.txt", "README.rst", "readme.md", "readme.txt")

DATASET_DIR_NAMES = {
    "datasets", "dataset_files", "raw_datasets", "raw_data", "m4_data",
    "m4_dataset", "m4_datasets", "m4-dataset", "m4-datasets",
}


def is_dataset_dir(relative_parts: tuple[str, ...]) -> bool:
    """Exclude M4 dataset directories without hiding source modules named data."""
    parts = tuple(p.lower() for p in relative_parts)
    folder = parts[-1]
    if folder in DATASET_DIR_NAMES:
        return True
    if folder == "m4" and (len(parts) == 1 or "data" in parts[:-1] or "datasets" in parts[:-1]):
        return True
    if folder in {"data", "dataset"}:
        # Preserve e.g. core/data/ or src/data/ if they contain Python source.
        if len(parts) > 1 and parts[-2] in {"core", "src", "source"}:
            return False
        return True
    return False


HEAVY_PREDICTION_FILES = {
    "m4_predictions.parquet", "predictions.parquet", "prediction_cache.parquet",
}


def skip_reason(
    path: Path, relative: Path, byte_limit: int,
    include_predictions: bool = False, include_baselines: bool = False,
) -> str | None:
    """Explain why an individual file must not enter the archive."""
    if path.is_symlink():
        return "symbolic link"
    name = path.name.lower()
    parts = tuple(part.lower() for part in relative.parts)
    if not include_predictions and "results" in parts and name in HEAVY_PREDICTION_FILES:
        return "stored per-sample predictions"
    if (not include_baselines and "results" in parts and "baselines" in parts
            and path.suffix.lower() == ".npz"):
        return "precomputed baseline arrays"
    if name in IGNORE_FILES or name.startswith(".env."):
        return "private/local file"
    if path.suffix.lower() in IGNORE_EXTS:
        return "checkpoint/binary/archive/temporary file"
    try:
        if path.stat().st_size > byte_limit:
            return "file exceeds size limit"
    except OSError:
        return "unreadable file"
    return None


def collect(
    root: Path, output: Path, byte_limit: int,
    include_predictions: bool = False, include_baselines: bool = False,
):
    """Return archive entries, skip statistics and filesystem errors."""
    selected: list[tuple[Path, str]] = []
    skipped: Counter[str] = Counter()
    errors: list[str] = []

    def add(path: Path) -> None:
        if path == output:
            skipped["output ZIP"] += 1
            return
        if not path.is_file() and not path.is_symlink():
            return
        relative = path.relative_to(root)
        reason = skip_reason(path, relative, byte_limit, include_predictions, include_baselines)
        if reason is not None:
            skipped[reason] += 1
            return
        selected.append((path, (Path(root.name) / relative).as_posix()))

    for current, directories, filenames in os.walk(
        root, topdown=True, followlinks=False, onerror=lambda exc: errors.append(str(exc))
    ):
        folder = Path(current)
        allowed = []
        for dirname in sorted(directories):
            child = folder / dirname
            relative = child.relative_to(root)
            if child.is_symlink():
                skipped["symbolic-link directory"] += 1
            elif dirname.lower() in IGNORE_DIRS:
                skipped["temporary/checkpoint directory"] += 1
            elif is_dataset_dir(relative.parts):
                skipped["dataset directory"] += 1
                # Keep a dataset's README/license, but never the dataset itself.
                for readme in README_NAMES:
                    readme_path = child / readme
                    if readme_path.is_file():
                        add(readme_path)
                for license_name in ("LICENSE", "LICENSE.txt", "LICENSE.md"):
                    license_path = child / license_name
                    if license_path.is_file():
                        add(license_path)
            else:
                allowed.append(dirname)
        directories[:] = allowed
        for filename in sorted(filenames):
            add(folder / filename)

    selected.sort(key=lambda item: item[1])
    return selected, skipped, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent,
        help="Project root (default: directory containing make_zip.py)",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output path (default: ~/Downloads/<project>_arquivos_pequenos.zip)",
    )
    parser.add_argument(
        "--max-mb", type=float, default=5.0,
        help="Maximum size of each included file in MiB (default: 5)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List selection counts without creating an archive",
    )
    parser.add_argument(
        "--include-predictions", action="store_true",
        help="Include per-sample prediction Parquet files (may add ~80 MiB)",
    )
    parser.add_argument(
        "--include-baselines", action="store_true",
        help="Include precomputed baseline .npz arrays (may add ~14 MiB)",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Project directory not found: {root}")
    if args.max_mb <= 0:
        parser.error("--max-mb must be positive")

    output = (
        args.out.expanduser()
        if args.out is not None
        else Path.home() / "Downloads" / f"{root.name}_arquivos_pequenos.zip"
    ).resolve()
    byte_limit = int(args.max_mb * 1024 * 1024)

    selected, skipped, errors = collect(
        root, output, byte_limit, args.include_predictions, args.include_baselines
    )
    total_bytes = sum(path.stat().st_size for path, _ in selected)
    print(f"Project: {root}")
    print(f"Selected files: {len(selected)} ({total_bytes / 1024 / 1024:.2f} MiB)")
    for reason, count in sorted(skipped.items()):
        print(f"  Excluded: {count} ({reason})")
    for error in errors[:20]:
        print(f"  Traversal error: {error}")

    if errors:
        raise SystemExit("Aborted: project traversal was incomplete")
    if args.dry_run:
        print("Dry run finished: no ZIP created")
        return
    if not selected:
        raise SystemExit("No eligible files; ZIP not created")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".make_zip_", suffix=".tmp", dir=output.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)

        failures: list[str] = []
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED,
            compresslevel=6, allowZip64=True,
        ) as archive:
            for path, archive_name in selected:
                try:
                    reason = skip_reason(
                        path, path.relative_to(root), byte_limit,
                        args.include_predictions, args.include_baselines,
                    )
                    if reason is not None:
                        failures.append(f"File changed since selection: {path} ({reason})")
                    else:
                        archive.write(path, archive_name)
                except (OSError, ValueError) as exc:
                    failures.append(f"{path}: {exc}")
        if failures:
            raise RuntimeError("ZIP not published because files changed/failed:\n" +
                               "\n".join(failures[:30]))
        os.replace(temporary, output)
        temporary = None
        print(f"ZIP created: {output}")
        print(f"ZIP size: {output.stat().st_size / 1024 / 1024:.2f} MiB")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
