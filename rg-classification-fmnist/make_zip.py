"""Package small project files without datasets or model checkpoints."""

from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import tempfile
import zipfile


IGNORE_DIRS = {
    ".git", ".hg", ".svn", ".idea", ".vscode", ".venv", "venv", "env",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".tox", ".nox", "node_modules", "wandb", "lightning_logs",
    "tensorboard", "checkpoints", "checkpoint", "trained_models",
    "model_checkpoints", "saved_models", "weights", "tmp", "temp",
}

IGNORE_EXTS = {
    ".pt", ".pth", ".ckpt", ".safetensors", ".onnx",
    ".bin", ".idx", ".ubyte",
    ".zip", ".7z", ".rar", ".tar", ".gz", ".tgz", ".bz2", ".xz",
    ".pyc", ".pyo", ".log", ".tmp", ".bak", ".swp",
    ".pem", ".key", ".p12", ".pfx",
}

IGNORE_FILES = {
    ".env", ".env.local", ".env.production", "credentials.json",
    "service-account.json", "id_rsa", "id_ed25519", ".ds_store", "thumbs.db",
}

README_NAMES = ("README.txt", "README.md", "README.rst", "readme.txt", "readme.md")


def is_dataset_dir(parts: tuple[str, ...]) -> bool:
    """Identify dataset folders while retaining source-code directories."""
    lowered = tuple(p.lower() for p in parts)
    name = lowered[-1]
    if name in {"datasets", "dataset_files", "raw_datasets", "raw_data"}:
        return True
    if name != "data":
        return False

    if len(lowered) > 1 and lowered[-2] in {"core", "src", "source"}:
        return False
    return True


def is_noise_cache_file(rel: Path) -> bool:
    """Identify small cached noise-label tensors."""
    return "noise_cache" in (part.lower() for part in rel.parts)


def skip_reason(path: Path, rel: Path, limit_bytes: int) -> str | None:
    name = path.name.lower()
    if path.is_symlink():
        return "link simbolico"
    if name in IGNORE_FILES or name.startswith(".env."):
        return "arquivo privado/temporario"
    if name == "make_zip.py":
        return None
    ext = path.suffix.lower()
    if ext == ".pt" and is_noise_cache_file(rel):
        pass
    elif ext in IGNORE_EXTS:
        return "checkpoint/objeto binario/arquivo compactado"
    try:
        size = path.stat().st_size
    except OSError:
        return "erro de leitura"
    if size > limit_bytes:
        return "arquivo maior que o limite"
    return None


def collect(root: Path, output: Path, temporary: Path | None, limit_bytes: int):
    """Return selected files, exclusion counts, and traversal errors."""
    selected = []
    skipped = Counter()
    errors = []
    seen = set()

    def add_file(path: Path):
        if path == output or (temporary is not None and path == temporary):
            skipped["ZIP de saida"] += 1
            return
        if not path.is_file() and not path.is_symlink():
            return
        rel = path.relative_to(root)
        reason = skip_reason(path, rel, limit_bytes)
        if reason:
            skipped[reason] += 1
            return
        if rel in seen:
            return
        seen.add(rel)
        archive_name = (Path(root.name) / rel).as_posix()
        selected.append((path, archive_name))

    def on_error(error):
        errors.append(str(error))

    for current, dirnames, filenames in os.walk(root, topdown=True, followlinks=False, onerror=on_error):
        folder = Path(current)
        allowed = []
        for dirname in sorted(dirnames):
            child = folder / dirname
            rel = child.relative_to(root)
            if child.is_symlink():
                skipped["pasta com link simbolico"] += 1
            elif dirname.lower() in IGNORE_DIRS:
                skipped["pasta temporaria/checkpoint"] += 1
            elif is_dataset_dir(rel.parts):
                skipped["pasta de dataset"] += 1

                for readme in README_NAMES:
                    candidate = child / readme
                    if candidate.is_file():
                        add_file(candidate)
            else:
                allowed.append(dirname)
        dirnames[:] = allowed
        for filename in sorted(filenames):
            add_file(folder / filename)

    return selected, skipped, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent,
                        help="Pasta a varrer (padrao: pasta onde esta make_zip.py)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Caminho completo do ZIP (padrao: Downloads de Israel no Windows)")
    parser.add_argument("--max-mb", type=float, default=5.0,
                        help="Tamanho maximo individual por arquivo em MB (padrao: 5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Mostra o que seria incluido sem gerar arquivo")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"A pasta do projeto nao existe: {root}")
    if args.max_mb <= 0:
        parser.error("--max-mb deve ser maior que zero")

    downloads = Path.home() / "Downloads"

    output = (args.out.expanduser() if args.out else
              downloads / f"{root.name}_arquivos_pequenos.zip").resolve()
    limit_bytes = int(args.max_mb * 1024 * 1024)

    selected, skipped, errors = collect(root, output, None, limit_bytes)
    print(f"Pasta analisada: {root}")
    print(f"Arquivos selecionados: {len(selected)}")
    print(f"Tamanho original total: {sum(p.stat().st_size for p, _ in selected) / 1024 / 1024:.2f} MB")
    for reason, amount in sorted(skipped.items()):
        print(f"  Ignorados: {amount} ({reason})")
    if errors:
        print(f"ATENCAO: {len(errors)} erros ao percorrer pastas:")
        for message in errors[:20]:
            print("  ", message)

    if args.dry_run:
        print("Simulacao concluida; nenhum ZIP foi criado.")
        return
    if not selected:
        raise SystemExit("Nenhum arquivo selecionado; ZIP nao criado.")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".make_zip_", suffix=".tmp",
                                         dir=output.parent, delete=False) as file:
            temporary = Path(file.name)
        failures = []
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6, allowZip64=True) as archive:
            for path, archive_name in selected:
                try:

                    if skip_reason(path, path.relative_to(root), limit_bytes) is None:
                        archive.write(path, archive_name)
                    else:
                        failures.append(f"Arquivo mudou/desapareceu: {path}")
                except (OSError, ValueError) as exc:
                    failures.append(f"{path}: {exc}")

        if failures or errors:
            raise RuntimeError("ZIP incompleto, nao publicado:\n" +
                               "\n".join((errors + failures)[:30]))
        os.replace(temporary, output)
        temporary = None
        print(f"ZIP criado: {output}")
        print(f"Tamanho compactado: {output.stat().st_size / 1024 / 1024:.2f} MB")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
