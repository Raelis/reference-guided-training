# Validate vector PDF font embedding for figure exports.
from pathlib import Path
import re
import shutil
import subprocess


def verify_pdf_fonts(file_path):
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF figure was not produced: {path}")
    data = path.read_bytes()
    if not data.startswith(b"%PDF-"):
        raise RuntimeError(f"Not a PDF: {path}")
    if re.search(rb"/Subtype\s*/Type3\b", data):
        raise RuntimeError(
            f"NONCOMPLIANT PDF FONT (Type 3) in {path}. "
            "The generator is not using the corrected font configuration. "
            "Please, check which core/evaluation_distributions.py Python is importing."
        )
    has_text_fonts = bool(re.search(
        rb"/Subtype\s*/(?:CIDFontType2|TrueType|Type1|CIDFontType0)\b", data
    ))
    pdffonts = shutil.which("pdffonts")
    if pdffonts:
        result = subprocess.run([pdffonts, str(path)], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"pdffonts failed for {path}: {result.stderr}")
        for line in result.stdout.splitlines()[2:]:
            if "Type 3" in line:
                raise RuntimeError(f"Type 3 font detected in {path}: {line}")
            parts = line.split()
            if len(parts) >= 8 and parts[-5].lower() != 'yes':
                raise RuntimeError(f"Nonembedded font detected in {path}: {line}")
    print(f"[PDF font OK] {path} "
          + ("(embedded text fonts, no Type 3)" if has_text_fonts
             else "(no Type 3; no font objects, possibly outlined text)"),
          flush=True)
    return path


def verify_figure_directory(folder):
    folder = Path(folder)
    files = sorted(folder.glob("*.pdf"))
    if not files:
        raise FileNotFoundError(f"No PDF figures found in {folder}")
    errors = []
    for path in files:
        try:
            verify_pdf_fonts(path)
        except (OSError, RuntimeError) as exc:
            errors.append(str(exc))
    if errors:
        raise RuntimeError(f"{len(errors)} of {len(files)} PDF figures failed validation:\n" + "\n".join(errors))
    print(f"[PDF font validation] PASS: {len(files)} figures", flush=True)
    return files
