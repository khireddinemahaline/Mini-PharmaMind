from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def save_pdf(content: str, filename: str, std_path: str = "generated_reports") -> str:
    """Save LaTeX content as a PDF file using xelatex."""
    output_dir = Path(std_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_name = filename if filename.endswith(".pdf") else f"{filename}.pdf"
    tex_name = Path(pdf_name).with_suffix(".tex").name
    tex_path = output_dir / tex_name
    pdf_path = output_dir / pdf_name

    tex_path.write_text(content, encoding="utf-8")

    compiler = shutil.which("xelatex")
    if not compiler:
        raise RuntimeError("xelatex is required to generate PDF files")

    subprocess.run(
        [
            compiler,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-output-directory",
            str(output_dir),
            str(tex_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for suffix in (".aux", ".log", ".out", ".toc"):
        temp_file = pdf_path.with_suffix(suffix)
        if temp_file.exists():
            temp_file.unlink()

    return str(pdf_path.resolve())
