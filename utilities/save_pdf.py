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
    # Diagnostic logging
    try:
        print(f"📄 save_pdf called: filename={pdf_name} std_path={std_path}")
        content_len = len(content) if content is not None else 0
        print(f"📄 LaTeX content length: {content_len} bytes")

        tex_path.write_text(content, encoding="utf-8")

        compiler = shutil.which("xelatex")
        if not compiler:
            msg = "xelatex is required to generate PDF files"
            print(f"❌ save_pdf error: {msg}")
            raise RuntimeError(msg)

        print(f"📄 Running xelatex: {compiler} on {tex_path}")

        proc = subprocess.run(
            [
                compiler,
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-output-directory",
                str(output_dir),
                str(tex_path),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        stdout = proc.stdout.decode(errors="ignore")
        stderr = proc.stderr.decode(errors="ignore")

        print(f"📄 xelatex exit code: {proc.returncode}")
        if stdout:
            print("📄 xelatex stdout:\n" + stdout)
        if stderr:
            print("📄 xelatex stderr:\n" + stderr)

        if proc.returncode != 0:
            raise RuntimeError(f"xelatex failed with exit code {proc.returncode}")

        # Cleanup auxiliary files
        for suffix in (".aux", ".log", ".out", ".toc"):
            temp_file = pdf_path.with_suffix(suffix)
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass

        print(f"✅ PDF generated: {pdf_path}")
        return str(pdf_path.resolve())

    except Exception as exc:
        print(f"❌ Unexpected error in save_pdf: {exc}")
        raise
