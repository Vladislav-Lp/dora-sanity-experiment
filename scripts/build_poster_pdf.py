"""Export the primary Russian AIRI poster to a print-ready A1 PDF with LibreOffice."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "poster" / "Lapin_Vladislav_DoRA_AIRI_2026_RU.pptx"
DEFAULT_OUTPUT = ROOT / "poster" / "Lapin_Vladislav_DoRA_AIRI_2026_RU.pdf"


def export_pdf(input_path: Path, output_path: Path) -> None:
    office = shutil.which("soffice") or shutil.which("libreoffice")
    if office is None:
        raise RuntimeError(
            "LibreOffice is required for reproducible command-line export. "
            "Alternatively, export the supplied A1 PPTX from PowerPoint."
        )

    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="airi-poster-") as tmp:
        tmp_dir = Path(tmp)
        profile = tmp_dir / "libreoffice-profile"
        export_dir = tmp_dir / "export"
        profile.mkdir()
        export_dir.mkdir()

        subprocess.run(
            [
                office,
                f"-env:UserInstallation={profile.as_uri()}",
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                str(export_dir),
                str(input_path),
            ],
            check=True,
        )

        generated = export_dir / f"{input_path.stem}.pdf"
        if not generated.is_file():
            raise RuntimeError(f"LibreOffice did not create {generated}")
        shutil.copy2(generated, output_path)

    pdfinfo = shutil.which("pdfinfo")
    if pdfinfo:
        info = subprocess.run(
            [pdfinfo, str(output_path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if "Pages:           1" not in info or "(A1)" not in info:
            raise RuntimeError("Exported PDF is not a one-page ISO A1 document.")

    print(f"saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    export_pdf(args.input, args.output)


if __name__ == "__main__":
    main()
