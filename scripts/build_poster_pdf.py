"""Wrap the verified high-resolution poster render in an exact ISO A1 PDF."""

from __future__ import annotations

import argparse
from pathlib import Path

from reportlab.lib.units import mm
from reportlab.pdfgen.canvas import Canvas


A1_PORTRAIT = (594 * mm, 841 * mm)


def build_pdf(image_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    page_width, page_height = A1_PORTRAIT

    pdf = Canvas(str(output_path), pagesize=A1_PORTRAIT, pageCompression=1)
    pdf.setTitle("DoRA vs LoRA: when does weight decomposition help?")
    pdf.setAuthor("Vladislav Lapin")
    pdf.setSubject("AIRI Summer School 2026 research poster")
    pdf.setCreator("Artifact Tool render + ReportLab A1 wrapper")
    pdf.drawImage(
        str(image_path),
        0,
        0,
        width=page_width,
        height=page_height,
        preserveAspectRatio=True,
        anchor="c",
        mask="auto",
    )
    pdf.showPage()
    pdf.save()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build_pdf(args.image, args.output)


if __name__ == "__main__":
    main()
