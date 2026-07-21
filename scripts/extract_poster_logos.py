from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "poster" / "Lapin_Vladislav_DoRA_AIRI_2026_RU.pptx"
OUT = ROOT / "poster" / ".build" / "assets"


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    OUT.mkdir(parents=True, exist_ok=True)

    images: list[tuple[str, bytes, int, int, float]] = []
    with ZipFile(SOURCE) as archive:
        for name in archive.namelist():
            if not name.startswith("ppt/media/") or name.endswith("/"):
                continue
            raw = archive.read(name)
            try:
                with Image.open(BytesIO(raw)) as image:
                    width, height = image.size
                    ratio = width / height
            except Exception:
                continue
            images.append((name, raw, width, height, ratio))

    wide = [item for item in images if item[2] >= 500 and item[4] >= 3.0]
    if len(wide) < 3:
        raise RuntimeError(
            f"Could not identify header and partner logos in {SOURCE}; "
            f"candidates={[(n, w, h, r) for n, _, w, h, r in wide]}"
        )

    header = max(wide, key=lambda item: item[2])
    partners = [item for item in wide if item is not header and item[2] <= 1200]
    if len(partners) < 2:
        raise RuntimeError(
            f"Could not identify both partner logos; "
            f"candidates={[(n, w, h, r) for n, _, w, h, r in partners]}"
        )

    sber = min(partners, key=lambda item: item[4])
    avito = max(partners, key=lambda item: item[4])

    for filename, item in [
        ("header_logos.png", header),
        ("sber_logo.png", sber),
        ("avito_logo.png", avito),
    ]:
        (OUT / filename).write_bytes(item[1])
        print(filename, item[0], item[2], item[3], f"ratio={item[4]:.3f}")


if __name__ == "__main__":
    main()
