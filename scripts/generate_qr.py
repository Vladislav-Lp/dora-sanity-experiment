from pathlib import Path

import qrcode
from qrcode.constants import ERROR_CORRECT_H


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "poster" / "assets" / "github_qr.png"
URL = "https://github.com/Vladislav-Lp/dora-sanity-experiment"


def main() -> None:
    qr = qrcode.QRCode(version=None, error_correction=ERROR_CORRECT_H, box_size=18, border=4)
    qr.add_data(URL)
    qr.make(fit=True)
    image = qr.make_image(fill_color="black", back_color="white")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT)
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
