"""Render original, artificial receipts; no customer data or benchmark images."""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def font(size: int, bold: bool = False):
    candidates = [
        Path("C:/Windows/Fonts/consolab.ttf" if bold else "C:/Windows/Fonts/consola.ttf"),
        Path(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
        ),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def main():
    destination = ROOT / "frontend/public/samples"
    destination.mkdir(parents=True, exist_ok=True)
    samples = [
        (
            "sample-a",
            "PAPER & BEAN",
            [("OAT LATTE", "4.50"), ("BUTTER CROISSANT", "3.80"), ("FILTER COFFEE", "3.20")],
            "11.50",
            "0.92",
            "12.42",
        ),
        (
            "sample-b",
            "THE GREEN MARKET",
            [
                ("SOURDOUGH LOAF", "5.20"),
                ("OLIVE OIL", "9.90"),
                ("SEASONAL APPLES", "4.40"),
                ("OAT MILK", "3.50"),
            ],
            "23.00",
            "1.84",
            "24.84",
        ),
        (
            "sample-c",
            "STUDIO SUPPLY",
            [("SKETCHBOOK", "12.00"), ("GRAPHITE PENCILS", "6.50"), ("PAPER TAPE", "4.25")],
            "22.75",
            "1.82",
            "24.57",
        ),
    ]
    targets = {}
    for key, merchant, items, subtotal, tax, total in samples:
        image = Image.new("RGB", (720, 1100), "#fcfbf8")
        draw = ImageDraw.Draw(image)
        draw.rectangle((24, 24, 696, 1076), outline="#ddd9d0", width=1)
        draw.text((360, 80), merchant, font=font(35, True), anchor="mt", fill="#262b25")
        draw.text(
            (360, 130), "SYNTHETIC PORTFOLIO SAMPLE", font=font(18), anchor="mt", fill="#78786f"
        )
        draw.text((62, 193), "01 OCT 2026    10:24", font=font(23), fill="#333630")
        draw.text(
            (62, 232), "RECEIPT # RL-" + key[-1].upper() + "001", font=font(23), fill="#333630"
        )
        draw.line((62, 291, 658, 291), fill="#9f9e96", width=2)
        draw.text((62, 322), "ITEM", font=font(22, True), fill="#333630")
        draw.text((658, 322), "AMOUNT", font=font(22, True), anchor="rt", fill="#333630")
        y = 380
        for name, amount in items:
            draw.text((62, y), name, font=font(25), fill="#262b25")
            draw.text((658, y), amount, font=font(25), anchor="rt", fill="#262b25")
            y += 63
        draw.line((62, 698, 658, 698), fill="#9f9e96", width=2)
        for y, name, value in [
            (736, "SUBTOTAL", subtotal),
            (785, "TAX", tax),
            (848, "TOTAL", total),
        ]:
            draw.text(
                (62, y),
                name,
                font=font(30 if name == "TOTAL" else 25, name == "TOTAL"),
                fill="#262b25",
            )
            draw.text(
                (658, y),
                value,
                font=font(30 if name == "TOTAL" else 25, name == "TOTAL"),
                anchor="rt",
                fill="#262b25",
            )
        draw.text(
            (360, 970), "THANK YOU. KEEP CREATING.", font=font(21), anchor="mt", fill="#78786f"
        )
        image.save(destination / f"{key}.png")
        targets[key] = {
            "merchant": merchant,
            "fields": {"subtotal": subtotal, "tax": tax, "discount": None, "total": total},
            "items": [{"description": name, "amount": amount} for name, amount in items],
        }
    (destination / "targets.json").write_text(json.dumps(targets, indent=2), encoding="utf-8")
    print(f"Rendered {len(samples)} synthetic receipts to {destination}")


if __name__ == "__main__":
    main()
