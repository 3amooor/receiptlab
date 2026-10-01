from __future__ import annotations

import threading

import numpy as np
from PIL import Image

_ENGINE = None
_LOCK = threading.Lock()


def run_ocr(image: Image.Image):
    global _ENGINE
    rgb = image.convert("RGB")
    width, height = rgb.size
    if width < 2 or height < 2:
        raise ValueError("Image is too small for OCR.")
    with _LOCK:
        if _ENGINE is None:
            from rapidocr_onnxruntime import RapidOCR

            _ENGINE = RapidOCR()
        result, _ = _ENGINE(np.asarray(rgb))
    lines = []
    for box, text, confidence in result or []:
        coordinates = np.asarray(box, dtype=float)
        normalized = [
            coordinates[:, 0].min() / width,
            coordinates[:, 1].min() / height,
            coordinates[:, 0].max() / width,
            coordinates[:, 1].max() / height,
        ]
        lines.append(
            {
                "text": str(text).strip(),
                "bbox": [min(1.0, max(0.0, float(value))) for value in normalized],
                "confidence": float(confidence),
            }
        )
    lines.sort(key=lambda line: ((line["bbox"][1] + line["bbox"][3]) / 2, line["bbox"][0]))
    return lines
