"""Durable document processing; one local worker owns OCR and model inference."""

import asyncio
import math
import os
from pathlib import Path

from PIL import Image

from .schemas import MONEY_FIELDS, ReviewItem, financial_warnings, normalize_money
from .store import Store


def score(value) -> float:
    try:
        number = float(value)
        return max(0.0, min(1.0, number)) if math.isfinite(number) else 0.0
    except (TypeError, ValueError):
        return 0.0


def box(value) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    output = [score(item) for item in value]
    return output if output[2] >= output[0] and output[3] >= output[1] else None


def normalize_extraction(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("Invalid extraction result")
    warnings = [str(item)[:240] for item in raw.get("warnings", [])[:30]]
    fields = {}
    for name in MONEY_FIELDS:
        field = raw.get("fields", {}).get(name) or {}
        try:
            value = normalize_money(field.get("value"))
        except ValueError:
            value = None
            warnings.append(f"The extracted {name} needs a valid decimal amount.")
        fields[name] = {
            "value": value,
            "confidence": score(field.get("confidence")),
            "bbox": box(field.get("bbox")),
            "origin": str(field.get("origin", "learned_model"))[:40],
        }
    items = []
    for item in raw.get("items", [])[:250]:
        try:
            validated = ReviewItem.model_validate(
                {key: item.get(key) for key in ("description", "quantity", "unit_price", "amount")}
            )
            items.append({**validated.model_dump(), "confidence": score(item.get("confidence"))})
        except (ValueError, TypeError):
            warnings.append("An extracted line item needs manual correction.")
    lines = []
    for line in raw.get("lines", [])[:500]:
        if isinstance(line, dict):
            lines.append(
                {
                    "text": str(line.get("text", ""))[:500],
                    "bbox": box(line.get("bbox")),
                    "confidence": score(line.get("confidence")),
                    "label": str(line.get("label", "other"))[:40],
                    "model_confidence": score(line.get("model_confidence")),
                }
            )
    warnings.extend(financial_warnings(fields, items))
    warnings = list(dict.fromkeys(warnings))
    result = {
        "fields": fields,
        "items": items,
        "lines": lines,
        "warnings": warnings,
        "model_version": str(raw.get("model_version", "unknown"))[:120],
        "confidence": score(raw.get("confidence")),
        "needs_review": bool(raw.get("needs_review", True)) or bool(warnings),
        "extraction_mode": str(raw.get("extraction_mode", "learned_model"))[:40],
    }
    if raw.get("raw_extraction"):
        result["raw_extraction"] = normalize_extraction(raw["raw_extraction"])
    return result


class Worker:
    def __init__(self, store: Store, models_dir: Path, processor=None):
        self.store, self.models_dir, self.processor = store, models_dir, processor
        self.predictor = None
        self.backend = os.environ.get("RECEIPTLAB_MODEL_BACKEND", "classic")
        self.model_ready = processor is not None
        self.stop_event = asyncio.Event()
        self.task, self.loop = None, None
        self.wake_event = asyncio.Event()

    def _initialize(self):
        if self.processor is not None:
            return
        if self.backend == "classic":
            from .ml import get_predictor

            self.predictor = get_predictor(self.models_dir)
        elif self.backend == "transformer":
            from .transformer import TransformerPredictor

            self.predictor = TransformerPredictor(self.models_dir / "layoutlmv3")
        else:
            raise ValueError("Unknown model backend")
        self.model_ready = True

    def _process(self, document: dict):
        try:
            with Image.open(self.store.images / f"{document['id']}.png") as opened:
                image = opened.convert("RGB")
                if self.processor:
                    raw = self.processor(image)
                else:
                    if self.predictor is None:
                        self._initialize()
                    from .assisted import assisted_extract
                    from .ocr import run_ocr

                    predictor = (
                        self.predictor.bind(image)
                        if self.backend == "transformer"
                        else self.predictor
                    )
                    raw = assisted_extract(run_ocr(image), predictor)
            self.store.finish(document["id"], normalize_extraction(raw))
        except Exception:
            self.store.finish(
                document["id"],
                None,
                "Processing failed. Check local model installation and receipt readability.",
            )

    async def run(self):
        try:
            await asyncio.to_thread(self._initialize)
        except Exception:
            self.model_ready = False
        while not self.stop_event.is_set():
            document = await asyncio.to_thread(self.store.claim)
            if document:
                await asyncio.to_thread(self._process, document)
                continue
            self.wake_event.clear()
            try:
                await asyncio.wait_for(self.wake_event.wait(), timeout=0.3)
            except asyncio.TimeoutError:
                pass

    async def start(self):
        self.loop = asyncio.get_running_loop()
        self.store.recover()
        self.task = asyncio.create_task(self.run())

    async def stop(self):
        self.stop_event.set()
        self.wake_event.set()
        if self.task:
            await self.task

    def wake(self):
        if self.loop:
            self.loop.call_soon_threadsafe(self.wake_event.set)
        else:
            self.wake_event.set()
