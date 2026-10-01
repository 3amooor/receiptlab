"""Explicit text anchors assist OCR extraction; they never claim model confidence."""

from __future__ import annotations

import copy
import re
from decimal import Decimal

from receiptlab.extraction import extract, parse_amount

_KEY = re.compile(
    r"^\s*(sub[\s-]*total|grand[\s-]*total|total(?:\s*(?:bayar|payment|pembayaran|harga|belanja))?|jumlah\s+(?:bayar|pembayaran)|tax|pajak|ppn|disc(?:ount)?|diskon|potongan)\b\s*[:=]?\s*(.*)$",
    re.IGNORECASE,
)
_RATE = re.compile(r"^\(?\d+(?:[.,]\d+)?\s*%\)?\s*[:=]?\s*")
_PREFIX = re.compile(r"^(?:rp\.?|idr|usd|\$)\s*", re.IGNORECASE)
_SUFFIX = re.compile(r"\s*(?:idr|usd)$", re.IGNORECASE)
_MONEY = re.compile(r"^\(?-?\d[\d.,]*\)?$")
_META = re.compile(
    r"^(?:date|tanggal|tgl|time|waktu|cashier|kasir|receipt|invoice|order|transaction|tel|phone|fax|email|address|alamat|jalan|jl\.)\b",
    re.IGNORECASE,
)
_DATE = re.compile(r"^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}(?:\s.*)?$")
_TIME = re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?(?:\s*[ap]m)?$", re.IGNORECASE)
_PAYMENT = re.compile(
    r"^(?:cash(?:\s+payment)?|change|kembalian|tunai|credit(?:\s+card)?|debit(?:\s+card)?|paid|payment)\s*[:=]?\s*$",
    re.IGNORECASE,
)
_FIELDS = ("subtotal", "tax", "discount", "total")


def _money(text):
    value = _SUFFIX.sub("", _PREFIX.sub("", str(text).strip()))
    value = re.sub(r"\s*([.,])\s*", r"\1", value)
    if " " in value:
        if not re.fullmatch(r"\(?-?\d{1,3}(?:\s+\d{3})+(?:[.,]\d{1,2})?\)?", value):
            return None
        value = re.sub(r"\s+", "", value)
    return parse_amount(value) if _MONEY.fullmatch(value) else None


def _anchor(text):
    match = _KEY.fullmatch(str(text).strip())
    if not match:
        return None
    key, tail = match.groups()
    key = re.sub(r"[\s-]", "", key.casefold())
    if key.startswith("sub"):
        field = "subtotal"
    elif key in ("tax", "pajak", "ppn"):
        field = "tax"
    elif key in ("disc", "discount", "diskon", "potongan"):
        field = "discount"
    else:
        field = "total"
    if field in ("tax", "discount"):
        tail = _RATE.sub("", tail.strip())
    if not tail.strip():
        return field, None
    value = _money(tail)
    return (field, value) if value is not None else None


def _center(line):
    box = line["bbox"]
    return (box[1] + box[3]) / 2


def _same_row(left, right):
    lb, rb = left["bbox"], right["bbox"]
    tolerance = max(0.008, 0.55 * max(lb[3] - lb[1], rb[3] - rb[1]))
    return abs(_center(left) - _center(right)) <= tolerance


def _metadata(text):
    value = str(text).strip()
    return bool(
        _META.search(value)
        or _DATE.fullmatch(value)
        or _TIME.fullmatch(value)
        or _PAYMENT.fullmatch(value)
    )


def assisted_extract(lines, predictor):
    raw = extract(lines, predictor)
    result = copy.deepcopy(raw)
    result["raw_extraction"] = copy.deepcopy(raw)
    result["extraction_mode"] = "assisted_text_anchor"
    candidates = {field: [] for field in _FIELDS}
    unresolved, anchor_indices = set(), set()
    for index, line in enumerate(result["lines"]):
        anchored = _anchor(line["text"])
        if anchored is None:
            continue
        field, inline = anchored
        anchor_indices.add(index)
        if inline is not None:
            candidates[field].append((inline, list(line["bbox"])))
            continue
        neighbors = []
        for other_index, neighbor in enumerate(result["lines"]):
            if index == other_index or not _same_row(line, neighbor):
                continue
            if neighbor["bbox"][0] < line["bbox"][2] - 0.01:
                continue
            value = _money(neighbor["text"])
            if value is not None:
                neighbors.append((value, list(neighbor["bbox"])))
        if neighbors:
            candidates[field].extend(neighbors)
        else:
            unresolved.add(field)
    warnings = []
    for field in _FIELDS:
        result["fields"][field]["origin"] = "learned_model"
        values = {value for value, _ in candidates[field]}
        previous = raw["fields"][field]["value"]
        if len(values) > 1:
            result["fields"][field] = {
                "value": None,
                "confidence": 0.0,
                "bbox": None,
                "origin": "text_anchor",
            }
            warnings.append(
                f"Conflicting text-anchor {field} amounts were withheld; verify manually."
            )
        elif len(values) == 1:
            value, box = candidates[field][0]
            if previous is not None and Decimal(previous) != Decimal(value):
                result["fields"][field] = {
                    "value": None,
                    "confidence": 0.0,
                    "bbox": None,
                    "origin": "text_anchor",
                }
                warnings.append(
                    f"Learned and text-anchor {field} disagree; amount withheld for manual review."
                )
            else:
                result["fields"][field] = {
                    "value": value,
                    "confidence": 0.0,
                    "bbox": box,
                    "origin": "text_anchor",
                }
                warnings.append(
                    f"{field.capitalize()} recovered from explicit text; inspect its highlighted source."
                )
        elif field in unresolved:
            warnings.append(f"Explicit {field} key has no unambiguous monetary neighbor.")
    anchor_texts = {str(result["lines"][index]["text"]).strip() for index in anchor_indices}
    retained = []
    for item in raw["items"]:
        description = str(item["description"]).strip()
        if item.get("amount") is None or description in anchor_texts or _metadata(description):
            continue
        descriptions = [
            line
            for line in raw["lines"]
            if line["label"] == "item_description" and str(line["text"]).strip() == description
        ]
        matched = any(
            _same_row(d, a)
            and _money(a["text"]) is not None
            and Decimal(_money(a["text"])) == Decimal(item["amount"])
            for d in descriptions
            for a in raw["lines"]
            if a["label"] == "item_amount"
        )
        if matched:
            retained.append(copy.deepcopy(item))
    result["items"] = retained
    result["filtered_item_count"] = len(raw["items"]) - len(retained)
    if result["filtered_item_count"]:
        warnings.append(
            "Unsupported descriptions or receipt metadata were removed; verify retained items."
        )
    warnings.append(
        "Assisted extraction requires human review. Text anchors carry no learned confidence."
    )
    result["warnings"] = list(dict.fromkeys(warnings))
    result["needs_review"] = True
    result["confidence"] = 0.0
    return result
