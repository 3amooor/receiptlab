from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

_NUMBER = re.compile(r"\(?-?\d[\d.,]*\)?")
_FIELDS = ("subtotal", "tax", "discount", "total")


def parse_amount(text):
    tokens = _NUMBER.findall(str(text).strip())
    if not tokens:
        return None
    token = tokens[-1].strip()
    negative = token.startswith("-") or (token.startswith("(") and token.endswith(")"))
    token = token.strip("()-")
    if "." in token and "," in token:
        decimal_separator = "." if token.rfind(".") > token.rfind(",") else ","
        thousand_separator = "," if decimal_separator == "." else "."
        token = token.replace(thousand_separator, "").replace(decimal_separator, ".")
    else:
        separator = "." if "." in token else "," if "," in token else None
        if separator:
            chunks = token.split(separator)
            if len(chunks) > 1 and all(len(chunk) == 3 for chunk in chunks[1:]):
                token = "".join(chunks)
            elif len(chunks) == 2 and len(chunks[-1]) in (1, 2):
                token = ".".join(chunks)
            else:
                return None
    try:
        value = Decimal(token)
    except InvalidOperation:
        return None
    if not value.is_finite() or value > Decimal("1000000000"):
        return None
    if negative:
        value = -value
    return format(value.quantize(Decimal(".01")), "f")


def _confidence(line):
    return min(float(line.get("model_confidence", 0)), float(line.get("confidence", 1)))


def _center(line):
    box = line.get("bbox", [0, 0, 1, 1])
    return (box[1] + box[3]) / 2


def extract(lines, predictor):
    predicted = predictor.predict(lines)
    threshold = float(getattr(predictor, "confidence_threshold", 0.8))
    warnings, fields = [], {}
    for field in _FIELDS:
        candidates = [
            (line, parse_amount(line["text"])) for line in predicted if line["label"] == field
        ]
        candidates = [(line, value) for line, value in candidates if value is not None]
        candidates.sort(key=lambda pair: _confidence(pair[0]), reverse=True)
        if not candidates:
            fields[field] = {"value": None, "confidence": 0.0, "bbox": None}
            continue
        line, value = candidates[0]
        fields[field] = {"value": value, "confidence": _confidence(line), "bbox": line.get("bbox")}
        if len({value for _, value in candidates}) > 1:
            warnings.append(f"Conflicting {field} candidates require review.")
        if _confidence(line) < threshold:
            warnings.append(f"Low-confidence {field} requires review.")
    descriptions = [line for line in predicted if line["label"] == "item_description"]
    numbers = [
        line
        for line in predicted
        if line["label"] in ("item_quantity", "item_unit_price", "item_amount")
    ]
    items, used = [], set()
    for description in descriptions:
        box = description.get("bbox", [0, 0, 1, 1])
        tolerance = max(0.014, (box[3] - box[1]) * 0.8)
        associated = [
            (index, line)
            for index, line in enumerate(numbers)
            if index not in used and abs(_center(line) - _center(description)) <= tolerance
        ]
        item = {
            "description": description["text"],
            "quantity": None,
            "unit_price": None,
            "amount": None,
            "confidence": _confidence(description),
        }
        for role, key in [
            ("item_quantity", "quantity"),
            ("item_unit_price", "unit_price"),
            ("item_amount", "amount"),
        ]:
            candidates = [(index, line) for index, line in associated if line["label"] == role]
            candidates.sort(key=lambda pair: _confidence(pair[1]), reverse=True)
            if candidates:
                index, line = candidates[0]
                value = parse_amount(line["text"])
                if value is not None:
                    if key == "quantity" and Decimal(value) >= 0:
                        value = format(Decimal(value).normalize(), "f")
                    item[key] = value
                    item["confidence"] = min(item["confidence"], _confidence(line))
                    used.add(index)
        items.append(item)
    if not fields["total"]["value"]:
        warnings.append("Total was not extracted; manual review is required.")
    if not items:
        warnings.append("No item descriptions were extracted.")
    if any(item["confidence"] < threshold or item["amount"] is None for item in items):
        warnings.append("Some line items need manual review.")
    total, subtotal = fields["total"]["value"], fields["subtotal"]["value"]
    if total is not None and subtotal is not None:
        expected = (
            Decimal(subtotal)
            + Decimal(fields["tax"]["value"] or "0")
            - abs(Decimal(fields["discount"]["value"] or "0"))
        )
        if abs(expected - Decimal(total)) > Decimal(".02"):
            warnings.append("Subtotal, tax and discount do not reconcile with total.")
    evidence = [entry["confidence"] for entry in fields.values() if entry["value"] is not None]
    evidence.extend(item["confidence"] for item in items)
    confidence = min(evidence) if evidence else 0.0
    return {
        "fields": fields,
        "items": items,
        "lines": predicted,
        "warnings": list(dict.fromkeys(warnings)),
        "model_version": getattr(predictor, "model_version", "unknown"),
        "confidence": confidence,
        "needs_review": bool(warnings) or confidence < threshold,
    }
