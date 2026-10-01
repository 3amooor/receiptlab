"""Strict review inputs and deterministic decimal financial validation."""

import re
from decimal import ROUND_HALF_UP, Decimal
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator

MONEY_FIELDS = ("subtotal", "tax", "discount", "total")
MONEY_PATTERN = re.compile(r"^(?:0|[1-9]\d{0,7})(?:\.\d{1,2})?$")
QUANTITY_PATTERN = re.compile(r"^(?:0|[1-9]\d{0,5})(?:\.\d{1,3})?$")


def normalize_money(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not MONEY_PATTERN.fullmatch(value):
        raise ValueError("Use a nonnegative decimal string with at most two decimal places.")
    return format(Decimal(value).quantize(Decimal("0.01")), "f")


class ReviewFields(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    subtotal: str | None = None
    tax: str | None = None
    discount: str | None = None
    total: str | None = None

    @field_validator(*MONEY_FIELDS)
    @classmethod
    def validate_amount(cls, value: str | None) -> str | None:
        return normalize_money(value)


class ReviewItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    description: Annotated[str, Field(min_length=1, max_length=240)]
    quantity: str | None = None
    unit_price: str | None = None
    amount: str | None = None

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        value = value.strip()
        if not value or any(ord(char) < 32 for char in value):
            raise ValueError("Description must contain printable text.")
        return value

    @field_validator("unit_price", "amount")
    @classmethod
    def validate_amount(cls, value: str | None) -> str | None:
        return normalize_money(value)

    @field_validator("quantity")
    @classmethod
    def validate_quantity(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if (
            not isinstance(value, str)
            or not QUANTITY_PATTERN.fullmatch(value)
            or Decimal(value) <= 0
        ):
            raise ValueError("Quantity must be a positive decimal string.")
        return value


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    revision: Annotated[StrictInt, Field(ge=1)]
    fields: ReviewFields
    items: Annotated[list[ReviewItem], Field(max_length=250)]
    approve: StrictBool = False


def financial_warnings(fields: dict, items: list[dict]) -> list[str]:
    """Recalculate warnings from editable data; model scores cannot bypass these."""
    values = {
        name: Decimal(field["value"])
        for name, field in fields.items()
        if name in MONEY_FIELDS and field.get("value") is not None
    }
    warnings: list[str] = []
    if "total" not in values:
        warnings.append("A receipt total is required before approval.")
    if "subtotal" in values and "total" in values:
        expected = (
            values["subtotal"] + values.get("tax", Decimal(0)) - values.get("discount", Decimal(0))
        )
        if abs(expected - values["total"]) > Decimal("0.02"):
            warnings.append("Subtotal plus tax minus discount does not match total.")
    amounts = [Decimal(item["amount"]) for item in items if item.get("amount") is not None]
    if items and len(amounts) != len(items):
        warnings.append("Every line item needs an amount before approval.")
    if amounts and len(amounts) == len(items) and "subtotal" in values:
        if abs(sum(amounts) - values["subtotal"]) > Decimal("0.02"):
            warnings.append("Line item amounts do not match subtotal.")
    for index, item in enumerate(items, start=1):
        if all(item.get(name) is not None for name in ("quantity", "unit_price", "amount")):
            expected = (Decimal(item["quantity"]) * Decimal(item["unit_price"])).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            if abs(expected - Decimal(item["amount"])) > Decimal("0.02"):
                warnings.append(
                    f"Line item {index}: quantity times unit price does not match amount."
                )
    return warnings
