"""Shared ID-like column assessment for the readiness report.

Replaces ad-hoc ``n_unique / n_rows >= 0.9`` checks with a confidence score that
down-weights continuous floats and datetimes (so sensor-style measurements are
not marked Poor) while still flagging integer / UUID / name-hint identifiers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

# Must clear this uniqueness floor before any ID-like score can rise.
ID_UNIQUE_RATIO_FLOOR = 0.9
# score >= this → treat as ID-like (drop / exclude / profile Poor).
ID_LIKE_THRESHOLD = 0.80
# softer band used only by the per-feature profile (Warning).
ID_POSSIBLE_THRESHOLD = 0.55

_EXACT_UNIQUE_BONUS = 0.10
_INTEGER_LIKE_BONUS = 0.10
_IDENTIFIER_SHAPE_BONUS = 0.10
_NAME_HINT_BONUS = 0.10
_CONTINUOUS_FLOAT_PENALTY = 0.50
_DATETIME_PENALTY = 0.35
_BOOLEAN_PENALTY = 0.15

_ID_NAME_TOKENS = frozenset({
    "id", "uuid", "guid", "pk", "key", "index", "record", "patient",
    "user", "member", "row", "case",
})
_ID_NAME_SUBSTRINGS = (
    "uuid", "guid", "record", "patient", "member",
)

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_HEX_RE = re.compile(r"^[0-9a-f]{16,}$", re.IGNORECASE)
_DIGIT_TOKEN_RE = re.compile(r"^\d{6,}$")

_SHAPE_SAMPLE_SIZE = 50
_SHAPE_MATCH_RATIO = 0.8


@dataclass(frozen=True)
class IdLikeAssessment:
    """Result of :func:`assess_id_likeness` for one column."""

    unique_ratio: float
    n_unique: int
    score: float
    is_id_like: bool
    is_possible_id: bool
    reason: Optional[str]
    signals: dict


def is_integer_like_numeric(series: pd.Series) -> bool:
    """True for integer dtypes or float columns whose non-null values are whole numbers."""
    if pd.api.types.is_integer_dtype(series):
        return True
    if not pd.api.types.is_float_dtype(series):
        return False
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    return bool((non_null % 1 == 0).all())


def _name_hint_bonus(col_name: str) -> float:
    """Return a small bonus when the column name looks like an identifier."""
    lower = str(col_name).lower().strip()
    if not lower:
        return 0.0
    tokens = {t for t in re.split(r"[^a-z0-9]+", lower) if t}
    if tokens & _ID_NAME_TOKENS:
        return _NAME_HINT_BONUS
    if lower.endswith("_id") or lower.startswith("id_"):
        return _NAME_HINT_BONUS
    # CamelCase ...Id / ...ID
    if re.search(r"[a-z]Id$", str(col_name)) or re.search(r"[a-z]ID$", str(col_name)):
        return _NAME_HINT_BONUS
    for hint in _ID_NAME_SUBSTRINGS:
        if hint in lower:
            return _NAME_HINT_BONUS * 0.8
    return 0.0


def _identifier_shaped_string(series: pd.Series) -> bool:
    """True when a large share of sampled non-null values look like UUIDs/hex/digit IDs."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    sample = non_null
    if len(sample) > _SHAPE_SAMPLE_SIZE:
        sample = sample.iloc[:_SHAPE_SAMPLE_SIZE]
    matches = 0
    checked = 0
    for val in sample:
        text = str(val).strip()
        if not text:
            continue
        checked += 1
        if _UUID_RE.match(text) or _HEX_RE.match(text) or _DIGIT_TOKEN_RE.match(text):
            matches += 1
    if checked == 0:
        return False
    return (matches / checked) >= _SHAPE_MATCH_RATIO


def assess_id_likeness(
    series: pd.Series,
    *,
    col_name: str,
    n_rows: Optional[int] = None,
    n_unique: Optional[int] = None,
) -> IdLikeAssessment:
    """Score how ID-like a column is for readiness filtering.

    Columns below :data:`ID_UNIQUE_RATIO_FLOOR` always get ``score=0``. Above the
    floor, bonuses (exact unique, integer-like, identifier-shaped strings, name
    hints) and penalties (continuous float, datetime, boolean) adjust a base
    score derived from the unique ratio.
    """
    rows = int(n_rows) if n_rows is not None else len(series)
    if n_unique is None:
        n_unique_val = int(series.nunique(dropna=True))
    else:
        n_unique_val = int(n_unique)

    if rows <= 0:
        return IdLikeAssessment(
            unique_ratio=0.0,
            n_unique=n_unique_val,
            score=0.0,
            is_id_like=False,
            is_possible_id=False,
            reason=None,
            signals={"n_rows": rows},
        )

    unique_ratio = n_unique_val / rows
    non_null_count = int(series.notna().sum())
    exact_unique = (
        n_unique_val == rows
        and non_null_count == rows
        and rows > 0
    )

    is_bool = bool(pd.api.types.is_bool_dtype(series))
    is_datetime = bool(pd.api.types.is_datetime64_any_dtype(series))
    is_numeric = bool(pd.api.types.is_numeric_dtype(series)) and not is_bool
    integer_like = is_integer_like_numeric(series) if is_numeric else False
    continuous_float = is_numeric and not integer_like
    shaped = False
    if not is_numeric and not is_datetime and not is_bool:
        shaped = _identifier_shaped_string(series)
    name_bonus = _name_hint_bonus(col_name)

    signals: dict[str, Any] = {
        "unique_ratio": unique_ratio,
        "exact_unique": exact_unique,
        "integer_like": integer_like,
        "continuous_float": continuous_float,
        "datetime": is_datetime,
        "boolean": is_bool,
        "identifier_shaped": shaped,
        "name_hint_bonus": name_bonus,
    }

    if unique_ratio < ID_UNIQUE_RATIO_FLOOR:
        return IdLikeAssessment(
            unique_ratio=unique_ratio,
            n_unique=n_unique_val,
            score=0.0,
            is_id_like=False,
            is_possible_id=False,
            reason=None,
            signals=signals,
        )

    score = float(unique_ratio)
    if exact_unique:
        score += _EXACT_UNIQUE_BONUS
    if integer_like:
        score += _INTEGER_LIKE_BONUS
    if shaped:
        score += _IDENTIFIER_SHAPE_BONUS
    score += name_bonus
    if continuous_float:
        score -= _CONTINUOUS_FLOAT_PENALTY
    if is_datetime:
        score -= _DATETIME_PENALTY
    if is_bool:
        score -= _BOOLEAN_PENALTY

    score = max(0.0, min(1.0, score))
    is_id_like = score >= ID_LIKE_THRESHOLD
    is_possible_id = (not is_id_like) and score >= ID_POSSIBLE_THRESHOLD
    reason = "ID-like (near-unique values)" if is_id_like else None

    return IdLikeAssessment(
        unique_ratio=unique_ratio,
        n_unique=n_unique_val,
        score=score,
        is_id_like=is_id_like,
        is_possible_id=is_possible_id,
        reason=reason,
        signals=signals,
    )
