from __future__ import annotations

import re
import string
from collections.abc import Callable, Sequence
from dataclasses import dataclass


Matcher = Callable[[str, Sequence[str]], bool]
Parser = Callable[[str], list[str]]


@dataclass(frozen=True)
class Score:
    exact: bool
    precision: float
    recall: float
    f1: float
    predictions: tuple[str, ...]
    gold_hits: tuple[bool, ...]
    wrong_hits: tuple[bool, ...]


def normalize_text(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9 ]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_qacc(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    value = "".join(character for character in value if character not in string.punctuation)
    return " ".join(value.split())


def present_strict(target: str, predictions: Sequence[str]) -> bool:
    target_tokens = normalize_text(target).split()
    if not target_tokens:
        return False
    size = len(target_tokens)
    for prediction in predictions:
        prediction_tokens = normalize_text(prediction).split()
        if any(
            prediction_tokens[index : index + size] == target_tokens
            for index in range(len(prediction_tokens) - size + 1)
        ):
            return True
    return False


def present_bidirectional(target: str, predictions: Sequence[str]) -> bool:
    normalized_target = normalize_text(target)
    if not normalized_target:
        return False
    for prediction in predictions:
        normalized_prediction = normalize_text(prediction)
        if not normalized_prediction:
            continue
        if normalized_target in normalized_prediction or normalized_prediction in normalized_target:
            return True
        target_tokens = normalized_target.split()
        prediction_tokens = set(normalized_prediction.split())
        if len(target_tokens) >= 4:
            overlap = sum(token in prediction_tokens for token in target_tokens) / len(target_tokens)
            if overlap >= 0.6:
                return True
    return False


def present_qacc(target: str, predictions: Sequence[str]) -> bool:
    normalized_target = normalize_qacc(target)
    return bool(normalized_target) and any(
        normalize_qacc(prediction) == normalized_target for prediction in predictions
    )


def present_verdict(target: str, predictions: Sequence[str]) -> bool:
    normalized_target = normalize_text(target)
    return any(normalize_text(prediction) == normalized_target for prediction in predictions)


def parse_answers(text: str) -> list[str]:
    match = re.search(r"All Correct Answers:\s*\[(.*?)\]", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return []
    inner = match.group(1).strip()
    if not inner:
        return []
    quoted = re.findall(r'"([^"]*)"|\'([^\']*)\'', inner)
    if quoted:
        return [(left or right).strip() for left, right in quoted if (left or right).strip()]
    protected = re.sub(r"(\d),(\d)", r"\1<COMMA>\2", inner)
    return [part.strip().replace("<COMMA>", ",") for part in protected.split(",") if part.strip()]


def parse_verdict(text: str) -> list[str]:
    match = re.search(r"Verdict:\s*(true|false)", text, re.IGNORECASE)
    if match:
        return [match.group(1).lower()]
    return []


def parse_option(text: str) -> list[str]:
    match = re.search(r"Option:\s*([A-D])\b", text, re.IGNORECASE)
    if match:
        return [match.group(1).upper()]
    stripped = text.strip().upper().rstrip(".")
    return [stripped] if stripped in {"A", "B", "C", "D"} else []


def present_option(target: str, predictions: Sequence[str]) -> bool:
    expected = target.strip().upper()
    return expected in {"A", "B", "C", "D"} and any(
        prediction.strip().upper() == expected for prediction in predictions
    )


def _deduplicate(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = normalize_text(value)
        if key and key not in seen:
            seen.add(key)
            result.append(value.strip())
    return result


def score_predictions(
    predictions: Sequence[str],
    gold: Sequence[str],
    wrong: Sequence[str] = (),
    *,
    matcher: Matcher = present_strict,
) -> Score:
    unique_predictions = _deduplicate(predictions)
    gold_values = _deduplicate(gold)
    wrong_values = _deduplicate(wrong)

    gold_hits = tuple(matcher(answer, unique_predictions) for answer in gold_values)
    wrong_hits = tuple(matcher(answer, unique_predictions) for answer in wrong_values)
    prediction_hits = tuple(
        any(matcher(answer, [prediction]) for answer in gold_values)
        for prediction in unique_predictions
    )
    recall = sum(gold_hits) / len(gold_values) if gold_values else 0.0
    precision = sum(prediction_hits) / len(unique_predictions) if unique_predictions else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact = bool(gold_values) and bool(unique_predictions)
    exact = exact and all(gold_hits) and all(prediction_hits) and not any(wrong_hits)
    return Score(
        exact=exact,
        precision=precision,
        recall=recall,
        f1=f1,
        predictions=tuple(unique_predictions),
        gold_hits=gold_hits,
        wrong_hits=wrong_hits,
    )
