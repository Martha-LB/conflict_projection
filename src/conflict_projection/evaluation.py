from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .methods import Method
from .schemas import Instance, RankedUnit
from .scoring import Matcher, Parser, Score, parse_answers, present_strict, score_predictions


Scorer = Callable[[Sequence[str], Sequence[str], Sequence[str]], Score]


@dataclass(frozen=True)
class ItemResult:
    instance_id: str
    question: str
    gold_answers: tuple[str, ...]
    raw_response: str
    score: Score
    selected_units: tuple[RankedUnit, ...]
    method_metadata: dict


@dataclass(frozen=True)
class EvaluationResult:
    method_name: str
    items: tuple[ItemResult, ...]
    elapsed_seconds: float

    @property
    def exact_vector(self) -> np.ndarray:
        return np.asarray([item.score.exact for item in self.items], dtype=float)

    @property
    def summary(self) -> dict[str, float | int]:
        if not self.items:
            return {"n": 0, "exact": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        return {
            "n": len(self.items),
            "exact": float(np.mean([item.score.exact for item in self.items])),
            "precision": float(np.mean([item.score.precision for item in self.items])),
            "recall": float(np.mean([item.score.recall for item in self.items])),
            "f1": float(np.mean([item.score.f1 for item in self.items])),
        }


def subset_result(
    result: EvaluationResult,
    instance_ids: Sequence[str],
    *,
    name: str | None = None,
) -> EvaluationResult:
    """Return a metric-compatible subset while preserving the requested ID order."""
    by_id = {item.instance_id: item for item in result.items}
    missing = [identifier for identifier in instance_ids if identifier not in by_id]
    if missing:
        raise ValueError(f"Result is missing instance IDs: {missing[:5]}")
    items = tuple(by_id[identifier] for identifier in instance_ids)
    return EvaluationResult(name or result.method_name, items, result.elapsed_seconds)


def _json_record(item: ItemResult) -> dict:
    return {
        "instance_id": item.instance_id,
        "question": item.question,
        "gold_answers": list(item.gold_answers),
        "raw_response": item.raw_response,
        "score": asdict(item.score),
        "selected_units": [asdict(unit) for unit in item.selected_units],
        "method_metadata": item.method_metadata,
    }


def run_evaluation(
    instances: Sequence[Instance],
    method: Method,
    method_name: str,
    *,
    n: int | None = None,
    parser: Parser = parse_answers,
    matcher: Matcher = present_strict,
    scorer: Scorer | None = None,
    output_path: Path | None = None,
    progress_every: int = 10,
) -> EvaluationResult:
    selected = list(instances[:n] if n is not None else instances)
    if not selected:
        raise ValueError("No instances selected for evaluation")
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")

    started = time.time()
    items: list[ItemResult] = []
    for index, instance in enumerate(selected, start=1):
        output = method(instance)
        predictions = parser(output.text)
        score = (
            scorer(predictions, instance.gold_answers, instance.wrong_answers)
            if scorer is not None
            else score_predictions(
                predictions, instance.gold_answers, instance.wrong_answers, matcher=matcher
            )
        )
        item = ItemResult(
            instance_id=instance.instance_id or f"position-{index - 1}",
            question=instance.question,
            gold_answers=tuple(instance.gold_answers),
            raw_response=output.text,
            score=score,
            selected_units=output.selected_units,
            method_metadata=output.metadata,
        )
        items.append(item)
        if output_path is not None:
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(_json_record(item), ensure_ascii=False) + "\n")
        if progress_every and index % progress_every == 0:
            elapsed = time.time() - started
            print(f"{method_name}: {index}/{len(selected)} ({elapsed:.0f}s)")
    return EvaluationResult(method_name, tuple(items), time.time() - started)


def bootstrap_interval(
    values: Sequence[float], *, n_boot: int = 5000, seed: int = 42
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        raise ValueError("Cannot bootstrap an empty sequence")
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(n_boot, len(array)), replace=True).mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def paired_delta_interval(
    method_a: EvaluationResult,
    method_b: EvaluationResult,
    *,
    n_boot: int = 5000,
    seed: int = 42,
) -> tuple[float, float, float]:
    a, b = _aligned_vectors(method_a, method_b)
    differences = a - b
    low, high = bootstrap_interval(differences, n_boot=n_boot, seed=seed)
    return float(differences.mean()), low, high


def _aligned_vectors(
    method_a: EvaluationResult, method_b: EvaluationResult
) -> tuple[np.ndarray, np.ndarray]:
    a = {item.instance_id: float(item.score.exact) for item in method_a.items}
    b = {item.instance_id: float(item.score.exact) for item in method_b.items}
    if set(a) != set(b):
        missing_a = sorted(set(b) - set(a))[:5]
        missing_b = sorted(set(a) - set(b))[:5]
        raise ValueError(f"Evaluation IDs differ; missing from A={missing_a}, missing from B={missing_b}")
    identifiers = [item.instance_id for item in method_a.items]
    return np.asarray([a[key] for key in identifiers]), np.asarray([b[key] for key in identifiers])


def exact_mcnemar(method_a: EvaluationResult, method_b: EvaluationResult) -> dict[str, float | int]:
    a, b = _aligned_vectors(method_a, method_b)
    a_only = int(((a == 1) & (b == 0)).sum())
    b_only = int(((a == 0) & (b == 1)).sum())
    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
    else:
        lower = min(a_only, b_only)
        tail = sum(math.comb(discordant, k) for k in range(lower + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {"a_only": a_only, "b_only": b_only, "discordant": discordant, "p_value": p_value}


def result_row(result: EvaluationResult, *, seed: int = 42) -> dict[str, float | int | str]:
    summary = result.summary
    low, high = bootstrap_interval(result.exact_vector, seed=seed)
    return {
        "method": result.method_name,
        **summary,
        "exact_ci_low": low,
        "exact_ci_high": high,
        "elapsed_seconds": result.elapsed_seconds,
    }
