from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence

import numpy as np

from .config import ProjectionConfig
from .features import domain_reliability_score
from .projection import rank_with_projection
from .runtime import ModelRuntime
from .schemas import Instance


def domain_distribution(instances: Sequence[Instance]) -> list[tuple[str, int]]:
    counts = Counter(
        document.source_domain
        for instance in instances
        for document in instance.documents
        if document.source_domain
    )
    return counts.most_common()


def metadata_coverage(instances: Sequence[Instance]) -> dict[str, float | int]:
    documents = [document for instance in instances for document in instance.documents]
    count = len(documents)
    domains = sum(document.source_domain is not None for document in documents)
    dates = sum(document.pub_date is not None for document in documents)
    supports = sum(document.supports_answer is not None for document in documents)
    return {
        "documents": count,
        "source_count": domains,
        "source_fraction": domains / count if count else 0.0,
        "date_count": dates,
        "date_fraction": dates / count if count else 0.0,
        "answer_support_count": supports,
        "answer_support_fraction": supports / count if count else 0.0,
    }


def projection_diagnostics(
    instances: Sequence[Instance],
    runtime: ModelRuntime,
    config: ProjectionConfig,
    *,
    n: int = 100,
) -> dict:
    lambda_values: dict[str, list[float]] = defaultdict(list)
    divergences: list[float] = []
    violations: list[float] = []
    convergence: list[bool] = []
    divergence_by_conflict: dict[bool, list[float]] = defaultdict(list)
    weights_by_tier: dict[str, list[float]] = defaultdict(list)

    for instance in instances[:n]:
        ranking = rank_with_projection(instance, runtime, config)
        result = ranking.projection
        for name, value in zip(result.feature_names, result.lambdas):
            lambda_values[name].append(float(value))
        q = result.distribution
        p = ranking.prior
        divergence = float(np.sum(q * np.log((q + 1e-300) / (p + 1e-300))))
        divergences.append(divergence)
        divergence_by_conflict[instance.has_conflict].append(divergence)
        violations.append(result.max_violation)
        convergence.append(result.converged)
        for unit, weight in zip(ranking.all_units, q):
            scores = [
                domain_reliability_score(instance.documents[index].source_domain)
                for index in unit.doc_indices
            ]
            average = float(np.mean(scores))
            tier = "high" if average >= 0.85 else "medium" if average >= 0.5 else "low"
            weights_by_tier[tier].append(float(weight))

    lambda_summary = {
        name: {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "max": float(np.max(values)),
            "binding_fraction": float(np.mean(np.asarray(values) > 0.01)),
        }
        for name, values in lambda_values.items()
    }
    return {
        "n": len(divergences),
        "converged_fraction": float(np.mean(convergence)) if convergence else 0.0,
        "max_violation": float(np.max(violations)) if violations else 0.0,
        "kl_mean": float(np.mean(divergences)) if divergences else 0.0,
        "kl_median": float(np.median(divergences)) if divergences else 0.0,
        "lambda_summary": lambda_summary,
        "mean_weight_by_domain_tier": {
            tier: float(np.mean(values)) for tier, values in weights_by_tier.items() if values
        },
        "mean_kl_by_conflict": {
            str(flag): float(np.mean(values))
            for flag, values in divergence_by_conflict.items()
            if values
        },
    }

