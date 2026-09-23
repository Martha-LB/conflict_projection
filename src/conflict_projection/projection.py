from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass

import numpy as np

from .config import ProjectionConfig
from .features import (
    compute_g2,
    compute_g3,
    compute_g4,
    compute_g6,
    compute_g12,
    compute_h,
    compute_hedge_ratio,
    compute_sentiment_extremity,
)
from .runtime import ModelRuntime
from .schemas import Document, Instance, ProjectionResult, RankedUnit, SourceUnit


@dataclass(frozen=True)
class RankingResult:
    ranked_units: tuple[RankedUnit, ...]
    projection: ProjectionResult
    all_units: tuple[SourceUnit, ...]
    prior: np.ndarray


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def bm25_probabilities(question: str, documents: list[Document], temperature: float = 1.0) -> np.ndarray:
    from rank_bm25 import BM25Okapi

    if not documents:
        raise ValueError("Cannot score an empty document list")
    corpus = [_tokens(document.text) for document in documents]
    scores = np.asarray(BM25Okapi(corpus).get_scores(_tokens(question)), dtype=float)
    if not np.isfinite(scores).all() or scores.max(initial=0.0) <= 0:
        return np.full(len(documents), 1.0 / len(documents))
    logits = scores / temperature
    logits -= logits.max()
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum()


def deduplicate(
    documents: list[Document], embeddings: np.ndarray, *, similarity_threshold: float
) -> list[SourceUnit]:
    units: list[SourceUnit] = []
    centroids: list[np.ndarray] = []
    for index, embedding in enumerate(embeddings):
        best_index = -1
        best_similarity = similarity_threshold
        for candidate, centroid in enumerate(centroids):
            similarity = float(embedding @ centroid)
            if similarity >= best_similarity:
                best_index = candidate
                best_similarity = similarity
        if best_index < 0:
            units.append(SourceUnit(doc_indices=[index], representative=documents[index].text))
            centroids.append(embedding.copy())
            continue
        units[best_index].doc_indices.append(index)
        centroid = embeddings[units[best_index].doc_indices].mean(axis=0)
        norm = np.linalg.norm(centroid)
        centroids[best_index] = centroid / norm if norm else centroid
    return units


def build_unit_embeddings(units: list[SourceUnit], document_embeddings: np.ndarray) -> np.ndarray:
    values: list[np.ndarray] = []
    for unit in units:
        members = document_embeddings[unit.doc_indices]
        unit.member_embeddings = members
        mean = members.mean(axis=0)
        norm = np.linalg.norm(mean)
        values.append(mean / norm if norm else mean)
    return np.stack(values)


def cluster_units(
    units: list[SourceUnit], unit_embeddings: np.ndarray, cluster_count: int | None = None
) -> tuple[list[SourceUnit], int]:
    from sklearn.cluster import KMeans

    count = len(units)
    if count == 1:
        units[0].cluster_id = 0
        return units, 1
    clusters = min(count, cluster_count or max(2, count // 2))
    labels = KMeans(n_clusters=clusters, random_state=42, n_init=10).fit_predict(unit_embeddings)
    for unit, label in zip(units, labels):
        unit.cluster_id = int(label)
    return units, clusters


def cluster_units_from_metadata(
    units: list[SourceUnit], documents: list[Document]
) -> tuple[list[SourceUnit], int]:
    raw_labels: list[int] = []
    for unit in units:
        labels = [documents[index].metadata.get("cluster") for index in unit.doc_indices]
        valid = [int(label) for label in labels if label is not None]
        if not valid:
            raise ValueError("cluster_strategy='metadata' requires a cluster label on every document")
        raw_labels.append(max(set(valid), key=valid.count))
    unique = sorted(set(raw_labels))
    remap = {label: index for index, label in enumerate(unique)}
    for unit, label in zip(units, raw_labels):
        unit.cluster_id = remap[label]
    return units, len(unique)


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(values.max())
    return maximum + math.log(float(np.exp(values - maximum).sum()))


def _violations(distribution: np.ndarray, matrix: np.ndarray, targets: np.ndarray, signs: np.ndarray) -> np.ndarray:
    expectations = matrix.T @ distribution
    return np.maximum(0.0, -signs * (expectations - targets))


def information_projection(
    prior: np.ndarray,
    matrix: np.ndarray,
    targets: np.ndarray,
    signs: np.ndarray,
    *,
    feature_names: tuple[str, ...] | None = None,
    max_iter: int = 200,
    tolerance: float = 1e-8,
) -> ProjectionResult:
    """Minimize KL(q || prior) subject to sign * (E_q[g] - target) >= 0."""
    if matrix.shape[0] != len(prior):
        raise ValueError("Constraint matrix rows must match the prior length")
    if matrix.shape[1] != len(targets) or len(targets) != len(signs):
        raise ValueError("Constraint columns, targets, and signs must have the same length")
    names = feature_names or tuple(f"constraint_{index}" for index in range(matrix.shape[1]))
    if matrix.shape[1] == 0:
        return ProjectionResult(prior.copy(), np.zeros(0), True, 0, 0.0, names)

    prior = np.asarray(prior, dtype=float)
    prior = prior / prior.sum()
    signed_matrix = matrix * signs[None, :]
    signed_targets = signs * targets
    log_prior = np.log(prior + 1e-300)
    lambdas = np.zeros(matrix.shape[1], dtype=float)

    def dual(values: np.ndarray) -> float:
        return _logsumexp(log_prior + signed_matrix @ values) - float(values @ signed_targets)

    converged = False
    iteration = 0
    for iteration in range(1, max_iter + 1):
        logits = log_prior + signed_matrix @ lambdas
        distribution = np.exp(logits - _logsumexp(logits))
        expectation = signed_matrix.T @ distribution
        gradient = expectation - signed_targets
        # KKT projected gradient: positive gradient at lambda=0 is already optimal.
        projected_gradient = np.where(lambdas > 0.0, gradient, np.minimum(gradient, 0.0))
        if float(np.linalg.norm(projected_gradient, ord=np.inf)) < tolerance:
            converged = True
            break

        centered = signed_matrix - expectation[None, :]
        hessian = (distribution[:, None] * centered).T @ centered
        hessian += 1e-8 * np.eye(matrix.shape[1])
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]

        start_value = dual(lambdas)
        step_size = 1.0
        candidate = lambdas
        for _ in range(40):
            candidate = np.maximum(lambdas - step_size * step, 0.0)
            if dual(candidate) <= start_value + 1e-12:
                break
            step_size *= 0.5
        if np.allclose(candidate, lambdas, atol=tolerance, rtol=0.0):
            break
        lambdas = candidate

    logits = log_prior + signed_matrix @ lambdas
    distribution = np.exp(logits - _logsumexp(logits))
    max_violation = float(_violations(distribution, matrix, targets, signs).max(initial=0.0))
    converged = converged and max_violation <= max(1e-6, tolerance * 100)
    return ProjectionResult(distribution, lambdas, converged, iteration, max_violation, names)


def _feature_target(
    values: np.ndarray,
    prior: np.ndarray,
    sign: float,
    config: ProjectionConfig,
    *,
    feature_name: str,
) -> float:
    if config.threshold_strategy == "legacy_mean":
        if feature_name == "g2":
            return float((values.min() + values.mean()) / 2.0)
        return float(values.mean())
    baseline = float(prior @ values)
    if sign > 0:
        return baseline + config.shift_strength * (float(values.max()) - baseline)
    return baseline - config.shift_strength * (baseline - float(values.min()))


def build_constraints(
    units: list[SourceUnit],
    documents: list[Document],
    unit_embeddings: np.ndarray,
    cluster_count: int,
    prior: np.ndarray,
    runtime: ModelRuntime,
    config: ProjectionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    columns: list[np.ndarray] = []
    targets: list[float] = []
    signs: list[float] = []
    names: list[str] = []

    def add_vector(name: str, values: np.ndarray | None, sign: float) -> None:
        if values is None:
            return
        columns.append(values[:, None])
        targets.append(_feature_target(values, prior, sign, config, feature_name=name))
        signs.append(sign)
        names.append(name)

    if "g2" in config.features:
        add_vector("g2", compute_g2(units, runtime, documents), -1.0)
    if "g6" in config.features:
        add_vector("g6", compute_g6(unit_embeddings, units, documents), 1.0)
    if "h" in config.features and cluster_count > 1:
        coverage = compute_h(units, cluster_count)
        columns.append(coverage)
        if config.cluster_floors is not None:
            if len(config.cluster_floors) != cluster_count:
                raise ValueError(
                    f"Expected {cluster_count} cluster floors, got {len(config.cluster_floors)}"
                )
            floors = list(config.cluster_floors)
        else:
            floor = min(config.cluster_floor, 0.5 / cluster_count)
            floors = [floor] * cluster_count
        if sum(floors) > 1.0 + 1e-12:
            raise ValueError("Cluster floor constraints sum to more than 1")
        targets.extend(floors)
        signs.extend([1.0] * cluster_count)
        names.extend([f"h_{index}" for index in range(cluster_count)])
    if "g_sent" in config.features:
        add_vector("g_sent", compute_sentiment_extremity(units), -1.0)
    if "g_hedge" in config.features:
        add_vector("g_hedge", compute_hedge_ratio(units), -1.0)
    if "g4" in config.features:
        add_vector("g4", compute_g4(units, documents), 1.0)
    if "g3" in config.features:
        add_vector("g3", compute_g3(units, documents), 1.0)
    if "g12" in config.features:
        add_vector("g12", compute_g12(units, documents), 1.0)

    if not columns:
        return np.empty((len(units), 0)), np.empty(0), np.empty(0), ()
    return np.hstack(columns), np.asarray(targets), np.asarray(signs), tuple(names)


def rank_with_projection(
    instance: Instance, runtime: ModelRuntime, config: ProjectionConfig
) -> RankingResult:
    documents = list(instance.documents)
    source_indices = list(range(len(documents)))
    if config.prefilter_k is not None and len(documents) > config.prefilter_k:
        scores = bm25_probabilities(instance.question, documents)
        chosen = np.argsort(scores)[::-1][: config.prefilter_k]
        documents = [documents[index] for index in chosen]
        source_indices = [source_indices[index] for index in chosen]

    document_prior = (
        bm25_probabilities(instance.question, documents)
        if config.use_retriever_prior
        else np.full(len(documents), 1.0 / len(documents))
    )
    document_embeddings = runtime.encode([document.text for document in documents])
    units = (
        deduplicate(documents, document_embeddings, similarity_threshold=config.sim_threshold)
        if config.deduplicate_documents
        else [
            SourceUnit(doc_indices=[index], representative=document.text)
            for index, document in enumerate(documents)
        ]
    )
    unit_embeddings = build_unit_embeddings(units, document_embeddings)
    unit_prior = np.asarray(
        [document_prior[unit.doc_indices].sum() for unit in units], dtype=float
    )
    unit_prior /= unit_prior.sum()
    if config.cluster_strategy == "metadata":
        units, cluster_count = cluster_units_from_metadata(units, documents)
    else:
        units, cluster_count = cluster_units(units, unit_embeddings)
    matrix, targets, signs, names = build_constraints(
        units, documents, unit_embeddings, cluster_count, unit_prior, runtime, config
    )
    result = information_projection(
        unit_prior,
        matrix,
        targets,
        signs,
        feature_names=names,
        max_iter=config.max_iter,
        tolerance=config.tolerance,
    )
    if not result.converged:
        warnings.warn(
            f"Projection did not fully converge for {instance.instance_id or instance.question!r}; "
            f"max violation={result.max_violation:.3g}",
            stacklevel=2,
        )

    ordered_indices = np.argsort(result.distribution)[::-1]
    ranked_indices = ordered_indices if config.top_k is None else ordered_indices[: config.top_k]
    ranked: list[RankedUnit] = []
    for unit_index in ranked_indices:
        unit = units[int(unit_index)]
        local_first = unit.doc_indices[0]
        representative = documents[local_first]
        ranked.append(
            RankedUnit(
                text=unit.representative,
                weight=float(result.distribution[unit_index]),
                doc_indices=tuple(source_indices[index] for index in unit.doc_indices),
                source_domain=representative.source_domain,
                pub_date=representative.pub_date,
            )
        )
    mapped_units = tuple(
        SourceUnit(
            doc_indices=[source_indices[index] for index in unit.doc_indices],
            representative=unit.representative,
            cluster_id=unit.cluster_id,
            member_embeddings=unit.member_embeddings,
        )
        for unit in units
    )
    return RankingResult(tuple(ranked), result, mapped_units, unit_prior)


def rank_with_bm25(instance: Instance, top_k: int = 5) -> tuple[RankedUnit, ...]:
    scores = bm25_probabilities(instance.question, instance.documents)
    indices = np.argsort(scores)[::-1][:top_k]
    return tuple(
        RankedUnit(
            text=instance.documents[index].text,
            weight=float(scores[index]),
            doc_indices=(int(index),),
            source_domain=instance.documents[index].source_domain,
            pub_date=instance.documents[index].pub_date,
        )
        for index in indices
    )
