from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from .schemas import Document, SourceUnit

if TYPE_CHECKING:
    from .runtime import ModelRuntime


TRUSTED_DOMAINS = {
    "apnews.com",
    "bbc.co.uk",
    "bbc.com",
    "britannica.com",
    "nationalgeographic.com",
    "ncbi.nlm.nih.gov",
    "nytimes.com",
    "reuters.com",
    "smithsonianmag.com",
    "theguardian.com",
    "wikipedia.org",
}
UGC_DOMAINS = {
    "facebook.com",
    "instagram.com",
    "medium.com",
    "pinterest.com",
    "quora.com",
    "reddit.com",
    "tiktok.com",
    "tumblr.com",
    "twitter.com",
    "x.com",
    "youtube.com",
}
EDUCATION_DOMAINS = {
    "byjus.com",
    "chegg.com",
    "cliffsnotes.com",
    "khanacademy.org",
    "quizlet.com",
    "sparknotes.com",
    "toppr.com",
}
VERTICAL_DOMAINS = {
    "espn.com",
    "ew.com",
    "genius.com",
    "hollywoodreporter.com",
    "imdb.com",
    "rottentomatoes.com",
    "screenrant.com",
    "secondhandsongs.com",
    "songfacts.com",
}
CMS_HOSTS = ("wordpress.com", "blogspot.com", "wixsite.com", "weebly.com", "squarespace.com")
SUSPECT_TLDS = (".info", ".biz", ".xyz", ".click", ".top")
GOV_EDU_PATTERN = re.compile(r"\.gov$|\.edu$|\.ac\.[a-z]{2}$")

HEDGES = {
    "allegedly",
    "appear",
    "could",
    "indicate",
    "maybe",
    "might",
    "partly",
    "perhaps",
    "possibly",
    "probably",
    "reportedly",
    "seem",
    "somewhat",
    "suggest",
}
BOOSTERS = {
    "absolutely",
    "always",
    "certainly",
    "clearly",
    "definitely",
    "everyone",
    "never",
    "nobody",
    "obviously",
    "proven",
    "undoubtedly",
}


def _domain_matches(domain: str, root: str) -> bool:
    return domain == root or domain.endswith(f".{root}")


def domain_reliability_score(domain: str | None) -> float:
    if not domain:
        return 0.4
    value = domain.lower().rstrip(".")
    if any(_domain_matches(value, root) for root in TRUSTED_DOMAINS):
        return 1.0
    if GOV_EDU_PATTERN.search(value) or "pubmed" in value:
        return 0.9
    if any(_domain_matches(value, root) for root in UGC_DOMAINS):
        return 0.2
    if any(_domain_matches(value, root) for root in CMS_HOSTS):
        return 0.25
    if value.endswith(SUSPECT_TLDS):
        return 0.35
    if any(_domain_matches(value, root) for root in EDUCATION_DOMAINS):
        return 0.6
    if any(_domain_matches(value, root) for root in VERTICAL_DOMAINS):
        return 0.7
    return 0.4


def _precomputed_unit_feature(
    units: list[SourceUnit], documents: list[Document], key: str
) -> np.ndarray | None:
    if not all(key in document.metadata for document in documents):
        return None
    return np.asarray(
        [
            np.mean([float(documents[index].metadata[key]) for index in unit.doc_indices])
            for unit in units
        ],
        dtype=float,
    )


def compute_g2(
    units: list[SourceUnit], runtime: "ModelRuntime", documents: list[Document] | None = None
) -> np.ndarray:
    if documents is not None:
        precomputed = _precomputed_unit_feature(units, documents, "precomputed_g2")
        if precomputed is not None:
            return precomputed
    count = len(units)
    if count == 1:
        return np.zeros(1)
    texts = []
    for unit in units:
        claims = (
            [documents[index].metadata.get("claim") for index in unit.doc_indices]
            if documents is not None
            else []
        )
        texts.append(next((str(claim) for claim in claims if claim), unit.representative))
    pairs = [(texts[i], texts[j]) for i in range(count) for j in range(count) if i != j]
    probabilities = runtime.contradiction_probabilities(pairs)
    contradictions = np.zeros((count, count), dtype=float)
    cursor = 0
    for i in range(count):
        for j in range(count):
            if i != j:
                contradictions[i, j] = probabilities[cursor]
                cursor += 1
    contradictions = 0.5 * (contradictions + contradictions.T)
    return contradictions.sum(axis=1) / (count - 1)


def compute_g6(
    unit_embeddings: np.ndarray,
    units: list[SourceUnit] | None = None,
    documents: list[Document] | None = None,
) -> np.ndarray:
    if units is not None and documents is not None:
        precomputed = _precomputed_unit_feature(units, documents, "precomputed_g6")
        if precomputed is not None:
            return precomputed
    count = len(unit_embeddings)
    if count == 1:
        return np.ones(1)
    similarity = unit_embeddings @ unit_embeddings.T
    np.fill_diagonal(similarity, -np.inf)
    return -similarity.max(axis=1)


def compute_h(units: list[SourceUnit], cluster_count: int) -> np.ndarray:
    matrix = np.zeros((len(units), cluster_count), dtype=float)
    for index, unit in enumerate(units):
        matrix[index, unit.cluster_id] = 1.0
    return matrix


def compute_g4(units: list[SourceUnit], documents: list[Document]) -> np.ndarray | None:
    if not any(doc.source_domain for doc in documents):
        return None
    scores: list[float] = []
    for unit in units:
        domains = [documents[index].source_domain for index in unit.doc_indices]
        values = [domain_reliability_score(domain) for domain in domains]
        scores.append(float(np.mean(values)))
    return np.asarray(scores, dtype=float)


def compute_g3(
    units: list[SourceUnit],
    documents: list[Document],
    *,
    anchor: datetime = datetime(2024, 6, 1),
) -> np.ndarray | None:
    ages: list[float | None] = []
    for unit in units:
        values: list[int] = []
        for index in unit.doc_indices:
            raw = documents[index].pub_date
            if not raw:
                continue
            try:
                parsed = datetime.strptime(raw, "%Y-%m-%d")
            except ValueError:
                continue
            values.append(max(0, (anchor - parsed).days))
        ages.append(float(np.mean(values)) if values else None)
    valid = [age for age in ages if age is not None]
    if not valid:
        return None
    fill = float(np.mean(valid))
    array = np.asarray([age if age is not None else fill for age in ages], dtype=float)
    span = float(array.max() - array.min())
    return np.full(len(array), 0.5) if span < 1e-12 else 1.0 - (array - array.min()) / span


def _normalize_answer(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower().strip())


def compute_g12(units: list[SourceUnit], documents: list[Document]) -> np.ndarray | None:
    """Score support for the global majority answer, not a per-unit majority."""
    all_support = [
        _normalize_answer(doc.supports_answer)
        for doc in documents
        if doc.supports_answer and _normalize_answer(doc.supports_answer)
    ]
    if not all_support:
        return None
    global_majority = Counter(all_support).most_common(1)[0][0]
    unit_scores: list[float | None] = []
    for unit in units:
        support = [
            _normalize_answer(documents[index].supports_answer or "")
            for index in unit.doc_indices
            if documents[index].supports_answer
        ]
        unit_scores.append(
            sum(answer == global_majority for answer in support) / len(support) if support else None
        )
    valid = [score for score in unit_scores if score is not None]
    fill = float(np.mean(valid)) if valid else 0.5
    return np.asarray([score if score is not None else fill for score in unit_scores], dtype=float)


def compute_sentiment_extremity(units: list[SourceUnit]) -> np.ndarray:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    return np.asarray(
        [abs(analyzer.polarity_scores(unit.representative)["compound"]) for unit in units],
        dtype=float,
    )


def compute_hedge_ratio(units: list[SourceUnit]) -> np.ndarray:
    values: list[float] = []
    for unit in units:
        words = re.findall(r"[a-z']+", unit.representative.lower())
        values.append(sum(word in HEDGES or word in BOOSTERS for word in words) / len(words) if words else 0.0)
    return np.asarray(values, dtype=float)
