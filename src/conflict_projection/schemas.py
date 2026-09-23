from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Document:
    text: str
    doc_type: str = "unknown"
    answer: str = ""
    source_domain: str | None = None
    pub_date: str | None = None
    supports_answer: str | None = None
    source_url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceUnit:
    doc_indices: list[int]
    representative: str
    cluster_id: int = -1
    member_embeddings: np.ndarray | None = None


@dataclass(frozen=True)
class Instance:
    question: str
    documents: list[Document]
    gold_answers: list[str]
    wrong_answers: list[str] = field(default_factory=list)
    has_conflict: bool = False
    instance_id: str | None = None
    split: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RankedUnit:
    text: str
    weight: float
    doc_indices: tuple[int, ...]
    source_domain: str | None = None
    pub_date: str | None = None


@dataclass(frozen=True)
class ProjectionResult:
    distribution: np.ndarray
    lambdas: np.ndarray
    converged: bool
    iterations: int
    max_violation: float
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class MethodOutput:
    text: str
    selected_units: tuple[RankedUnit, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

