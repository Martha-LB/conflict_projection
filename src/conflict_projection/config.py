from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


ThresholdStrategy = Literal["prior_shift", "legacy_mean"]
ClusterStrategy = Literal["kmeans", "metadata"]


@dataclass(frozen=True)
class ModelConfig:
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    contradiction_label: str | int = "contradiction"
    temperature: float = 0.0
    max_tokens: int = 400
    seed: int | None = 42


@dataclass(frozen=True)
class ProjectionConfig:
    features: tuple[str, ...] = ("g2", "h", "g6")
    top_k: int | None = 5
    prefilter_k: int | None = None
    use_retriever_prior: bool = False
    sim_threshold: float = 0.88
    deduplicate_documents: bool = True
    cluster_strategy: ClusterStrategy = "kmeans"
    cluster_floor: float = 0.05
    cluster_floors: tuple[float, ...] | None = None
    threshold_strategy: ThresholdStrategy = "prior_shift"
    shift_strength: float = 0.15
    max_iter: int = 200
    tolerance: float = 1e-8

    def __post_init__(self) -> None:
        known = {"g2", "g3", "g4", "g6", "g12", "h", "g_sent", "g_hedge"}
        unknown = set(self.features) - known
        if unknown:
            raise ValueError(f"Unknown projection features: {sorted(unknown)}")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if (
            self.prefilter_k is not None
            and self.top_k is not None
            and self.prefilter_k < self.top_k
        ):
            raise ValueError("prefilter_k must be greater than or equal to top_k")
        if self.cluster_floors is not None and any(value < 0 for value in self.cluster_floors):
            raise ValueError("cluster_floors must contain non-negative values")
        if not 0.0 <= self.shift_strength <= 1.0:
            raise ValueError("shift_strength must be in [0, 1]")


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def cache_db(self) -> Path:
        return self.outputs / "llm_cache.sqlite3"

    @classmethod
    def discover(cls, start: Path | None = None) -> "ProjectPaths":
        current = (start or Path.cwd()).resolve()
        for candidate in (current, *current.parents):
            if (candidate / "pyproject.toml").exists():
                return cls(candidate)
        raise FileNotFoundError("Could not find pyproject.toml from the current directory")
