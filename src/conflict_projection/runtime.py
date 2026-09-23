from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .config import ModelConfig


@dataclass
class ModelRuntime:
    """Lazily load local embedding and NLI models only when they are needed."""

    config: ModelConfig = field(default_factory=ModelConfig)
    _embedding_model: Any = field(default=None, init=False, repr=False)
    _nli_model: Any = field(default=None, init=False, repr=False)
    _contradiction_index: int | None = field(default=None, init=False, repr=False)

    @property
    def embedding_model(self) -> Any:
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(self.config.embedding_model)
        return self._embedding_model

    @property
    def nli_model(self) -> Any:
        if self._nli_model is None:
            from sentence_transformers import CrossEncoder

            self._nli_model = CrossEncoder(self.config.nli_model)
            self._contradiction_index = self._resolve_contradiction_index(self._nli_model)
        return self._nli_model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self.embedding_model.encode(list(texts), normalize_embeddings=True), dtype=float
        )

    def contradiction_probabilities(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        model = self.nli_model
        probabilities = np.asarray(model.predict(list(pairs), apply_softmax=True), dtype=float)
        if probabilities.ndim != 2:
            raise ValueError(f"Expected NLI probabilities with 2 dimensions, got {probabilities.shape}")
        assert self._contradiction_index is not None
        return probabilities[:, self._contradiction_index]

    def _resolve_contradiction_index(self, model: Any) -> int:
        requested = self.config.contradiction_label
        if isinstance(requested, int):
            return requested

        raw_mapping = getattr(getattr(model, "model", None), "config", None)
        id_to_label = getattr(raw_mapping, "id2label", {}) or {}
        for raw_index, label in id_to_label.items():
            if str(label).lower() == requested.lower():
                return int(raw_index)
        labels = {int(index): str(label) for index, label in id_to_label.items()}
        raise ValueError(
            f"NLI model does not expose label {requested!r}; id2label={labels}. "
            "Set ModelConfig.contradiction_label to the correct label or integer index."
        )

