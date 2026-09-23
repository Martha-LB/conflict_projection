from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import urllib.request
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import numpy as np

from .schemas import Document, Instance


RGB_URLS = {
    "en_refine.json": "https://raw.githubusercontent.com/chen700564/RGB/master/data/en_refine.json",
    "en_fact.json": "https://raw.githubusercontent.com/chen700564/RGB/master/data/en_fact.json",
}
QACC_RAW_URL = (
    "https://raw.githubusercontent.com/amazon-science/"
    "qa-with-conflicting-context/{revision}/data/ConflictQA_Dataset.json"
)

_PUBDATE_PATTERNS = (
    re.compile(
        r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+"
        r"\d{1,2},\s+\d{4}",
        re.IGNORECASE,
    ),
    re.compile(r"^\d{4}-\d{2}-\d{2}"),
    re.compile(r"^\d{1,2}/\d{1,2}/\d{4}"),
)


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def _clean_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def extract_domain(url_or_domain: str | None) -> str | None:
    """Return a normalized host without corrupting non-`www` hostnames."""
    value = _clean_string(url_or_domain)
    if not value:
        return None
    candidate = value if "://" in value else f"https://{value}"
    try:
        host = (urlparse(candidate).hostname or "").lower().rstrip(".")
    except ValueError:
        return None
    return host.removeprefix("www.") or None


def extract_publication_date(text: str) -> str | None:
    """Extract a date only when a snippet starts with a date-like prefix."""
    head = text.strip()[:60]
    for pattern in _PUBDATE_PATTERNS:
        match = pattern.search(head)
        if not match:
            continue
        raw = match.group(0)
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%b %d, %Y", "%B %d, %Y"):
            try:
                return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
    return None


def load_ramdocs(*, revision: str | None = None, split: str = "test") -> list[Instance]:
    from datasets import load_dataset

    dataset = load_dataset("HanNight/RAMDocs", revision=revision)
    if split not in dataset:
        raise KeyError(f"RAMDocs has no {split!r} split; available: {list(dataset)}")
    instances: list[Instance] = []
    for position, row in enumerate(dataset[split]):
        documents = [
            Document(
                text=_clean_string(doc.get("text")),
                doc_type=_clean_string(doc.get("type")) or "unknown",
                answer=_clean_string(doc.get("answer")),
            )
            for doc in row.get("documents", [])
            if _clean_string(doc.get("text"))
        ]
        question = _clean_string(row.get("question"))
        if not question or not documents:
            continue
        instances.append(
            Instance(
                question=question,
                documents=documents,
                gold_answers=[_clean_string(x) for x in row.get("gold_answers", []) if _clean_string(x)],
                wrong_answers=[_clean_string(x) for x in row.get("wrong_answers", []) if _clean_string(x)],
                instance_id=_clean_string(row.get("id")) or f"ramdocs-{position}",
                split=split,
            )
        )
    return instances


def load_ambigdocs(*, revision: str | None = None) -> list[Instance]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "yoonsanglee/AmbigDocs",
        filename="test.json",
        repo_type="dataset",
        revision=revision,
    )
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    instances: list[Instance] = []
    for position, row in enumerate(rows):
        docs_field = row.get("documents", [])
        if isinstance(docs_field, list):
            pairs = [(_clean_string(d.get("text")), _clean_string(d.get("answer"))) for d in docs_field]
        else:
            pairs = list(
                zip(
                    (_clean_string(x) for x in docs_field.get("text", [])),
                    (_clean_string(x) for x in docs_field.get("answer", [])),
                )
            )
        pairs = [(text, answer) for text, answer in pairs if text]
        question = _clean_string(row.get("question"))
        if not question or not pairs:
            continue
        gold = list(dict.fromkeys(answer for _, answer in pairs if answer))
        documents = [Document(text=text, doc_type="correct", answer=answer) for text, answer in pairs]
        instances.append(
            Instance(
                question=question,
                documents=documents,
                gold_answers=gold,
                instance_id=_clean_string(row.get("id")) or f"ambigdocs-{position}",
                split="test",
            )
        )
    return instances


def load_conflictbank(
    *,
    revision: str | None = None,
    split: str = "train",
    n: int = 1000,
    seed: int = 42,
    precomputed_features_path: Path | None = None,
) -> list[Instance]:
    """Load ConflictBank's 1-vs-3 multiple-choice setting.

    Each instance contains the default evidence (correct) and three conflicting
    documents (misinformation, temporal, and semantic). When a precomputed
    feature parquet is supplied, its qid order is used and g2/g6 are attached to
    document metadata for exact reuse by the projection pipeline.
    """
    from datasets import load_dataset

    dataset = load_dataset("Warrieryes/CB_qa", split=split, revision=revision)
    precomputed: dict[tuple[str, str], dict[str, float]] = {}
    requested_qids: list[str] | None = None
    if precomputed_features_path is not None:
        import pandas as pd

        frame = pd.read_parquet(precomputed_features_path)
        required = {"qid", "doc_id", "g2", "g6"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"ConflictBank feature file is missing columns: {sorted(missing)}")
        requested_qids = frame["qid"].drop_duplicates().astype(str).tolist()[:n]
        for row in frame.itertuples(index=False):
            precomputed[(str(row.qid), str(row.doc_id))] = {
                "precomputed_g2": float(row.g2),
                "precomputed_g6": float(row.g6),
            }

    if requested_qids is not None:
        sampled = dataset.shuffle(seed=seed).select(range(min(len(requested_qids), len(dataset))))
        sampled_rows = list(sampled)
        sampled_qids = [
            f"{_clean_string(row.get('subject'))}_{_clean_string(row.get('relation'))}"
            for row in sampled_rows
        ]
        if sampled_qids != requested_qids:
            raise ValueError(
                "Precomputed ConflictBank qid order does not match the requested dataset "
                "revision/seed. Regenerate features or use the matching revision."
            )
        selected_rows = sampled_rows
    else:
        count = min(n, len(dataset))
        selected_rows = list(dataset.shuffle(seed=seed).select(range(count)))

    field_map = {
        "default": ("default_claim", ("default_evidence",)),
        "misinformation": (
            "misinformation_conflict_claim",
            ("misinformation_conflict_evidence_evidence", "misinformation_conflict_evidence"),
        ),
        "temporal": ("temporal_conflict_claim", ("temporal_conflict_evidence",)),
        "semantic": ("semantic_conflict_claim", ("semantic_conflict_evidence",)),
    }
    instances: list[Instance] = []
    for row in selected_rows:
        qid = f"{_clean_string(row.get('subject'))}_{_clean_string(row.get('relation'))}"
        options = [_clean_string(option) for option in row.get("options", [])]
        correct_option = _clean_string(row.get("correct_option")).upper()
        if correct_option not in {"A", "B", "C", "D"} or len(options) != 4:
            raise ValueError(f"Invalid ConflictBank options for {qid}")
        correct_index = ord(correct_option) - ord("A")
        documents: list[Document] = []
        for category, (claim_field, evidence_fields) in field_map.items():
            claim = _clean_string(row.get(claim_field))
            evidence = next(
                (_clean_string(row.get(field)) for field in evidence_fields if _clean_string(row.get(field))),
                "",
            )
            if not claim or not evidence:
                raise ValueError(f"ConflictBank {qid} is missing {category} claim/evidence")
            feature_metadata = precomputed.get((qid, category), {})
            documents.append(
                Document(
                    text=evidence,
                    doc_type=category,
                    answer=options[correct_index] if category == "default" else "",
                    metadata={
                        "claim": claim,
                        "cluster": 0 if category == "default" else 1,
                        **feature_metadata,
                    },
                )
            )
        instances.append(
            Instance(
                question=_clean_string(row.get("question")),
                documents=documents,
                gold_answers=[correct_option],
                has_conflict=True,
                instance_id=qid,
                split=split,
                metadata={
                    "options": options,
                    "correct_option": correct_option,
                    "uncertain_option": _clean_string(row.get("uncertain_option")).upper(),
                    "replace_option": _clean_string(row.get("replace_option")).upper(),
                    "setting": "1-vs-3",
                },
            )
        )
    return instances


def download_rgb(data_dir: Path, *, overwrite: bool = False) -> dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for filename, url in RGB_URLS.items():
        destination = data_dir / filename
        if overwrite or not destination.exists():
            urllib.request.urlretrieve(url, destination)
        paths[filename] = destination
    return paths


def download_qacc(
    data_dir: Path,
    *,
    revision: str = "main",
    overwrite: bool = False,
    timeout: float = 120.0,
) -> Path:
    """Download the official QACC JSON and return its local path.

    Existing valid data is reused. For a final experiment, pass a Git commit SHA
    as ``revision`` so the downloaded artifact is reproducible.
    """
    destination = (
        data_dir
        / "qa-with-conflicting-context"
        / "data"
        / "ConflictQA_Dataset.json"
    )
    if destination.exists() and not overwrite:
        _validate_qacc_file(destination)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.download")
    url = QACC_RAW_URL.format(revision=revision)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "conflict-projection/0.1 (dataset downloader)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            temporary.write_bytes(response.read())
        _validate_qacc_file(temporary)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _validate_qacc_file(path: Path) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = _flatten_qacc_rows(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Downloaded QACC file is not valid JSON: {path}") from error
    if not rows or not any(row.get("question") and row.get("contexts") for row in rows):
        raise ValueError(f"Downloaded QACC file has no recognizable QACC records: {path}")


def _read_json_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(parsed, list):
        return parsed
    raise ValueError(f"Expected a JSON array or JSONL file at {path}")


def load_rgb(path: Path, *, negative_type: str = "noise") -> list[Instance]:
    instances: list[Instance] = []
    for position, row in enumerate(_read_json_rows(path)):
        raw_answer = row.get("answer", "")
        if isinstance(raw_answer, str):
            gold = [_clean_string(raw_answer)] if _clean_string(raw_answer) else []
        elif isinstance(raw_answer, list) and raw_answer and isinstance(raw_answer[0], list):
            gold = [_clean_string(a) for group in raw_answer for a in group if _clean_string(a)]
        elif isinstance(raw_answer, list):
            gold = [_clean_string(a) for a in raw_answer if _clean_string(a)]
        else:
            gold = []

        documents: list[Document] = []
        for value in row.get("positive", []):
            text = " ".join(map(str, value)) if isinstance(value, list) else _clean_string(value)
            if text:
                documents.append(Document(text=text, doc_type="correct", answer=gold[0] if gold else ""))
        for value in row.get("negative", []):
            text = " ".join(map(str, value)) if isinstance(value, list) else _clean_string(value)
            if text:
                documents.append(Document(text=text, doc_type=negative_type))

        question = _clean_string(row.get("query") or row.get("question"))
        if question and documents:
            instances.append(
                Instance(
                    question=question,
                    documents=documents,
                    gold_answers=list(dict.fromkeys(gold)),
                    instance_id=_clean_string(row.get("id")) or f"rgb-{path.stem}-{position}",
                    split="test",
                )
            )
    return instances


def _parse_list_field(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not _clean_string(value):
        return []
    parsed = ast.literal_eval(str(value))
    return parsed if isinstance(parsed, list) else []


def load_raguard(
    *,
    revision: str | None = None,
    require_mixed: bool = True,
    min_supporting: int = 1,
    min_misleading: int = 1,
    limit_documents: int | None = 10,
    seed: int | None = 42,
    limit_instances: int | None = None,
) -> list[Instance]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    claims_path = hf_hub_download(
        "UCSC-IRKM/RAGuard", filename="claims.csv", repo_type="dataset", revision=revision
    )
    docs_path = hf_hub_download(
        "UCSC-IRKM/RAGuard", filename="documents.csv", repo_type="dataset", revision=revision
    )
    claims = pd.read_csv(claims_path)
    docs = pd.read_csv(docs_path)
    docs_by_id = {int(row["Document ID"]): row for _, row in docs.iterrows()}

    instances: list[Instance] = []
    for position, row in claims.iterrows():
        try:
            doc_ids = _parse_list_field(row.get("Document IDs"))
            labels = _parse_list_field(row.get("Document Labels"))
        except (ValueError, SyntaxError):
            continue
        if not doc_ids or len(doc_ids) != len(labels):
            continue

        documents: list[Document] = []
        for doc_id, label in zip(doc_ids, labels):
            source = docs_by_id.get(int(doc_id))
            if source is None:
                continue
            title = _clean_string(source.get("Title"))
            body = _clean_string(source.get("Full Text"))
            text = "\n\n".join(part for part in (title, body) if part)
            if text:
                documents.append(
                    Document(text=text, doc_type=_clean_string(label).lower() or "unknown")
                )

        if require_mixed:
            n_supporting = sum(doc.doc_type == "supporting" for doc in documents)
            n_misleading = sum(doc.doc_type == "misleading" for doc in documents)
            if n_supporting < min_supporting or n_misleading < min_misleading:
                continue

        question = _clean_string(row.get("Claim"))
        label = _clean_string(row.get("Label")).lower()
        if question and documents and label in {"true", "false"}:
            if limit_documents and len(documents) > limit_documents:
                from .projection import bm25_probabilities

                scores = bm25_probabilities(question, documents)
                indices = np.argsort(scores)[::-1][:limit_documents]
                documents = [documents[index] for index in indices]
            instances.append(
                Instance(
                    question=question,
                    documents=documents,
                    gold_answers=[label],
                    instance_id=_clean_string(row.get("Claim ID")) or f"raguard-{position}",
                )
            )

    if seed is not None:
        rng = np.random.default_rng(seed)
        rng.shuffle(instances)
    return instances[:limit_instances] if limit_instances is not None else instances


def _flatten_qacc_rows(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        rows: list[dict[str, Any]] = []
        for value in raw.values():
            if isinstance(value, list):
                rows.extend(value)
            elif isinstance(value, dict):
                rows.append(value)
        return rows
    raise ValueError("QACC JSON must contain a list or dictionary of records")


def _is_valid_answer(value: Any) -> bool:
    return bool(_clean_string(value))


def load_qacc(path: Path, *, split: str | None = "test") -> list[Instance]:
    rows = _flatten_qacc_rows(json.loads(path.read_text(encoding="utf-8")))
    has_split_field = any(_clean_string(row.get("split")) for row in rows)
    if split is not None and has_split_field:
        selected = [row for row in rows if _clean_string(row.get("split")).lower() == split.lower()]
        if not selected:
            available = sorted({_clean_string(row.get("split")) for row in rows if row.get("split")})
            raise ValueError(f"QACC split {split!r} not found; available splits: {available}")
        rows = selected
    elif split is not None and not has_split_field:
        warnings.warn(
            "QACC file has no split field; using all rows. Record this explicitly in reports.",
            stacklevel=2,
        )

    instances: list[Instance] = []
    for position, row in enumerate(rows):
        question = _clean_string(row.get("question"))
        contexts = row.get("contexts") or []
        sources = row.get("sources") or []
        documents: list[Document] = []
        for index, context in enumerate(contexts):
            text = _clean_string(context)
            if not text:
                continue
            source_url = _clean_string(sources[index]) if index < len(sources) else ""
            documents.append(
                Document(
                    text=text,
                    source_domain=extract_domain(source_url),
                    source_url=source_url or None,
                    pub_date=extract_publication_date(text),
                    # Annotator answer attribution is deliberately not copied here.
                    supports_answer=None,
                )
            )
        gold = _clean_string(row.get("correctAnswer"))
        if not question or not documents or not gold:
            continue

        second_answer = _is_valid_answer(row.get("secondAnswer"))
        existence_flag = _clean_string(row.get("secondAnswerExist")).lower()
        # In the released QACC file, A means a second answer exists and B means it does not.
        has_conflict = second_answer or existence_flag in {"a", "yes", "true", "1"}
        row_id = _clean_string(row.get("id")) or _stable_id("qacc", f"{position}|{question}")
        instances.append(
            Instance(
                question=question,
                documents=documents,
                gold_answers=[gold],
                has_conflict=has_conflict,
                instance_id=row_id,
                split=_clean_string(row.get("split")) or None,
                metadata={"reason": _clean_string(row.get("reasons"))},
            )
        )
    return instances


def describe_instances(instances: Iterable[Instance]) -> dict[str, int | float]:
    items = list(instances)
    document_count = sum(len(item.documents) for item in items)
    dated = sum(doc.pub_date is not None for item in items for doc in item.documents)
    sourced = sum(doc.source_domain is not None for item in items for doc in item.documents)
    return {
        "instances": len(items),
        "documents": document_count,
        "conflicting_instances": sum(item.has_conflict for item in items),
        "dated_documents": dated,
        "sourced_documents": sourced,
        "date_coverage": dated / document_count if document_count else 0.0,
        "source_coverage": sourced / document_count if document_count else 0.0,
    }
