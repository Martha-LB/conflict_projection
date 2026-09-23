# Conflict Projection


## Layout

```text
src/conflict_projection/
  config.py       experiment and model configuration
  schemas.py      shared data structures
  datasets.py     RAMDocs, ConflictBank, AmbigDocs, RGB, RAGuard, and QACC loaders
  runtime.py      lazy embedding and NLI model loading
  llm.py          OpenAI client, retries, and SQLite cache
  prompts.py      prompt templates
  features.py     g2, g3, g4, g6, g12, sentiment, and hedging
  projection.py   deduplication, constraints, and information projection
  methods.py      concat, BM25, projected retrieval, and MADAM-RAG
  scoring.py      parsers and metrics
  evaluation.py   experiment runner, persistence, confidence intervals
  diagnostics.py  constraint and projection diagnostics
notebooks/
  experiments.ipynb
tests/
```

## Installation

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
```

Set the API key in the shell. The project never overwrites it:

```bash
export OPENAI_API_KEY='...'
```

Then open `notebooks/experiments.ipynb`. Run a small smoke test before increasing
the sample size, because MADAM-RAG can make many model calls per item.

## Data

`load_ramdocs`, `load_ambigdocs`, and `load_raguard` use Hugging Face.
`load_rgb` downloads the official JSONL files into `data/rgb` when requested.

`load_conflictbank` loads `Warrieryes/CB_qa` and constructs the 1-vs-3
multiple-choice setting. If `../inter/conflictbank_features.parquet` is present,
the paper notebook reuses its qid order and precomputed g2/g6 values. The default
notebook protocol clusters without answer labels. `legacy_label_cluster` exists
only to audit earlier experiments that grouped the known-correct `default`
document separately; it must not be described as an unlabeled method.

For QACC, the notebook calls `download_qacc(paths.data)`. It downloads the
official JSON on the first run and reuses the validated local file afterward:

```text
data/qa-with-conflicting-context/data/ConflictQA_Dataset.json
```

For a final reported experiment, pass a Git commit SHA rather than `main`:

```python
qacc_path = download_qacc(paths.data, revision="<official-repository-commit-sha>")
```

Pass `split="test"` for reported results. If the source file has no split field,
the loader states that it is using all rows rather than silently pretending that
they are test data.

Important cohort audit: the official QACC test split has 813 rows, with 207
conflict and 606 non-conflict items. The 166/647 counts in the earlier draft come
from the first 813 rows of the complete file, which mixes train, dev, and test.
The notebook defaults to `QACC_COHORT="official_test"` and retains
`legacy_first_813` only for transparent reproduction of the old table.

## Fair comparisons

There are three separate intervention choices:

1. which documents are selected;
2. whether numerical projection weights are shown to the LLM;
3. whether source/date metadata is shown to the LLM.

For a retrieval-only ablation, keep (2) and (3) disabled for both BM25 and the
projected method. Test weight annotation and metadata exposure in separate rows.
Do not describe a comparison as the “MaxEnt effect” when its prompt or visible
metadata also changes.
