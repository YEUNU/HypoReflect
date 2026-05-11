# HypoReflect

HypoReflect is a GraphRAG research codebase for multi-hop QA, evaluated on FinanceBench. The repository includes:

- `hyporeflect`: the main reflective/agentic pipeline
- `naive`, `hoprag`, `ms_graphrag`: baseline and comparison strategies
- OCR, indexing, benchmarking, and report-generation scripts
- Local service orchestration for Neo4j, vLLM generation, embedding, OCR, and reranking

## Requirements

- Python `>=3.12`
- A local virtualenv at `.venv`
- NVIDIA GPU environment for the vLLM-backed services
- Neo4j available either by:
  - `NEO4J_BIN`
  - `$NEO4J_HOME/bin/neo4j`
  - `neo4j` on `PATH`
  - Docker fallback via `neo4j:5-community`

The service scripts assume `.venv/bin/vllm` and `.venv/bin/uvicorn` exist.

## Setup

Create the environment and install dependencies:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

If you use `uv`, this repository also includes [`pyproject.toml`] and `uv.lock`.

Optional environment variables:

```bash
export JAVA_HOME=/path/to/java17
export NEO4J_HOME=/path/to/neo4j
export NEO4J_BIN=/path/to/neo4j
export NEO4J_CONTAINER_NAME=hyporeflect-neo4j

export OPENAI_API_KEY=...
export EVAL_MODEL=gpt-5.5-2026-04-23
export REFLECTION_MODEL=gpt-5.4-mini-2026-03-17
export REFINEMENT_MODEL=gpt-5.4-mini-2026-03-17
```

`OPENAI_API_KEY` is only needed if you want judge/reflection/refinement calls to use OpenAI instead of the local served model. Leave the per-stage `*_MODEL` vars empty for full-local (degrades quality — see CLAUDE.md).

## Services

Start all services:

```bash
./run_servers.sh all
```

Start individual services:

```bash
./run_servers.sh neo4j
./run_servers.sh gen
./run_servers.sh ocr
./run_servers.sh embed
./run_servers.sh rerank
```

Stop services:

```bash
./stop_servers.sh all
```

Default ports:

- Neo4j HTTP: `7474`
- Neo4j Bolt: `7687`
- Generation model: `28000`
- OCR model: `28001`
- Embedding model: `18082`
- Reranker: `18083`

Quick checks:

```bash
python3 scripts/probe_ports.py
python3 scripts/check_env.py
```

## Data Preparation

Prepare FinanceBench:

```bash
python data/prepare_financebench.py
```

Optional corpus build:

```bash
python data/prepare_financebench.py --build-corpus
```

Main generated paths:

- `data/finance_pdfs/`
- `data/financebench_queries.json`
- `data/financebench_document_information.jsonl`
- `data/finance_corpus/` when `--build-corpus` is used

## OCR

The OCR pipeline reads PDFs from `data/finance_pdfs` by default and writes:

- raw page metadata to `data/finance_corpus_ocr/raw`
- extracted text to `data/finance_corpus_ocr/text`

Run OCR on the sample company set:

```bash
./run_ocr.sh --sample
```

Run OCR on the first `n` sample companies:

```bash
./run_ocr.sh --n 1
```

Run OCR on the full dataset:

```bash
./run_ocr.sh
```

Custom input/output:

```bash
./run_ocr.sh --pdf-dir /path/to/pdfs --output /path/to/output
```

Notes:

- `--n` automatically enables sample mode.
- Default behavior is `--no_convert_tables`.
- Use `--convert_tables` only if you explicitly want OCR-stage table-to-text conversion.

## Indexing

Primary entrypoint:

```bash
./run_index.sh
```

Supported strategies:

- `hyporeflect`
- `naive`
- `hoprag`
- `ms_graphrag`
- `all`

Sample raw corpus:

```bash
./run_index.sh --model hyporeflect --sample --corpus-tag sample_raw
```

Sample OCR corpus:

```bash
./run_index.sh --model hyporeflect --sample --ocr --corpus-tag sample_ocr
```

First `n` sample companies:

```bash
./run_index.sh --model hyporeflect --n 1 --ocr --corpus-tag sample_ocr_n1
```

Full dataset with explicit path:

```bash
./run_index.sh --model hyporeflect --dataset data/finance_corpus_ocr/text --corpus-tag ocr
```

Useful options:

- `--dataset <path>`: input directory containing `.txt` or `.md`
- `--model <strategy>`
- `--sample`
- `--n <k>`: sample company count, not file count
- `--ocr`: auto-select sample/full OCR dataset when applicable
- `--raw-ocr`: force original `data/finance_corpus`
- `--corpus-tag <tag>`: isolates corpora inside Neo4j
- `--clear-graph`: delete existing graph content before indexing
- `--save-intermediate`: save debugging artifacts
- `--save-to <dir>`: copy the sampled files to a directory
- `--skip-server`: skip service startup checks when services are already running

By default, indexing starts `neo4j`, `gen`, `embed`, and `rerank` unless `--skip-server` is used.

### Baseline indexing (official upstream, routed to local vLLM)

Each baseline now uses **its own published indexing pipeline**, not hyporeflect's
`GraphRAG` engine. All three share the same LLM (`generation-model` on vLLM
:28000) and embedding model (`embedding-model` on vLLM :18082) so the
comparison isolates pipeline architecture, not model choice.

- `naive` — `NaiveRAG` (independent code path under `models/naive/`).
- `hoprag` — `models/hoprag/official_indexer.py` drives `third_party/HopRAG/HopBuilder.QABuilder`.
  Stores nodes under Neo4j label `HO_<corpus_tag>` and edges as
  `HO_<corpus_tag>_p2a`, with corpus-tagged vector + fulltext indices.
  paddlenlp NER is replaced by spaCy `en_core_web_sm` (POS-filtered content
  words) — paddlenlp would force a numpy 1.26 downgrade and conflict with
  graphrag.
- `ms_graphrag` — `models/ms_graphrag/official_indexer.py` runs `graphrag.api.build_index`
  (extract_graph → Leiden communities → community reports → embeddings).
  Outputs parquet under `data/ms_graphrag_output/<corpus_tag>/` plus
  lancedb vector tables. Query-time adapter (`ms_adapter.py`) reads parquet
  directly; the upstream MS LocalSearch / GlobalSearch consume the snapshot.

Pre-flight cleanup of stale Neo4j labels/indices and parquet trees from
older runs:

```bash
python scripts/cleanup_old_indexings.py            # dry-run (preview)
python scripts/cleanup_old_indexings.py --apply    # actually drop
python scripts/cleanup_old_indexings.py --apply --drop-smoke  # also drop *_smoke_*
```

The script keeps `HY_*` (hyporeflect) and `NA_*` (naive) labels/indices
untouched; only `HO_*`, `MS_*`, `hoprag_*`, `ms_graphrag_*` namespaces are
candidates for deletion.

## Benchmarking

Run one strategy:

```bash
./run_benchmark.sh --model hyporeflect --sample
```

Run one strategy on OCR sample corpus:

```bash
./run_benchmark.sh --model hyporeflect --sample --corpus-tag sample_ocr
```

Run all strategies:

```bash
./run_benchmark.sh --all --sample
```

Enable shared agentic orchestration:

```bash
./run_benchmark.sh --model naive --sample --agentic on
```

Use a custom queries file:

```bash
./run_benchmark.sh --queries data/financebench_queries_sample_tagged.json --model hyporeflect --sample
```

Important options:

- `--model <strategy>`
- `--all`
- `--sample`
- `--n <k>`
- `--corpus-tag <tag>`
- `--queries <file>`
- `--agentic on|off`

Results are written under `data/results/<timestamp>/...` and post-processed by [`tools/benchmark_report.py`]

## Parallel Experiment Scripts

Index all ablation combinations in parallel:

```bash
./run_all_indexing_parallel.sh
```

Benchmark all combinations in parallel:

```bash
./run_all_benchmark_parallel.sh
```

Useful variants:

```bash
./run_all_indexing_parallel.sh --n 1
./run_all_benchmark_parallel.sh --n 1
./run_all_benchmark_parallel.sh --no-agentic
```

Logs are written to:

- `logs/indexing_parallel/`
- `logs/benchmark_parallel/`

## Evaluation Outputs

Benchmark outputs are organized like this:

```text
data/results/<timestamp>/<strategy>/<corpus_tag>/...
```

Depending on the run, the tree may also include:

- `refl_on/` or `refl_off/`
- `agentic_on/` or `agentic_off/`

Each result JSON includes:

- averaged metrics
- per-query `details`
- category breakdowns
- ablation metadata
- optional benchmark gate status

## Key Environment Toggles

Retrieval and indexing:

```bash
export NEO4J_FULLTEXT_ANALYZER=english
export RAG_RECREATE_TEXT_INDEX=False
export RAG_ENABLE_QUERY_REWRITE=True
export RAG_QUERY_REWRITE_COUNT=2
export RAG_QUERY_REWRITE_WEIGHT=0.85
export RAG_META_BOOST_WEIGHT=0.35
export RAG_BOILERPLATE_PENALTY_WEIGHT=0.25
```

Ablations:

```bash
export RAG_ABLATION_TABLE=True
export RAG_ABLATION_CHUNKING=True
export RAG_ABLATION_SUMMARY=True
export RAG_ENABLE_REFLECTION=True
```

Optional benchmark gate:

```bash
export RAG_BENCHMARK_GATE=True
export RAG_GATE_MAX_LATENCY=45
export RAG_GATE_MIN_LLM_JUDGE=0.55
export RAG_GATE_MIN_DOC_MATCH=0.60
```

## Tests

Run unit tests:

```bash
pytest -q
```

Run live integration tests:

```bash
pytest -q -m integration
```

Useful targeted checks:

```bash
python scripts/check_indexes.py
python scripts/test_neo4j_conn.py
```

## Main Entry Points

- [`main.py`]: unified CLI for `ocr`, `index`, `benchmark`, `benchmark_all`
- [`run_servers.sh`]: service startup
- [`run_ocr.sh`]: OCR wrapper
- [`run_index.sh`]: indexing wrapper
- [`run_benchmark.sh`]: benchmark wrapper
