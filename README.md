# HypoReflect

HypoReflect is a GraphRAG research codebase for multi-hop financial QA, evaluated on **FinanceBench**. The repository implements:

- `hyporeflect` — the proposed five-stage reflective/agentic pipeline (Perception → Planning → Execution → Reflection → Refinement)
- `naive`, `hoprag`, `ms_graphrag` — three comparison baselines, each run through its own published indexing/retrieval pipeline routed to the same local LLMs
- A shared `agentic_core` orchestrator that can wrap any baseline with the same Reflection/Refinement loop used by HypoReflect
- OCR, indexing, benchmarking, and report-generation scripts
- Local service orchestration for Neo4j, vLLM generation (×2), embedding, OCR, and reranking

The full architecture and stage boundaries are documented in `CLAUDE.md`.

## Requirements

- Python `>=3.12`
- A local virtualenv at `.venv`
- 2× NVIDIA GPUs for the vLLM-backed services (single-GPU works if you start only one `gen` server)
- Neo4j 5.x available either by:
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

If you use `uv`, this repository also includes `pyproject.toml` and `uv.lock`.

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

`OPENAI_API_KEY` is only required when judge/reflection/refinement calls should hit OpenAI. Leave the per-stage `*_MODEL` vars empty (`REFLECTION_MODEL= REFINEMENT_MODEL=`) for full-local — quality drops noticeably on the 4B model, see `CLAUDE.md`.

## Services

`run_servers.sh` starts each backing service with the correct CUDA device, port, and memory budget.

Start all services:

```bash
./run_servers.sh all
```

Start individual services:

```bash
./run_servers.sh neo4j
./run_servers.sh gen      # Qwen3-4B on CUDA 1, port 28000
./run_servers.sh gen2     # Qwen3-4B on CUDA 0, port 28010 (optional, enables round-robin)
./run_servers.sh ocr      # LightOnOCR on CUDA 1, port 28001
./run_servers.sh embed    # Qwen3-Embedding-0.6B on CUDA 0, port 18082
./run_servers.sh rerank   # Qwen3-Reranker-0.6B on CUDA 1, port 18083
```

Stop services:

```bash
./stop_servers.sh all
```

Default ports / GPU placement:

| Service     | Port  | CUDA device | Notes                                          |
| ----------- | ----- | ----------- | ---------------------------------------------- |
| Neo4j HTTP  | 7474  | (cpu)       | Bolt on 7687                                   |
| `gen`       | 28000 | 1           | Generation model (Qwen3-4B-Instruct-2507)      |
| `gen2`      | 28010 | 0           | Optional second generation instance            |
| `ocr`       | 28001 | 1           | LightOnOCR vLLM server                         |
| `embed`     | 18082 | 0           | Embedding model (Qwen3-Embedding-0.6B, 1024-d) |
| `rerank`    | 18083 | 1           | Reranker (Qwen3-Reranker-0.6B)                 |

When `gen2` is running, set `VLLM_URL_2=http://localhost:28010/v1` in your env so HypoReflect's `VLLMClient` round-robins generation calls across both GPUs. The MS GraphRAG indexer installs a LiteLLM Router across `RAG_MS_GEN_API_BASES` for the same effect.

Quick service checks:

```bash
python3 scripts/probe_ports.py    # pings /v1/models on each vLLM port
python3 scripts/check_env.py      # env sanity (paths, keys, model names)
```

## Data Preparation

Prepare FinanceBench (downloads PDFs and writes query/document metadata):

```bash
python data/prepare_financebench.py
```

Optional pre-built text corpus (skip OCR):

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

```bash
./run_ocr.sh --sample            # one company per sector
./run_ocr.sh --n 1               # first 1 sample company (implies --sample)
./run_ocr.sh                     # full FinanceBench
./run_ocr.sh --pdf-dir /path/to/pdfs --output /path/to/output
```

Notes:

- `--n` automatically enables sample mode.
- The wrapper defaults to `--no_convert_tables`; tables are written through to text without LLM rewriting. Pass `--convert_tables` to opt in.
- Calling `main.py --mode ocr` directly inverts this default (`convert_tables=True`); always go through `run_ocr.sh` if you want the standard behavior.

## Indexing

Wrapper around `main.py --mode index`. Starts required services automatically unless `--skip-server` is passed.

```bash
./run_index.sh --model hyporeflect --sample --ocr --corpus-tag sample_ocr
./run_index.sh --model hyporeflect --n 1 --ocr --corpus-tag sample_ocr_n1
./run_index.sh --model hyporeflect --dataset data/finance_corpus_ocr/text --corpus-tag ocr
```

Supported strategies: `hyporeflect`, `naive`, `hoprag`, `ms_graphrag`, `all`.

Key options:

- `--dataset <path>` — input directory of `.txt`/`.md` files
- `--model <strategy>`
- `--sample` — one company per sector (9 companies)
- `--n <k>` — first `k` sample companies (company count, implies `--sample`)
- `--ocr` — pick the OCR variant of the sample/full corpus automatically
- `--raw-ocr` — force original `data/finance_corpus` (no table-to-text)
- `--corpus-tag <tag>` — isolates indexes per strategy/tag (see below)
- `--clear-graph` — **wipes the entire Neo4j database** (`MATCH (n) DETACH DELETE n`), not just the active corpus tag — use only when you know what you're doing
- `--save-intermediate` — dump debug artifacts
- `--save-to <dir>` — copy the sampled `.txt` files to a directory
- `--skip-server` — skip the service startup probe when services are already running

### Corpus isolation

All Neo4j labels and indexes are derived from `(strategy, corpus_tag)`, so distinct tags never collide. Strategy prefixes are:

| Strategy      | Neo4j label prefix |
| ------------- | ------------------ |
| `hyporeflect` | `HY_`              |
| `naive`       | `NA_`              |
| `hoprag`      | `HO_`              |
| `ms_graphrag` | `MS_`              |

MS GraphRAG writes parquet artifacts (entities, relationships, communities, community reports, text units, documents) under `data/ms_graphrag_output/<corpus_tag>/` and a lancedb vector store next to them; it does not store communities in Neo4j.

### Baseline indexing — official upstream pipelines

Each baseline now runs its **own published indexing pipeline**, not the HypoReflect engine. All three share the same LLM (`generation-model` on vLLM 28000/28010) and embedding model (`embedding-model` on vLLM 18082) so the comparison isolates pipeline architecture, not model choice.

- `naive` — `NaiveRAG` (independent code path under `models/naive/`).
- `hoprag` — `models/hoprag/official_indexer.py` drives `third_party/HopRAG/HopBuilder.QABuilder`. Stores nodes under Neo4j label `HO_<corpus_tag>` and edges as `HO_<corpus_tag>_p2a`. paddlenlp NER is replaced by spaCy `en_core_web_sm` (POS-filtered content words) because paddlenlp forces a numpy 1.26 downgrade and conflicts with graphrag.
- `ms_graphrag` — `models/ms_graphrag/official_indexer.py` runs `graphrag.api.build_index` (Standard pipeline: extract_graph → Leiden communities → community reports → embeddings). Outputs land in `data/ms_graphrag_output/<corpus_tag>/`. The query-time adapter (`ms_adapter.py`) reads those parquet files directly; upstream MS LocalSearch / GlobalSearch consume the snapshot.

### MS GraphRAG runtime knobs

```bash
# Round-robin LLM completion across multiple vLLM gen endpoints. Comma-separated.
# When >1 base is listed, a LiteLLM Router is monkey-patched in on import.
export RAG_MS_GEN_API_BASES="http://localhost:28000/v1,http://localhost:28010/v1"

# Per-workflow asyncio.Semaphore size for extract_graph / summarize_descriptions.
# Default 48 saturates both gen endpoints when both are up. Lower on shared GPUs.
export RAG_MS_CONCURRENT_REQUESTS=48
```

### Important: ablation flags are HypoReflect-only

`RAG_ABLATION_TABLE`, `RAG_ABLATION_CHUNKING`, `RAG_ABLATION_SUMMARY` are read only by HypoReflect's indexing pipeline (`models/hyporeflect/indexing/chunking.py`). The `naive`, `hoprag`, and `ms_graphrag` baselines ignore them and run their published code verbatim. Running a baseline under a `_no_*` corpus tag produces an index that is **identical** to the `_full` variant. Paper Table 1 reports ablations on HypoReflect only.

### Pre-flight cleanup of stale Neo4j labels and parquet trees

```bash
python scripts/cleanup_old_indexings.py            # dry-run
python scripts/cleanup_old_indexings.py --apply    # actually drop
python scripts/cleanup_old_indexings.py --apply --drop-smoke  # also drop *_smoke_*
```

`HY_*` (hyporeflect) and `NA_*` (naive) Neo4j labels are preserved by default; only `HO_*`, `MS_*`, and `data/ms_graphrag_output/<corpus_tag>/` namespaces are candidates for deletion.

## Benchmarking

Single strategy:

```bash
./run_benchmark.sh --model hyporeflect --sample --corpus-tag sample_ocr
```

All four strategies:

```bash
./run_benchmark.sh --all --sample
```

Apply the shared agentic orchestrator (Reflection + Refinement) to a baseline:

```bash
./run_benchmark.sh --model naive --sample --agentic on
```

Custom queries file:

```bash
./run_benchmark.sh --queries data/financebench_queries_sample_tagged.json --model hyporeflect --sample
```

Options:

- `--model <strategy>` / `--all`
- `--sample` / `--n <k>`
- `--corpus-tag <tag>`
- `--queries <file>`
- `--agentic on|off`

Results land under `data/results/<timestamp>/...` and are post-processed by `tools/benchmark_report.py`.

## Reproducing the paper experiments

Two driver scripts run the full paper matrix end-to-end. They handle service startup, corpus selection, and parallel scheduling.

### 1. Build every required index

```bash
./run_all_indexing_parallel.sh            # sample (one company per sector), OCR corpus
./run_all_indexing_parallel.sh --full     # full FinanceBench OCR corpus
./run_all_indexing_parallel.sh --n 1      # first sample company only
./run_all_indexing_parallel.sh --skip-baselines   # only HypoReflect family
```

Produces 7 corpora:

| # | corpus_tag                  | strategy      | purpose                  |
| - | --------------------------- | ------------- | ------------------------ |
| 1 | `naive_full`                | `naive`       | baseline                 |
| 2 | `hoprag_full`               | `hoprag`      | baseline                 |
| 3 | `ms_graphrag_full`          | `ms_graphrag` | baseline (long-running)  |
| 4 | `hyporeflect_full`          | `hyporeflect` | proposed full            |
| 5 | `hyporeflect_no_table`      | `hyporeflect` | ablation: no table→text  |
| 6 | `hyporeflect_no_chunk`      | `hyporeflect` | ablation: no adaptive chunk |
| 7 | `hyporeflect_no_summary`    | `hyporeflect` | ablation: no rolling summary |

All 7 are dispatched in parallel; vLLM's batching absorbs the contention. The dominant cost is MS GraphRAG's `extract_graph` stage on the full ~33k text_unit corpus (~30h on 2× RTX 5000 Ada).

Per-task logs land under `logs/indexing_parallel/<corpus_tag>.log`. Confirm completion with `python scripts/check_indexes.py`.

### 2. Run every benchmark

```bash
./run_all_benchmark_parallel.sh             # sample matrix
./run_all_benchmark_parallel.sh --full      # full FinanceBench
./run_all_benchmark_parallel.sh --n 1
./run_all_benchmark_parallel.sh --no-agentic-on   # skip the four agentic-on baseline rows
```

Runs up to 12 benchmarks in parallel against the indexes from step 1:

| # | label                          | strategy      | corpus              | agentic | notes                              |
| - | ------------------------------ | ------------- | ------------------- | ------- | ---------------------------------- |
| 1 | `1_naive`                      | `naive`       | `naive_full`        | off     | baseline                           |
| 2 | `2_hoprag`                     | `hoprag`      | `hoprag_full`       | off     | baseline                           |
| 3 | `3_ms_graphrag`                | `ms_graphrag` | `ms_graphrag_full`  | off     | baseline                           |
| 4 | `4_hyporeflect_off`            | `hyporeflect` | `hyporeflect_full`  | off     | paper §4.4 reference               |
| 5 | `5_hyporeflect_full`           | `hyporeflect` | `hyporeflect_full`  | on      | proposed main result               |
| 6 | `6_hyporeflect_no_table`       | `hyporeflect` | `hyporeflect_no_table` | on   | ablation                           |
| 7 | `7_hyporeflect_no_chunk`       | `hyporeflect` | `hyporeflect_no_chunk` | on   | ablation                           |
| 8 | `8_hyporeflect_no_summary`     | `hyporeflect` | `hyporeflect_no_summary` | on | ablation                           |
| 9 | `9_naive_agentic_on`           | `naive`       | `naive_full`        | on      | agentic_core applied to baseline   |
| 10 | `10_hoprag_agentic_on`        | `hoprag`      | `hoprag_full`       | on      | agentic_core applied to baseline   |
| 11 | `11_ms_graphrag_agentic_on`   | `ms_graphrag` | `ms_graphrag_full`  | on      | agentic_core applied to baseline   |
| 12 | `12_hyporeflect_local`        | `hyporeflect` | `hyporeflect_full`  | on      | all-local Reflection/Refinement    |

All results land under `data/results/<timestamp>/...` and a coverage report is generated automatically via `tools/benchmark_report.py`.

Reflection/Refinement (rows 5–8) call OpenAI by default — set `OPENAI_API_KEY` and ensure `REFLECTION_MODEL` / `REFINEMENT_MODEL` are not empty. Row 12 forces them empty for an all-local ablation.

## Evaluation Outputs

Benchmark outputs are organised as:

```text
data/results/<timestamp>/<strategy>/<corpus_tag>/[refl_{on,off}/][agentic_{on,off}/]...
```

Each result JSON includes:

- averaged metrics (LLM-as-judge score, hallucination flag, doc/page match, latency, answer-attempted rate)
- per-query `details`
- category breakdowns
- ablation metadata (records the env values present at run time)
- optional benchmark gate status

## Key Environment Toggles

Retrieval / indexing:

```bash
export NEO4J_FULLTEXT_ANALYZER=english
export RAG_RECREATE_TEXT_INDEX=False
export RAG_ENABLE_QUERY_REWRITE=True
export RAG_QUERY_REWRITE_COUNT=2
export RAG_QUERY_REWRITE_WEIGHT=0.85
export RAG_META_BOOST_WEIGHT=0.50
export RAG_BOILERPLATE_PENALTY_WEIGHT=0.25
```

Ablations (HypoReflect only — see the note in `Indexing`):

```bash
export RAG_ABLATION_TABLE=True
export RAG_ABLATION_CHUNKING=True
export RAG_ABLATION_SUMMARY=True
export RAG_ENABLE_REFLECTION=True
```

MS GraphRAG runtime knobs:

```bash
export RAG_MS_GEN_API_BASES="http://localhost:28000/v1,http://localhost:28010/v1"
export RAG_MS_CONCURRENT_REQUESTS=48
```

Multi-seed benchmark aggregation:

```bash
export RAG_BENCHMARK_SEEDS=0,1,2   # writes seed_<S>/ subdirs + seeds_aggregate.json
```

Optional benchmark gate (fails the run when metrics drop below thresholds):

```bash
export RAG_BENCHMARK_GATE=True
export RAG_GATE_MAX_LATENCY=45
export RAG_GATE_MIN_LLM_JUDGE=0.55
export RAG_GATE_MIN_DOC_MATCH=0.60
```

## Tests

```bash
pytest -q                     # unit tests
pytest -q -m integration      # live tests (require Neo4j + vLLM up)
pytest -q tests/test_chunking.py::test_name
```

The `integration` marker is declared in `pytest.ini`.

Targeted utilities:

```bash
python scripts/check_indexes.py      # list Neo4j labels/indexes and parquet artifacts
python scripts/test_neo4j_conn.py    # smoke-test the Neo4j driver
python scripts/probe_ports.py        # ping each vLLM endpoint
python scripts/check_env.py          # env sanity (paths, keys, model names)
```

## Main Entry Points

- `main.py` — unified CLI for `ocr`, `index`, `benchmark`, `benchmark_all`
- `run_servers.sh` — service startup (`neo4j | gen | gen2 | ocr | embed | rerank | all`)
- `run_ocr.sh` — OCR wrapper
- `run_index.sh` — single-strategy indexing wrapper
- `run_benchmark.sh` — single-strategy benchmark wrapper
- `run_all_indexing_parallel.sh` — build every paper-required index
- `run_all_benchmark_parallel.sh` — run every paper-required benchmark
