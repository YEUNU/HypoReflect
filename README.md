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
export EVAL_MODEL=gpt-5.2
export HALLUCINATION_EVAL_MODEL=gpt-5.2
```

`OPENAI_API_KEY` is only needed if you want judge/evaluation calls to use OpenAI instead of the local served model.

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
python3 probe_ports.py
python3 check_env.py
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
python check_indexes.py
python test_neo4j_conn.py
```

## Main Entry Points

- [`main.py`]: unified CLI for `ocr`, `index`, `benchmark`, `benchmark_all`
- [`run_servers.sh`]: service startup
- [`run_ocr.sh`]: OCR wrapper
- [`run_index.sh`]: indexing wrapper
- [`run_benchmark.sh`]: benchmark wrapper
