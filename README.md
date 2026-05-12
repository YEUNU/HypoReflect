# HypoReflect: Reflective GraphRAG for Multi-Hop Financial QA

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- [![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX) -->

Official implementation of **HypoReflect**, a reflective agentic GraphRAG pipeline for multi-hop financial question answering over annual reports, evaluated on [FinanceBench](https://github.com/patronus-ai/financebench).

---

## Overview

Financial QA over multi-document corpora requires synthesizing dispersed numerical claims, cross-referenced filings, and temporally anchored figures. HypoReflect addresses this with a two-phase design:

**Offline Indexing (§3.1)**
- §3.1.1 Topology-Preserving OCR
- §3.1.2 Adaptive Context-Aware Chunking with rolling summary
- §3.1.3 Predictive Knowledge Mapping — hypothetical question Q⁻ and answer Q⁺ per chunk
- §3.1.4 Rank-Based HOP Edge Pre-Construction (company-anchored, offline)

**Query-Time Pipeline (§3.2)**
```
Perception → Planning → Execution → Reflection → Refinement
```

Perception, Planning, and Execution run on a local 4B-class LLM (Qwen3-4B). Reflection and Refinement use a stronger model to enforce verbatim-claim verification and arithmetic recomputation before the final answer is committed.

---

## Main Results

FinanceBench full corpus (150 queries). Metrics follow the official 3-way taxonomy — **Correct Answer / Incorrect Answer / Refusal** — with abstain detection unified across all strategies via `utils/abstain.py`.

| Strategy | Agentic | Judge ↑ | Halluc ↓ | Attempt | DocMatch | PageMatch | Latency |
|---|---|---|---|---|---|---|---|
| Naive RAG | ✗ | 0.040 | 0.080 | 1.000 | 0.900 | 0.000 | 2.2s |
| HopRAG | ✗ | 0.353 | 0.160 | 1.000 | 0.967 | 0.153 | 12.8s |
| MS GraphRAG | ✗ | TBD | TBD | TBD | TBD | TBD | TBD |
| HypoReflect | ✗ | 0.260 | 0.187 | 0.453 | 0.993 | 0.173 | 25.9s |
| **HypoReflect** | **✓** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

> **Judge**: LLM-as-judge correctness (0–1). **Halluc**: hallucination flag rate. **Attempt**: answer-attempted rate. HopRAG natural-language refusals (`I do not know`, etc.) are counted as Refusal, not Incorrect Answer.

---

## Requirements

- Python ≥ 3.12
- 2× NVIDIA GPUs (single GPU works with one generation server)
- Neo4j 5.x
- OpenAI API key for Reflection, Refinement, and the LLM-as-judge (optional — set `REFLECTION_MODEL=` / `REFINEMENT_MODEL=` for full-local ablation)

---

## Installation

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Or with `uv`:
```bash
uv sync
```

Set environment variables (`.env` or export):
```bash
OPENAI_API_KEY=...
EVAL_MODEL=<judge-model>
REFLECTION_MODEL=<reflection-model>
REFINEMENT_MODEL=<refinement-model>
```

---

## Quick Start

### 1. Prepare FinanceBench

```bash
python data/prepare_financebench.py          # downloads PDFs + writes query metadata
python data/prepare_financebench.py --build-corpus   # also extracts raw text (skip OCR)
```

### 2. Start services

```bash
./run_servers.sh all   # Neo4j + vLLM (gen, gen2, ocr, embed, rerank)
python3 scripts/probe_ports.py   # verify all ports are up
```

### 3. OCR + Index

```bash
./run_ocr.sh --sample
./run_index.sh --model hyporeflect --sample --ocr --corpus-tag sample_ocr
```

### 4. Benchmark

```bash
./run_benchmark.sh --model hyporeflect --sample --corpus-tag sample_ocr
```

Results appear under `data/results/<timestamp>/`.

---

## Reproducing Paper Experiments

Two driver scripts run the full paper matrix end-to-end.

```bash
# Step 1 — build all 7 required indexes
./run_all_indexing_parallel.sh --full

# Step 2 — run all 12 benchmarks
./run_all_benchmark_parallel.sh --full
```

**Indexes produced (7 total):**

| # | corpus_tag | Strategy | Purpose |
|---|---|---|---|
| 1 | `naive_full` | naive | baseline |
| 2 | `hoprag_full` | hoprag | baseline |
| 3 | `ms_graphrag_full` | ms_graphrag | baseline |
| 4 | `hyporeflect_full` | hyporeflect | proposed |
| 5 | `hyporeflect_no_table` | hyporeflect | ablation: no table→text |
| 6 | `hyporeflect_no_chunk` | hyporeflect | ablation: no adaptive chunking |
| 7 | `hyporeflect_no_summary` | hyporeflect | ablation: no rolling summary |

**Benchmarks run (12 total):**

| # | Strategy | Corpus | Agentic | Notes |
|---|---|---|---|---|
| 1 | naive | `naive_full` | ✗ | baseline |
| 2 | hoprag | `hoprag_full` | ✗ | baseline |
| 3 | ms_graphrag | `ms_graphrag_full` | ✗ | baseline |
| 4 | hyporeflect | `hyporeflect_full` | ✗ | paper §4.4 agentic-OFF reference |
| 5 | hyporeflect | `hyporeflect_full` | ✓ | **proposed main result** |
| 6 | hyporeflect | `hyporeflect_no_table` | ✓ | ablation |
| 7 | hyporeflect | `hyporeflect_no_chunk` | ✓ | ablation |
| 8 | hyporeflect | `hyporeflect_no_summary` | ✓ | ablation |
| 9 | naive | `naive_full` | ✓ | `agentic_core` applied to baseline |
| 10 | hoprag | `hoprag_full` | ✓ | `agentic_core` applied to baseline |
| 11 | ms_graphrag | `ms_graphrag_full` | ✓ | `agentic_core` applied to baseline |
| 12 | hyporeflect | `hyporeflect_full` | ✓ | all-local Reflection/Refinement ablation |

---

## Baselines

Each baseline runs its **own published indexing pipeline** against the same local LLM, isolating pipeline architecture from model choice.

| Strategy | Indexer | Reference |
|---|---|---|
| Naive RAG | Flat chunk + BM25/vector hybrid | `models/naive/` |
| HopRAG | `HopBuilder.QABuilder` | Liu et al., ACL Findings 2025 |
| MS GraphRAG | `graphrag.api.build_index` (Standard) | Edge et al., 2024 |

> Note: `RAG_ABLATION_TABLE/CHUNKING/SUMMARY` are read only by HypoReflect's indexing pipeline. Baselines ignore these flags and run their published code verbatim.

---

## Evaluation Outputs

```
data/results/<timestamp>/<strategy>/<corpus_tag>/[refl_{on,off}/][agentic_{on,off}/]
```

Each result JSON contains averaged metrics, per-query `details` with `financebench_label` ∈ {`Correct Answer`, `Incorrect Answer`, `Refusal`}, category breakdowns, and ablation metadata.

Re-derive the 3-way breakdown on any historical result without re-calling the LLM judge:

```bash
python tools/fairness_audit.py data/results/<timestamp>/.../<result>.json
```

---

## Citation

```bibtex
@article{hyporeflect2026,
  title     = {HypoReflect: Reflective GraphRAG for Multi-Hop Financial Question Answering},
  author    = {},
  year      = {2026}
}
```

---

<details>
<summary><b>Service Configuration</b></summary>

Start / stop services:

```bash
./run_servers.sh all          # start all
./run_servers.sh {neo4j|gen|gen2|ocr|embed|rerank}
./stop_servers.sh all
```

Default ports and GPU placement:

| Service | Port | CUDA | Model |
|---|---|---|---|
| Neo4j HTTP | 7474 | — | — (Bolt on 7687) |
| `gen` | 28000 | 1 | Qwen3-4B-Instruct-2507 |
| `gen2` | 28010 | 0 | Qwen3-4B-Instruct-2507 (optional) |
| `ocr` | 28001 | 1 | lightonai/LightOnOCR-1B-1025 |
| `embed` | 18082 | 0 | Qwen3-Embedding-0.6B (1024-d) |
| `rerank` | 18083 | 1 | Qwen3-Reranker-0.6B |

Set `VLLM_URL_2=http://localhost:28010/v1` to enable round-robin across `gen` and `gen2`.

</details>

<details>
<summary><b>Key Environment Toggles</b></summary>

Retrieval:
```bash
RAG_ENABLE_QUERY_REWRITE=True
RAG_HOP_MODE=offline              # offline (pre-built HOPs) | runtime (HopRAG-style ANN)
RAG_AGENTIC_OFF_GRAPH_DEPTH=1     # 0 = legacy retrieve-only path
RAG_FINANCE_MARKER_BOOST=0.0      # set 0.15 to boost financial statement pages
```

LLM-call budget:
```bash
RAG_MAX_TOOL_CALLS=3              # T_max (bootstrap counts as 1)
RAG_MAX_REFINEMENT=1              # R_max
RAG_PLANNING_MERGE=true           # plan + filter_policy in one LLM call
RAG_AGENT_INLINE_LEDGER=true      # agent emits EVIDENCE: lines inline
RAG_ENABLE_ATOMIZATION=false      # skip context atomize+pack LLM passes
RAG_REFINEMENT_REJUDGE=false      # skip reflection re-judge inside R_max loop
RAG_DETERMINISTIC_SLOT_FILL=false # skip regex operand fallback
```

Benchmark:
```bash
RAG_BENCHMARK_CONCURRENCY=1       # one query at a time for clean per-query traces
RAG_BENCHMARK_SEEDS=0,1,2         # multi-seed runs → seeds_aggregate.json
```

Benchmark gate (optional quality guardrail):
```bash
RAG_BENCHMARK_GATE=True
RAG_GATE_MAX_LATENCY=45
RAG_GATE_MIN_LLM_JUDGE=0.55
RAG_GATE_MIN_DOC_MATCH=0.60
```

MS GraphRAG multi-endpoint:
```bash
RAG_MS_GEN_API_BASES="http://localhost:28000/v1,http://localhost:28010/v1"
RAG_MS_CONCURRENT_REQUESTS=48
```

Per-run ablation overrides (leaves `.env` untouched):
```bash
REFLECTION_MODEL= REFINEMENT_MODEL= python main.py ...   # full-local
RAG_AGENTIC_MODE=off python main.py ...                  # agentic-OFF
```

</details>

<details>
<summary><b>Utility Scripts</b></summary>

```bash
python3 scripts/probe_ports.py        # ping each vLLM endpoint
python3 scripts/check_env.py          # env sanity (paths, keys, model names)
python3 scripts/check_indexes.py      # list Neo4j labels/indexes and parquet artifacts
python3 scripts/test_neo4j_conn.py    # smoke-test the Neo4j driver
pytest -q                             # unit tests
pytest -q -m integration             # live tests (require Neo4j + vLLM up)
```

</details>
