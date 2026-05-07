# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

HypoReflect is a GraphRAG research codebase for multi-hop financial QA, evaluated on **FinanceBench** (the sole benchmark — HotpotQA was previously dropped, do not reintroduce). The reflective/agentic pipeline is compared against three baselines: `naive`, `hoprag`, `ms_graphrag`.

## Pipeline architecture

The HypoReflect strategy implements a five-stage pipeline:

`Perception → Planning → Execution → Reflection → Refinement`

- **Perception, Planning, Execution**: local vLLM (Qwen 4B-class), served at `:28000`.
- **Reflection, Refinement**: `gpt-5.2-2025-12-11` via OpenAI API. The local 4B model degraded judge/refine quality (judge score 0.25 vs 0.34, hallucination 0.36 vs 0.16), so these stages were moved off-vLLM.
- **LLM-as-judge evaluation**: also `gpt-5.2`, configured by `EVAL_MODEL` and `HALLUCINATION_EVAL_MODEL`.

Implication: outages of the local vLLM service do **not** affect Reflection/Refinement, but do affect the upstream stages and indexing.

## Code layout

The `models/hyporeflect/` tree mirrors the paper's two pipelines (offline §3.1, query-time §3.2):

- `main.py` — single CLI entry point with `--mode {index,benchmark,benchmark_all,ocr}`. The shell wrappers (`run_*.sh`) all dispatch through this.
- `cli/{index,benchmark}.py` — per-mode runners invoked by `main.py`. (OCR runner moved to `models/hyporeflect/indexing/ocr.py`.)
- `core/`
  - `neo4j_service.py` — Neo4j driver with `global_close()` lifecycle.
  - `vllm_client.py` — `VLLMClient` and `get_llm_client(model_id)`; also has `global_close()`.
  - `config.py` (`RAGConfig`) — central retrieval/indexing thresholds; many env toggles read here.
  - `schemas.py` — shared pydantic models.
- `models/hyporeflect/`
  - `graphrag.py` — `GraphRAG` facade composing `IndexingPipeline` (paper §3.1) and `RetrievalPipeline` (paper §3.2.3). The vLLM handle on this class is **`self.llm`** (renamed from `self.vllm` during the 2026-03-30 refactor).
  - `indexing/` — paper §3.1 indexing pipeline:
    - `ocr.py` (§3.1.1 Topology-Preserving OCR runner)
    - `chunking.py` (§3.1.2 Adaptive Context-Aware Chunking + rolling context)
    - `knowledge_mapping.py` (§3.1.3 Q-/Q+ generation)
    - `hop_edges.py` (§3.1.4 Rank-Based HOP Edge Pre-Construction)
    - `graph_writer.py` (Neo4j storage + index lifecycle)
  - `retrieval/` — paper §3.2.3 query-time retrieval (used by both Execution and the non-agentic baseline): `text_utils.py`, `quality_gates.py`, `rewrite.py`, `hybrid.py` (RRF), `rerank.py` (τ_r), `traversal.py` (NEXT/HOP, offline + runtime), `retrieve.py` (two-stage Q-/Q+ entry).
  - `stages/` — one file per pipeline stage:
    - `perception.py` (§3.2.1), `planning.py` (§3.2.2), `reflection.py` (§3.2.4), `refinement.py` (§3.2.5 — also owns the R_max=2 lexicographic-guard loop in `RefinementOrchestrator`).
    - `stages/execution/` — paper §3.2.3 Execution as a sub-package: `handler.py` (base + T_max=6 loop), `planning_state.py`, `search.py`, `evidence.py`, `context.py`, `synthesis.py`, `calculator.py`. The full `ExecutionHandler` is composed in `stages/execution/__init__.py`.
    - `common.py`, `llm_json.py` — shared regexes / structured-JSON parser.
  - `orchestrator.py` — sequences the five-stage agentic flow + non-agentic baseline (paper §4.4 agentic-OFF).
  - `service.py` — thin `AgentService` facade exposing `run_workflow`.
  - `state.py`, `trace.py`, `schemas.py` — shared state/trace types.
- `models/agentic_core/` — shared agentic orchestrator (`orchestrator.py`, `full_stage_backend.py`) usable by non-hyporeflect strategies via `--agentic on`.
- `models/{naive,hoprag,ms_graphrag}/` — baseline strategies; each owns its own indexing/retrieval. Selected by `--strategy`/`--model`.
- `utils/prompts/` — externalized prompt templates (e.g. `RERANKER_INSTRUCTION`); always source prompts from here, do not inline.
- `tools/benchmark_report.py` — post-processes `data/results/<timestamp>/...` after a benchmark run.

## Neo4j corpus isolation

`GraphRAG.__init__` derives **all** Neo4j labels and index names from `(strategy, corpus_tag)`, e.g. `HY_sample_ocr_Chunk`, `hyporeflect_sample_ocr_vector_idx`. So:

- Different `--corpus-tag` values **never collide** in the same Neo4j instance — running indexing for `sample_raw` does not invalidate `sample_ocr`.
- The same tag across strategies also does not collide (prefix differs: `HY_`, `NA_`, `HO_`, `MS_`).
- `--clear-graph` runs `MATCH (n) DETACH DELETE n` — it wipes **everything**, not just the active tag. Avoid unless intentional.
- `main.py` auto-resolves `corpus_tag` defaults: `sample_raw`, `sample_ocr`, `raw`, `ocr`, or `default`. Pass `--corpus-tag` explicitly when ambiguous.

## Commands

Service orchestration (Neo4j + vLLM gen/ocr/embed/rerank on ports 7474/7687, 28000, 28001, 18082, 18083):

```bash
./run_servers.sh all          # start all
./run_servers.sh {neo4j|gen|ocr|embed|rerank}
./stop_servers.sh all
python3 probe_ports.py        # quick port check
python3 check_env.py          # env sanity check
```

Indexing / benchmarking (wrappers around `main.py`; auto-start required services unless `--skip-server`):

```bash
./run_index.sh --model hyporeflect --sample --ocr --corpus-tag sample_ocr
./run_benchmark.sh --model hyporeflect --sample --corpus-tag sample_ocr
./run_benchmark.sh --all --sample
./run_benchmark.sh --model naive --sample --agentic on    # apply shared agentic orchestrator to a baseline
```

Direct `main.py` invocation (when bypassing the shell wrappers):

```bash
python main.py --mode index --strategy hyporeflect --sample --ocr --corpus-tag sample_ocr
python main.py --mode benchmark --strategy hyporeflect --queries_file data/financebench_queries.json --corpus-tag sample_ocr
python main.py --mode benchmark_all --sample        # iterates all four strategies
```

Sampling: `--sample` means "one company per sector"; `--n K` further trims to the first K sample companies (it's a **company count**, not a query/file count, and it implies `--sample`).

Tests:

```bash
pytest -q                        # unit tests
pytest -q -m integration         # live tests requiring Neo4j + vLLM up
pytest -q tests/test_chunking.py::test_name   # single test
```

Integration tests are gated by the `integration` marker (defined in `pytest.ini`); they hit live Neo4j/vLLM, so start services first.

## Key environment toggles

Set in `.env` or exported. Most are read in `core/config.py` and the `indexing/` / `retrieval/` mixins.

Retrieval/indexing:
- `NEO4J_FULLTEXT_ANALYZER`, `RAG_RECREATE_TEXT_INDEX`
- `RAG_ENABLE_QUERY_REWRITE`, `RAG_QUERY_REWRITE_COUNT`, `RAG_QUERY_REWRITE_WEIGHT`
- `RAG_META_BOOST_WEIGHT`, `RAG_BOILERPLATE_PENALTY_WEIGHT`

Ablations (each toggles one paper-relevant component):
- `RAG_ABLATION_TABLE`, `RAG_ABLATION_CHUNKING`, `RAG_ABLATION_SUMMARY`, `RAG_ENABLE_REFLECTION`

Benchmark gate (fails the run if metrics fall below thresholds):
- `RAG_BENCHMARK_GATE`, `RAG_GATE_MAX_LATENCY`, `RAG_GATE_MIN_LLM_JUDGE`, `RAG_GATE_MIN_DOC_MATCH`

OpenAI side: `OPENAI_API_KEY`, `EVAL_MODEL`, `HALLUCINATION_EVAL_MODEL`.

## Result layout

```
data/results/<timestamp>/<strategy>/<corpus_tag>/[refl_{on,off}/][agentic_{on,off}/]...
```

Each result JSON contains averaged metrics, per-query `details`, category breakdowns, ablation metadata, and (if enabled) gate status. The parallel scripts `run_all_indexing_parallel.sh` / `run_all_benchmark_parallel.sh` log to `logs/indexing_parallel/` and `logs/benchmark_parallel/`.

## Conventions and gotchas

- The `GraphRAG` instance attribute is **`self.llm`** (not `self.vllm`). When patching in tests, mock `rag.llm.rerank` / `rag.llm.get_embeddings`.
- The full pipeline is reachable via `AgentService.run_workflow` (`models/hyporeflect/service.py`), which delegates to `Orchestrator` (`models/hyporeflect/orchestrator.py`). The R_max=2 refinement loop and lexicographic non-regression guard live in `RefinementOrchestrator` inside `stages/refinement.py` — that's the test patch point (`service._orchestrator.refinement_loop.run_loop`), not `service._run_refinement_loop` (which no longer exists).
- Always close drivers cleanly: `main.py` calls `Neo4jService.global_close()` and `VLLMClient.global_close()` in a `finally` block. New entry points should do the same.
- Ports are fixed (see Services section); changing them requires updating `run_servers.sh`, `probe_ports.py`, and any client URLs.
- Do not introduce new benchmarks; FinanceBench is the only target.
- Domain-specific post-refinement override builders (operating-margin/segment-drag/quick-ratio/debt-securities/capital-intensity/dividend-stability) were removed during the paper-aligned refactor. Don't reintroduce them — they were FinanceBench-specific patches not in the paper.
