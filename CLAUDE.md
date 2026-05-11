# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

HypoReflect is a GraphRAG research codebase for multi-hop financial QA, evaluated on **FinanceBench** (the sole benchmark — HotpotQA was previously dropped, do not reintroduce). The reflective/agentic pipeline is compared against three baselines: `naive`, `hoprag`, `ms_graphrag`.

## Pipeline architecture

The HypoReflect strategy implements a five-stage pipeline:

`Perception → Planning → Execution → Reflection → Refinement`

- **Perception, Planning, Execution**: local vLLM (Qwen 4B-class), served at `:28000` (and `:28010` if `VLLM_URL_2` is set — round-robin'd by `VLLMClient`). `run_servers.sh` pins `gen` to CUDA 1 (port 28000) and `gen2` to CUDA 0 (port 28010); `embed` shares CUDA 0, `reranker` shares CUDA 1.
- **Reflection, Refinement**: `gpt-5.4-mini-2026-03-17` via OpenAI API by default (`REFLECTION_MODEL` / `REFINEMENT_MODEL` in `.env`). The local 4B model degrades these stages — empirically judge 0.25 / hallucination 0.36 vs 0.34 / 0.16 with GPT — and the new verbatim-claim and recompute-on-claim reflection rules require careful instruction following that the 4B does not deliver. Set those vars to empty if you intentionally want all-local for an ablation.
- **LLM-as-judge evaluation**: `gpt-5.5-2026-04-23` (`EVAL_MODEL`). The judge produces both correctness score AND hallucination flag in a single combined LLM call (consolidated from two separate calls; `HALLUCINATION_EVAL_MODEL` was removed from `RAGConfig`). The judge also deterministically marks honest "insufficient evidence" abstentions as hallucination=0 even when GT is substantive — the LLM cannot override.

Implication: outages of the local vLLM service do **not** affect Reflection/Refinement/Judge, but do affect the upstream stages and indexing.

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
    - `chunking.py` (§3.1.2 Adaptive Context-Aware Chunking + rolling context). Adaptive threshold is two-sided clamp around `tau_chunk` (`SIMILARITY_THRESHOLD`); the earlier one-sided `min(...)` form forced one-way drift and never used the configured value.
    - `knowledge_mapping.py` (§3.1.3 Q-/Q+ generation)
    - `hop_edges.py` (§3.1.4 Rank-Based HOP Edge Pre-Construction). Same-company filter is enforced at the Cypher query level (FinanceBench queries are company-anchored — earlier v14 had ~18% cross-company HOPs that added retrieval noise).
    - `graph_writer.py` (Neo4j storage + index lifecycle)
  - `retrieval/` — paper §3.2.3 query-time retrieval (used by both Execution and the non-agentic baseline): `text_utils.py`, `quality_gates.py`, `rewrite.py`, `hybrid.py` (RRF), `rerank.py` (τ_r), `traversal.py` (NEXT/HOP, offline + runtime), `retrieve.py` (two-stage Q-/Q+ entry, paper-aligned weights Stage1 Q-=0.7/body=0.3 + Stage2 Q+=0.6/Q-=0.4).
  - `stages/` — one file per pipeline stage:
    - `perception.py` (§3.2.1), `planning.py` (§3.2.2), `reflection.py` (§3.2.4), `refinement.py` (§3.2.5 — also owns the R_max=2 lexicographic-guard loop in `RefinementOrchestrator`).
    - `stages/execution/` — paper §3.2.3 Execution as a sub-package: `handler.py` (base + T_max=6 loop), `planning_state.py`, `search.py`, `evidence.py`, `context.py`, `synthesis.py`, `calculator.py`. The full `ExecutionHandler` is composed in `stages/execution/__init__.py`.
    - `common.py`, `llm_json.py` — shared regexes / structured-JSON parser.
  - `orchestrator.py` — sequences the five-stage agentic flow + non-agentic baseline (paper §4.4 agentic-OFF).
  - `service.py` — thin `AgentService` facade exposing `run_workflow`.
  - `state.py`, `trace.py`, `schemas.py` — shared state/trace types.
- `models/agentic_core/` — shared agentic orchestrator (`orchestrator.py`, `full_stage_backend.py`) usable by non-hyporeflect strategies via `--agentic on`.
- `models/{naive,hoprag,ms_graphrag}/` — baseline strategies; each owns its own indexing/retrieval. Selected by `--strategy`/`--model`.
  - `models/ms_graphrag/official_indexer.py` — wraps `graphrag.api.build_index` (Standard pipeline). When `RAG_MS_GEN_API_BASES` lists more than one URL, it installs a `litellm.Router` (`simple-shuffle`) via monkey-patch of `litellm.acompletion`, with a `contextvars.ContextVar` re-entry guard so the Router's own internal calls don't recurse. Outputs parquet under `data/ms_graphrag_output/<corpus_tag>/` + a lancedb vector store. Concurrency capped by `RAG_MS_CONCURRENT_REQUESTS` (default 48).
  - `models/ms_graphrag/ms_adapter.py` — query-time adapter; reads `community_reports.parquet`, `entities.parquet`, `text_units.parquet` directly. Does **not** require Neo4j Community nodes.
  - `models/hoprag/official_indexer.py` — drives `third_party/HopRAG/HopBuilder.QABuilder`. paddlenlp NER is stubbed and replaced with spaCy `en_core_web_sm`.
- `utils/prompts/` — externalized prompt templates (e.g. `RERANKER_INSTRUCTION`); always source prompts from here, do not inline.
- `tools/benchmark_report.py` — post-processes `data/results/<timestamp>/...` after a benchmark run.

## Neo4j corpus isolation

`GraphRAG.__init__` derives **all** Neo4j labels and index names from `(strategy, corpus_tag)`, e.g. `HY_sample_ocr_Chunk`, `hyporeflect_sample_ocr_vector_idx`. So:

- Different `--corpus-tag` values **never collide** in the same Neo4j instance — running indexing for `sample_raw` does not invalidate `sample_ocr`.
- The same tag across strategies also does not collide (prefix differs: `HY_`, `NA_`, `HO_`, `MS_`).
- `--clear-graph` runs `MATCH (n) DETACH DELETE n` — it wipes **everything**, not just the active tag. Avoid unless intentional.
- `main.py` auto-resolves `corpus_tag` defaults: `sample_raw`, `sample_ocr`, `raw`, `ocr`, or `default`. Pass `--corpus-tag` explicitly when ambiguous.

## Commands

Service orchestration (Neo4j 7474/7687 + vLLM `gen` 28000 / `gen2` 28010 / `ocr` 28001 / `embed` 18082 / `rerank` 18083):

```bash
./run_servers.sh all          # start all (including gen2)
./run_servers.sh {neo4j|gen|gen2|ocr|embed|rerank}
./stop_servers.sh all
python3 scripts/probe_ports.py        # quick port check (covers 28000/28010/28001/18082/18083)
python3 scripts/check_env.py          # env sanity check
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

Full paper experiment (canonical, end-to-end — see `README.md` "Reproducing the paper experiments"):

```bash
./run_all_indexing_parallel.sh --full         # 7 indexes: 3 baselines × full + 4 hyporeflect (full + 3 ablations)
./run_all_benchmark_parallel.sh --full        # up to 12 benches (paper Table 1 + agentic_on variants)
```

The matrices intentionally exclude baseline ablations (`naive_no_*`, `hoprag_no_*`, `ms_graphrag_no_*`) — see "ablation flag scope" below.

Per-run env overrides (preferred for one-off ablations — leaves `.env` untouched):

```bash
# all-local ablation (REFLECTION/REFINEMENT empty)
REFLECTION_MODEL= REFINEMENT_MODEL= python main.py --mode benchmark --strategy hyporeflect --corpus-tag <tag>

# agentic-OFF baseline (paper §4.4)
RAG_AGENTIC_MODE=off python main.py --mode benchmark --strategy hyporeflect --corpus-tag <tag> --agentic off

# multi-seed (writes seed_<S>/ subdirs + seeds_aggregate.json with mean ± std + 95% CI)
RAG_BENCHMARK_SEEDS=0,1,2 python main.py --mode benchmark --strategy hyporeflect --corpus-tag <tag>
```

Sampling: `--sample` means "one company per sector" (9 companies); `--n K` further trims to the first K sample companies (it's a **company count**, not a query/file count, and it implies `--sample`).

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

**Ablation flag scope (important):** `RAG_ABLATION_TABLE/CHUNKING/SUMMARY` are read only by `models/hyporeflect/indexing/chunking.py` (and `naive_rag.py` purely as a cache-namespace key — no behavior change). The `hoprag` and `ms_graphrag` baselines run their published indexing pipelines verbatim and **ignore these flags**; setting them on a baseline run yields an index identical to the `_full` variant. The catchup/parallel scripts only run ablations on HypoReflect — do not reintroduce baseline ablations.

MS GraphRAG runtime knobs (read by `models/ms_graphrag/official_indexer.py`):
- `RAG_MS_GEN_API_BASES` — comma-separated vLLM gen URLs (default `28000+28010`). >1 entry triggers the LiteLLM Router monkey-patch.
- `RAG_MS_CONCURRENT_REQUESTS` — `asyncio.Semaphore` size for `extract_graph` and `summarize_descriptions` (default 48).

Benchmark gate (fails the run if metrics fall below thresholds):
- `RAG_BENCHMARK_GATE`, `RAG_GATE_MAX_LATENCY`, `RAG_GATE_MIN_LLM_JUDGE`, `RAG_GATE_MIN_DOC_MATCH`

OpenAI side: `OPENAI_API_KEY`, `EVAL_MODEL`.

## Result layout

```
data/results/<timestamp>/<strategy>/<corpus_tag>/[refl_{on,off}/][agentic_{on,off}/]...
```

Each result JSON contains averaged metrics, per-query `details`, category breakdowns, ablation metadata, and (if enabled) gate status. The parallel scripts `run_all_indexing_parallel.sh` / `run_all_benchmark_parallel.sh` log to `logs/indexing_parallel/` and `logs/benchmark_parallel/`.

## Conventions and gotchas

- The `GraphRAG` instance attribute is **`self.llm`** (not `self.vllm`). When patching in tests, mock `rag.llm.rerank` / `rag.llm.get_embeddings`.
- The full pipeline is reachable via `AgentService.run_workflow` (`models/hyporeflect/service.py`), which delegates to `Orchestrator` (`models/hyporeflect/orchestrator.py`). The R_max=2 refinement loop and lexicographic non-regression guard live in `RefinementOrchestrator` inside `stages/refinement.py` — that's the test patch point (`service._orchestrator.refinement_loop.run_loop`), not `service._run_refinement_loop` (which no longer exists).
- Always close drivers cleanly: `main.py` calls `Neo4jService.global_close()` and `VLLMClient.global_close()` in a `finally` block. New entry points should do the same.
- Ports are fixed (see Services section); changing them requires updating `run_servers.sh`, `scripts/probe_ports.py`, and any client URLs.
- The MS GraphRAG LiteLLM Router monkey-patch (`official_indexer.py::_install_litellm_router_for_gen`) needs a `contextvars.ContextVar` re-entry guard. `Router.acompletion` internally calls `litellm.acompletion`; without the guard the patched function recurses to RecursionError. The guard sets a contextvar on entry, checks it on every call, and bypasses to the original `acompletion` while it's True.
- Do not introduce new benchmarks; FinanceBench is the only target.
- Domain-specific post-refinement override builders (operating-margin/segment-drag/quick-ratio/debt-securities/capital-intensity/dividend-stability) were removed during the paper-aligned refactor. Don't reintroduce them — they were FinanceBench-specific patches not in the paper.

## Retrieval/refinement quality fixes (2026-05 sessions)

Several systemic defects were caught by single-company AMD deep-dives and 47-query sample dissection. Future edits should preserve these mechanisms:

- **Reranker top-up fallback** (`retrieval/rerank.py`, `retrieval/traversal.py`): the τ_r gate previously returned only chunks crossing the threshold (often 1 of `top_k`); now tops up to `top_k` with the next-best ungated candidates. Empirical: bootstrap retrieval went from ~1 chunk to 7-12 across queries.
- **Cross-turn visited dedup** (`AgentState.visited_chunk_ids` in `state.py`): every `graph_search` call accepts `excluded_chunk_ids` and writes back the IDs it returned, so subsequent turns explore fresh territory. NEXT/HOP traversal otherwise re-surfaces the same hub chunks via different seed paths (30-50% duplication observed before the fix; now 0%).
- **Runtime HOP always-on** (`retrieval/traversal.py`): the runtime Q+ ANN expansion is run unconditionally to complement the offline pre-built HOP edges, recovering answer chunks whose Q+ has no HOP peers above τ_r.
- **Rerank query simplification** (`GraphRAG._simplified_rerank_query`): the cross-encoder reranker collapses on long verbose queries (verified: same chunk drops 0.94 → 0.03 when wrapped in "Answer as if you are an equity analyst..."). A small LLM call strips role-framing/preludes; cached per user query.
- **Strict company filter in retrieval** (`retrieval/rerank.py`, `retrieval/traversal.py`): when the query is company-anchored, drop cross-company chunks rather than just demoting them. Falls back to no-filter if strict empties the pool. Company keys are derived from the user query (not from joined LLM-generated entities, which produce compound noise like "incomestatementofoperationsamdfy21").
- **Reflection structured pre-check** (`utils/prompts/synthesis.py` REFLECTION_PROMPT): forces the model to internally run four checks (A arithmetic+operand identity, B enumeration coverage, C intent alignment, D citation+verbatim+recompute) and emit `checks_performed: [...]` in the JSON output. Without this scaffold the default-PASS rule silently wins. Verbatim-claim and recompute-on-claim rules under D catch fabricated entities/numbers ("Chicago Stock Exchange" type) and operand-selection errors that the calculator cannot self-detect.
- **Refinement-failure abstain fallback** (`stages/refinement.py` `_unfixable_defect_persists`): when the refinement signature converges and reflection still flags `arithmetic_check=fail`, `operand_slot_mismatch`, `formula_identity_mismatch`, `numeric_compute_answer_mismatch_with_calculator_result`, `fabricated_citation`, or any entity-mismatch critique, the rolled-back wrong answer is overridden with `@@ANSWER: insufficient evidence`. Generic principle: confident-wrong (after self-detection + failed repair) is worse than honest abstain. Trace event: `refinement_force_insufficient`.
- **Synthesis intent-preservation** (`COMPLEX_AGENT_PROMPT_TEMPLATE` rule 13a/13b): when one required operand is missing but CONTEXT contains grounded narrative for the underlying intent (drivers, qualitative direction), report that grounded narrative instead of abstaining (13a). Paired with explicit "no concept substitution" guard (13b) — answering an adjacent question about a different metric is a hard FAIL.
- **TOOL_HISTORY mandatory variation** (`AGENT_EXECUTION_SYSTEM_PROMPT` rule 9): the next graph_search MUST include at least one new entity token; period/anchor restate variants on `_mismatch` reject_reasons; abandon-slot after 3 similar failures.

## Cache layout

`data/index_cache/v3/<corpus_tag>/doc__<filehash>__adapt=1-summary=1-table=1-qm=1-qp=1.json` — content-addressed by document SHA8; ablation flags in the filename mean different RAG_ABLATION_* settings produce different cache entries. Same-content document under a different `corpus_tag` is a cache miss by directory namespacing — copy across directories (hardlink works) when intentional. The chunking-algorithm change requires either a fresh corpus_tag or bumping `_CHUNK_CACHE_VERSION` (currently "v3").

## OpenAI cost knobs

- `EVAL_MODEL` — used per-query for the combined judge call (~3-5K input tokens, ~200-500 output).
- `REFLECTION_MODEL` / `REFINEMENT_MODEL` — fired up to (R_max+1=3) times per query in agentic-on flows; each call ~3-6K input.
- Empty values fall back to `DEFAULT_MODEL` (local served model name from `VLLM_SERVED_MODEL_NAME`, default `generation-model`).
- Setting `OPENAI_API_KEY` enables the routing; without it everything stays local.
