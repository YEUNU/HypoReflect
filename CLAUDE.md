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
  - `models/hoprag/official_indexer.py` — drives `third_party/HopRAG/HopBuilder.QABuilder`. paddlenlp NER is stubbed and replaced with spaCy `en_core_web_sm`. Three monkey-patches applied at startup inside `_setup_hoprag_modules`:
    1. **Doc-level parallelism** (`_patch_create_nodes_offline_parallel`): replaces the sequential `create_nodes_offline` loop with a `ThreadPoolExecutor(max_workers=RAG_HOP_DOC_WORKERS)` — processes multiple documents concurrently. Inner per-doc chunk parallelism (`RAG_HOP_MAX_THREADS`) is preserved, so total concurrent LLM calls ≈ `DOC_WORKERS × MAX_THREADS`.
    2. **Batched node INSERT** (`_patch_create_nodes_cache_batched`): replaces one-at-a-time `CREATE … RETURN id(n)` with `UNWIND $rows AS row CREATE … RETURN id(n)` in batches of `RAG_HOP_NODE_BATCH` (default 200). Removes the `time.sleep(1)` that fired every 10 docs. Neo4j UNWIND guarantees ID order matches input order; a `len` assertion guards against silent mismatch.
    3. **Batched edge INSERT** (`_patch_create_edge_batched`): wraps the pandas2-patched `create_edge` — runs it with a `_NullDriver` to populate `self.edges` / `self.abstract2chunk` without touching Neo4j, then inserts both edge types in `UNWIND` batches of `RAG_HOP_EDGE_BATCH` (default 500). numpy int64/float32 arrays are explicitly cast to Python int/list for Bolt serialization.
- `utils/prompts/` — externalized prompt templates (e.g. `RERANKER_INSTRUCTION`); always source prompts from here, do not inline.
- `tools/benchmark_report.py` — post-processes `data/results/<timestamp>/...` after a benchmark run.

## Data preservation — DO NOT DELETE

**Never delete the following without explicit user instruction.** These are the result of hours-to-days of indexing that cannot be easily recreated:

| Asset | Location | Rebuilt by |
|---|---|---|
| HypoReflect Neo4j graph | Docker container `hyporeflect-neo4j` (port 7687) | `./run_index.sh --model hyporeflect --corpus-tag <tag>` |
| HypoReflect chunk/QA cache | `data/index_cache/v3/<corpus_tag>/` | Same indexing run |
| HopRAG Neo4j graph | Same container, `HO_<tag>` labels | `./run_index.sh --model hoprag --corpus-tag <tag>` |
| HopRAG stage-1 cache | `data/hoprag_output/<corpus_tag>/_cache/` | Same — contains `docid2nodes.json` + `node2questiondict.pkl` |
| MS GraphRAG parquet+lancedb | `data/ms_graphrag_output/<corpus_tag>/` | `./run_index.sh --model ms_graphrag --corpus-tag <tag>` |
| Naive Neo4j graph | Same container, `NA_<tag>` labels | `./run_index.sh --model naive --corpus-tag <tag>` |

**Specific prohibitions:**
- Do **not** run `MATCH (n) DETACH DELETE n` or any label-level `DETACH DELETE` on Neo4j without the user explicitly asking to wipe a specific corpus.
- Do **not** `rm -rf data/index_cache/`, `data/hoprag_output/`, or `data/ms_graphrag_output/`.
- Do **not** issue large batched `DETACH DELETE` queries — they can crash Neo4j Community Edition's transaction engine (observed: `TransactionStartFailed` crash loop requiring `docker restart hyporeflect-neo4j`).
- If cleanup of outdated Neo4j labels is needed, delete in batches of ≤200 nodes per transaction using the project's `.venv` bolt driver, not the HTTP API.

**Currently active indexes (as of 2026-05-13):**

| Label prefix | corpus_tag | Strategy |
|---|---|---|
| `HY_full_v19_hyporeflect_` | full_v19_hyporeflect | hyporeflect (full) |
| `HY_full_v19_hyporeflect_no_chunk_` | full_v19_hyporeflect_no_chunk | hyporeflect (RAG_ABLATION_CHUNKING) |
| `HY_full_v19_hyporeflect_no_summary_` | full_v19_hyporeflect_no_summary | hyporeflect (RAG_ABLATION_SUMMARY) |
| `HY_full_v19_hyporeflect_no_table_` | full_v19_hyporeflect_no_table | hyporeflect (RAG_ABLATION_TABLE) |
| `NA_full_v19_naive_T1C1S1_` | full_v19_naive | naive |
| `HO_full_v19_hoprag` | full_v19_hoprag | hoprag (re-indexing in progress) |

MS GraphRAG: `data/ms_graphrag_output/full_v19_ms_graphrag/` (parquet + lancedb, no Neo4j).

## Neo4j corpus isolation

`GraphRAG.__init__` derives **all** Neo4j labels and index names from `(strategy, corpus_tag)`, e.g. `HY_sample_ocr_Chunk`, `hyporeflect_sample_ocr_vector_idx`. So:

- Different `--corpus-tag` values **never collide** in the same Neo4j instance — running indexing for `sample_raw` does not invalidate `sample_ocr`.
- The same tag across strategies also does not collide (prefix differs: `HY_`, `NA_`, `HO_`, `MS_`).
- `--clear-graph` runs `MATCH (n) DETACH DELETE n` — it wipes **everything**, not just the active tag. **Never use** unless the user explicitly requests a full wipe.
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
- `RAG_ABLATION_Q_PLUS` (default `True`) — when set to `False`, disables the Stage 2 Q+ hypothetical-answer expansion in `models/hyporeflect/retrieval/retrieve.py`. Empirically on FinanceBench failure samples, Q+ demotes the narrative/MD&A chunks that contain verbatim answers and inflates hallucination on a subset of queries; disabling it lifts agentic-off judge by ~5pp on the failure-curated sample20 (HopRAG-equivalent or better). Setting it to `False` is a paper-supported ablation, not a permanent default change.
- `RAG_ABLATION_Q_MINUS` (default `True`) — symmetric switch for Stage 1 Q- channel; rarely useful alone.

LLM-call reduction toggles (2026-05 sessions, all default ON unless noted):
- `RAG_ENABLE_ATOMIZATION` (default `false`) — when False, the execution loop skips `_extract_context_atoms` + `_pack_context_atoms` (2 LLM calls per ledger refresh) and uses the deterministic `_build_context_excerpt` directly. Saves ~6-12 LLM calls per agentic-on query. Atomization re-rephrases chunks into compact atoms and re-judges relevance — a responsibility the retrieval+ledger layers already own.
- `RAG_DETERMINISTIC_SLOT_FILL` (default `false`) — when False, `_run_compute_slot_fill_before_synthesis` and `_run_compute_slot_realign_before_synthesis` are skipped. The earlier deterministic regex-based fallback bound wrong operands when the LLM ledger returned 0 entries (e.g., negative `current liabilities` from a cash-flow working-capital line). Disabling lets missing slots stay missing — synthesis falls through to LLM-on-context and answers honestly or abstains.
- `RAG_PLANNING_MERGE` (default `true`) — when True, planning emits `plan` + `filter_policy` in a single JSON call (`PLANNING_MERGED_PROMPT`). Saves 1 LLM call per query vs the legacy two-pass form.
- `RAG_REFINEMENT_REJUDGE` (default `false`) — when False, the refinement loop skips re-running `reflection.run()` after each refinement, relying on the structural non-regression guard inside `_prefer_refined_candidate` (citation presence, single @@ANSWER prefix, grounded vs insufficient). Saves R_max LLM calls per query.
- `RAG_MAX_REFINEMENT` (default `1`) — paper R_max. Earlier default 2 ran 4 extra LLM calls per query (1 refinement + 1 reflection re-judge × 2 iterations) without meaningful J/H lift on local-4B refinement.
- `RAG_AGENT_INLINE_LEDGER` (default `true`, option D) — when True, the main agent LLM emits `EVIDENCE:` lines alongside its tool_call in the same response, and the post-tool `_extract_evidence_entries` LLM call is skipped. A fault-tolerant parser (`SearchSupport._parse_inline_evidence_pairs`) extracts pairs from the agent response; deterministic slot-binding (`_bind_inline_pairs_to_slots`) matches each pair to a required slot by metric substring and validates value-in-context (commas/$/whitespace-normalised substring check). Partial / malformed lines are silently skipped. Saves ~2-3 LLM calls per query.
- `RAG_MAX_TOOL_CALLS` (default `3`) — paper T_max. **Includes bootstrap** as 1 tool call against the budget. With default 3: bootstrap (1) + up to 2 agent-loop tool calls = 3 total. The `MAX_AGENT_TURNS = 6` constant in `core/config.py` is the outer for-loop iteration cap; the tool-call counter usually terminates the loop first.
- `RAG_AGENTIC_OFF_GRAPH_DEPTH` (default `1`) — when >0, agentic-off uses `graph_search` with `force_expand=True` instead of plain `retrieve()`. Adds deterministic 1-hop NEXT|HOP traversal, no LLM continuation check (`_need_more_for_next_depth` is skipped via `force_expand`). depth=0 restores the legacy `retrieve()`-only path.

Retrieval meta-boost knobs (`retrieval/text_utils.py::_meta_boost_for_node`):
- `RAG_YEAR_BOOST` (default `0.25`)
- `RAG_DOC_TYPE_BOOST` (default `0.15`)
- `RAG_COMPANY_BOOST` (default `0.35`)
- `RAG_FINANCE_MARKER_BOOST` (default `0.0`) — was 0.15 prior to 2026-05. The non-zero value promoted statement-table pages over narrative/MD&A pages that often hold the verbatim answer (e.g., 3M MD&A page 41: "net PP&E totaled $8.7B"). Restored with `RAG_FINANCE_MARKER_BOOST=0.15`.

**Ablation flag scope (important):** `RAG_ABLATION_TABLE/CHUNKING/SUMMARY` are read only by `models/hyporeflect/indexing/chunking.py` (and `naive_rag.py` purely as a cache-namespace key — no behavior change). The `hoprag` and `ms_graphrag` baselines run their published indexing pipelines verbatim and **ignore these flags**; setting them on a baseline run yields an index identical to the `_full` variant. The catchup/parallel scripts only run ablations on HypoReflect — do not reintroduce baseline ablations.

HopRAG indexing knobs (read by `models/hoprag/official_indexer.py`):
- `RAG_HOP_GEN_API_BASES` — comma-separated gen URLs (default `28000,28010`). Round-robin patch rotates across live endpoints per request; health-checked at startup.
- `RAG_HOP_MAX_THREADS` (default `8`) — `ThreadPoolExecutor` size for per-doc chunk parallelism inside `get_single_doc_qa`.
- `RAG_HOP_DOC_WORKERS` (default `4`) — outer doc-level parallelism; total concurrent LLM calls ≈ `DOC_WORKERS × MAX_THREADS`.
- `RAG_HOP_NODE_BATCH` (default `200`) — UNWIND batch size for node INSERT in Stage 2a (`create_nodes_cache`).
- `RAG_HOP_EDGE_BATCH` (default `500`) — UNWIND batch size for edge INSERT in Stage 2b (`create_edge`).

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

### FinanceBench 3-way taxonomy + abstain detection

Result rows carry a `financebench_label` field with one of `Correct Answer` / `Incorrect Answer` / `Refusal`, matching the `label` field on the official human-annotated results at <https://github.com/patronus-ai/financebench/tree/main/results>. Aggregates `financebench_{correct,incorrect,refusal}_count` and `_rate` are written into the per-run summary JSON.

Abstain detection is centralized in `utils/abstain.py` (`ABSTAIN_PHRASES`, `is_abstain()`, `financebench_label()`). Three call sites use it:
- `utils/metrics.py::_is_insufficient_text` — deterministic safety override for `hallucination` (an honest abstention is never a hallucination).
- `tools/benchmark_report.py::_is_insufficient_answer_text` — post-hoc reporting layer.
- `cli/benchmark.py` per-row scoring — sets `answer_attempted = 0` for refusals and bakes the 3-way label.

**Why centralized.** The previous one-substring rule (`"insufficient evidence"`) only caught Hypo's pipeline-specific prefix and treated HopRAG / naive natural-language abstains ("I do not know", "the context does not contain") as `answer_attempted = 1` Incorrect Answers. On HopRAG full 150 this swung 73 rows from `Incorrect` to `Refusal` (AttRate 1.00 → 0.50) — a metric-fairness bug, not a model gap. `tools/fairness_audit.py` re-derives the same 3-way breakdown for any historical JSON without re-calling the LLM judge.

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
- **Runtime HOP gated on HOP_MODE** (`retrieval/traversal.py`): runtime Q+ ANN expansion (`_runtime_hop_candidates`) now runs only when `RAGConfig.HOP_MODE == "runtime"`. In the default offline mode the Cypher traversal uses pre-built `[:NEXT|HOP]` edges (paper §3.1.4) and skips the live ANN supplement — adding both double-sourced HOP candidates and re-introduced chunks the offline rerank filter (tau_r=0.5, L_hop=5) had deliberately excluded, displacing ~3-4 seed chunks per query from the final top_k slice. Set `RAG_HOP_MODE=runtime` for the dynamic (HopRAG-style) path: Cypher walks `[:NEXT]` only and runtime Q+ ANN takes over as the HOP source.
- **Rerank query simplification** (`GraphRAG._simplified_rerank_query`): the cross-encoder reranker collapses on long verbose queries (verified: same chunk drops 0.94 → 0.03 when wrapped in "Answer as if you are an equity analyst..."). A small LLM call strips role-framing/preludes; cached per user query.
- **Strict company filter in retrieval** (`retrieval/rerank.py`, `retrieval/traversal.py`): when the query is company-anchored, drop cross-company chunks rather than just demoting them. Falls back to no-filter if strict empties the pool. Company keys are derived from the user query (not from joined LLM-generated entities, which produce compound noise like "incomestatementofoperationsamdfy21").
- **Reflection structured pre-check** (`utils/prompts/synthesis.py` REFLECTION_PROMPT): forces the model to internally run four checks (A arithmetic+operand identity, B enumeration coverage, C intent alignment, D citation+format/verbatim) and emit `checks_performed: [...]` in the JSON output. The 2026-05 prompt simplification collapsed the earlier verbatim/formula-identity exemption ladder (13a/13b/13c) into a single principle per check; line-item synonym equivalence is now in the synthesis prompt itself (capex ≡ purchases of PP&E, revenue ≡ net sales, etc.), so reflection no longer needs separate exemption clauses.
- **Refinement-failure abstain fallback** (`stages/refinement.py` `_unfixable_defect_persists`): when the refinement signature converges and reflection still flags **`numeric_compute_answer_mismatch_with_calculator_result`** or **`fabricated_citation`** (or any entity-mismatch critique), the rolled-back wrong answer is overridden with `@@ANSWER: insufficient evidence`. Generic principle: confident-wrong (after self-detection + failed repair) is worse than honest abstain. Trace event: `refinement_force_insufficient`. The earlier broader trigger set (`operand_slot_mismatch`, `operand_magnitude_anomaly`, `formula_identity_mismatch`) was removed in the 2026-05 simplification — those tokens fired on synonym-only differences and produced false force-insufficient cascades after the synthesis prompt absorbed synonym handling.
- **Synthesis layered-exception removal** (`COMPLEX_AGENT_PROMPT_TEMPLATE`): the prior 16-rule prompt with sub-rules 13a (intent preservation), 13b (no concept substitution), 13c (term equivalence list) plus matching reflection exemptions was collapsed to a single short prompt (~49 lines, 12 rules). Synonym equivalence is one rule, anti-concept-substitution is one rule, anti-over-abstain is one rule. The 4B model follows short rules and ignores long ones; layered exceptions caused inconsistent behavior across queries.
- **Agentic-OFF prompt minimization** (`models/hyporeflect/orchestrator.py::_build_simple_answer_prompt`): the baseline retrieve→synthesize path now uses an 8-line prompt structurally identical to HopRAG's (`You are a financial analyst. Answer using only the provided context. If insufficient, say you do not know.`). No `@@ANSWER:` prefix mandate, no inline citation mandate, no synonym block — `_ensure_answer_prefix` attaches `@@ANSWER:` downstream, and doc/page match metrics come from retrieved nodes, not the answer text. The earlier 5-rule version that ADDed Hypo-specific abstain phrasing collapsed the 4B model into "insufficient evidence" prefix even when CONTEXT contained the answer (AttRate 0.45 vs HopRAG 1.00 on full 150).
- **TOOL_HISTORY mandatory variation** (`AGENT_EXECUTION_SYSTEM_PROMPT` rule 9): the next graph_search MUST include at least one new entity token; period/anchor restate variants on `_mismatch` reject_reasons; abandon-slot after 3 similar failures.
- **Inline ledger extraction (option D)** (`stages/execution/search.py` + `AGENT_EXECUTION_SYSTEM_PROMPT`): the main agent LLM now emits `EVIDENCE: value=… | citation=[[…]] | metric=…` lines alongside its tool_call in the same response. Tolerant regex parser (`_INLINE_EVIDENCE_RE`) extracts pairs; malformed lines are silently dropped. Deterministic slot binding by metric substring + value-in-context substring (commas/$/whitespace-normalised) replaces the per-turn `_extract_evidence_entries` LLM call. Bootstrap still calls the LLM extractor (no agent at bootstrap time). Gated by `RAG_AGENT_INLINE_LEDGER` (default true).
- **EVIDENCE_LEDGER_PROMPT simplification** (`utils/prompts/execution.py`): 14 rules → 7. Removed rule 5 ("strict qualifier match") which caused 4B to reject `cash from operations` ≡ `net cash provided by operating activities` synonym matches. Added an explicit preamble: "Retrieval has already filtered CONTEXT for relevance — your job is to EXTRACT, not re-judge". Reject signals narrowed to entity/period mismatch, guidance values, and placeholder text. The rescue prompt is unchanged.
- **Planning merged into single call** (`stages/planning.py::run` + `PLANNING_MERGED_PROMPT`): outputs `{plan, filter_policy}` in one JSON call, saving 1 LLM call vs the prior text-plan-then-policy-JSON two-pass form.

## Cache layout

`data/index_cache/v3/<corpus_tag>/doc__<filehash>__adapt=1-summary=1-table=1-qm=1-qp=1.json` — content-addressed by document SHA8; ablation flags in the filename mean different RAG_ABLATION_* settings produce different cache entries. Same-content document under a different `corpus_tag` is a cache miss by directory namespacing — copy across directories (hardlink works) when intentional. The chunking-algorithm change requires either a fresh corpus_tag or bumping `_CHUNK_CACHE_VERSION` (currently "v3").

## OpenAI cost knobs

- `EVAL_MODEL` — used per-query for the combined judge call (~3-5K input tokens, ~200-500 output).
- `REFLECTION_MODEL` / `REFINEMENT_MODEL` — fired up to (R_max+1=3) times per query in agentic-on flows; each call ~3-6K input.
- Empty values fall back to `DEFAULT_MODEL` (local served model name from `VLLM_SERVED_MODEL_NAME`, default `generation-model`).
- Setting `OPENAI_API_KEY` enables the routing; without it everything stays local.
