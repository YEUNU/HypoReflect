# Reviewer-response ablation runbook

This document covers the three follow-up experiments requested in review:

1. **Paired bootstrap CI / McNemar significance test** (cheapest, no GPU)
2. **Q⁻/Q⁺ directionality ablation** (defends Predictive Knowledge Mapping novelty)
3. **Runtime vs offline HOP** (defends the rank-based pre-built HOP claim)

All three are now wired up. Below is exactly what to run.

## 0. Prerequisites

- Branch: `single-gpu-config` (this branch). Single-GPU vLLM config in `run_servers.sh` already adjusted (gen 0.55 / embed 0.15 / rerank 0.15 share GPU 0; OCR runs alone at 0.85).
- Neo4j up: `./run_servers.sh neo4j`
- vLLM stack up (gen + embed + rerank): `./run_servers.sh gen && ./run_servers.sh embed && ./run_servers.sh rerank`
  Confirm with `python probe_ports.py`.
- `data/financebench_queries.json` and existing FinanceBench `--corpus-tag full` index already present (verified 2026-05-07).

## 1. Bootstrap CI / McNemar

No new runs needed — operates on existing `data/results/<ts>/.../*.json`.

```bash
# Pair every system in a result timestamp against HypoReflect/full:
python tools/bootstrap_ci.py \
  --base-dir data/results/<TIMESTAMP> \
  --reference hyporeflect/full \
  --iterations 10000 \
  --json-out data/results/<TIMESTAMP>/_significance.json

# Or compare a single pair:
python tools/bootstrap_ci.py \
  --a 'data/results/<TIMESTAMP>/hyporeflect/full/refl_on/agentic_off/*.json' \
  --b 'data/results/<TIMESTAMP>/hoprag/hoprag_full/agentic_off/*.json'
```

**Outputs** per pair:

- Per-metric mean(A), mean(B), mean diff (A − B), 95% paired bootstrap CI, `*` if CI excludes zero.
- A>B / B>A / tie counts.
- McNemar two-sided exact p-value on `judge_score >= 0.5` (binary).

The metrics covered: `llm_judge_score`, `hallucination`, `doc_match`, `page_match`, `answer_attempted`, `latency`. `hallucination` and `latency` are treated as lower-is-better when counting wins/losses; the bootstrap diff itself is always A−B regardless.

**Use these to:**

- Verify the `0.34 → 0.20` headline (Table 1) holds with calibrated uncertainty.
- Test the **fragile** `0.14 → 0.11` hallucination claim (Table 2 GPT-5-mini): a 4–5 question swing on N=150 will likely have a CI that overlaps zero. If so, soften the claim language.
- Compare GPT-5-mini reflection vs Qwen-4B reflection on the same paired queries.

Cost: seconds. No GPU.

## 2. Q⁻/Q⁺ directionality ablation

Three new toggles added to `core/config.py`:

| env var | default | effect |
|---|---|---|
| `RAG_ABLATION_Q_MINUS` | `True` | When `False`: skip Q⁻ embedding/text generation at indexing; retrieve uses body channel only at stage 1 |
| `RAG_ABLATION_Q_PLUS`  | `True` | When `False`: skip Q⁺ embedding/text generation; **also** disables offline HOP construction (HOP is anchored on Q⁺); retrieve never enters stage-2 expansion |
| `RAG_HOP_MODE`         | `offline` | `runtime` skips offline HOP and expands frontier at query time via Q⁺ ANN+rerank (see §3) |

**Re-index per variant (each gets its own `corpus-tag` namespace so existing `full` index is untouched):**

```bash
# Q⁻ only (no Q⁺ channel, no HOP edges)
RAG_ABLATION_Q_PLUS=False ./run_index.sh \
    --model hyporeflect --corpus-tag full_qminus_only --skip-server

# Q⁺ only (no Q⁻ channel; HOP still built since Q⁺ is on)
RAG_ABLATION_Q_MINUS=False ./run_index.sh \
    --model hyporeflect --corpus-tag full_qplus_only --skip-server

# (control) re-confirm Full path also works under the new code
./run_index.sh --model hyporeflect --corpus-tag full_qcontrol --skip-server
```

**Benchmark (must echo the SAME env var so retrieval matches the index):**

```bash
RAG_ABLATION_Q_PLUS=False ./run_benchmark.sh \
    --model hyporeflect --corpus-tag full_qminus_only

RAG_ABLATION_Q_MINUS=False ./run_benchmark.sh \
    --model hyporeflect --corpus-tag full_qplus_only

./run_benchmark.sh --model hyporeflect --corpus-tag full_qcontrol
```

Then run bootstrap CI to compare each variant against `hyporeflect/full`.

**What this isolates:**

- `full` vs `qminus_only` → marginal contribution of Q⁺ + HOP.
- `full` vs `qplus_only` → marginal contribution of Q⁻ as the seed channel.
- `qminus_only` vs `qplus_only` → which channel carries more of the lift.

**Single-channel (HopRAG-style merged channel) is intentionally not auto-wired** — that requires a different prompt schema (one combined hypothetical query rather than directional split) and is closer to running the HopRAG baseline on HypoReflect's chunk supply. Hold off unless reviewer specifically asks.

**Cost on single GPU (32GB):** 3 indexing runs × ~ length of one Full indexing pass. Each ablation reuses the same upstream OCR/segmentation; the LLM cost is in Q-pair generation, not chunking. Plan ~ 3× Full-index wall time.

## 3. Runtime vs offline HOP

Same code path; flip `RAG_HOP_MODE`:

```bash
# Runtime HOP: q_plus embeddings still stored, no HOP edges built.
# At query time, frontier expansion runs Q+ ANN per source chunk.
RAG_HOP_MODE=runtime ./run_index.sh \
    --model hyporeflect --corpus-tag full_runtime_hop --skip-server

RAG_HOP_MODE=runtime ./run_benchmark.sh \
    --model hyporeflect --corpus-tag full_runtime_hop
```

**Implementation notes:**

- `models/hyporeflect/graphrag_parts/pipeline_support.py` skips offline HOP edge writes when `HOP_MODE != "offline"`.
- `models/hyporeflect/graphrag_parts/retrieval_support.py::graph_search` switches the cypher edge pattern from `[:NEXT|HOP]` to `[:NEXT]` and unions in fresh ANN candidates from `_runtime_hop_candidates`. NEXT (sequential adjacency) is preserved in both modes — only HOP changes.
- Per-source `K=10` ANN, `τ_r=0.5` reranker threshold inherited from `RAGConfig` — same as offline construction, so the comparison is timing-only.

**Compare:**

```bash
python tools/bootstrap_ci.py \
  --a 'data/results/<TIMESTAMP>/hyporeflect/full/refl_on/agentic_off/*.json' \
  --b 'data/results/<TIMESTAMP>/hyporeflect/full_runtime_hop/refl_on/agentic_off/*.json'
```

**Reads to expect:**

- Identical or near-identical `judge` / `halluc` → "offline HOP gives the same quality with lower latency" (a clean win the paper currently can only conjecture).
- Runtime materially better → "offline HOP fixes the graph at indexing reranker score; query context can't reach late-arriving links" (the trade-off §5.2 already hypothesises but doesn't measure).
- Either outcome is publishable. The current "future work" line in §6 / §Limitations becomes a paragraph in §5.2 / §4.5.

**Cost:** 1 indexing run + 1 benchmark run. No additional Q-pair generation since Q⁺ is still produced (we just don't pre-link them).

## 4. Recommended order on a single GPU

1. **Today (no GPU):** run bootstrap CI on existing `data/results/`. If headline 0.34 vs 0.20 holds with CI excluding zero, you can keep the framing; if 0.14→0.11 has CI overlapping zero, plan a wording fix.
2. **Next pass (gen+embed+rerank up):** index `full_runtime_hop`, then both Q⁻/Q⁺ variants. Re-benchmark in order:
   - `full_runtime_hop` (one run, fastest signal)
   - `full_qminus_only` (most expected to drop)
   - `full_qplus_only` (smaller drop expected)
3. After each benchmark, re-run `tools/bootstrap_ci.py --base-dir data/results/<new_ts> --reference hyporeflect/full`. The output is a paste-ready table for the camera-ready response.

## 5. What did NOT change

- Prompt templates (`utils/prompts/`) untouched.
- `graphrag.py` core attributes (`self.llm`, `self.q_minus_vector_index`, etc.) untouched.
- Existing `corpus-tag full` / `no_chunk` / `no_summary` / `no_table` indices in Neo4j untouched. New ablation tags use disjoint label namespaces so they never collide.
- HopRAG / MS-GraphRAG / Naive baselines untouched.
