#!/bin/bash
#
# run_all_indexing_parallel.sh — Build every index needed for the paper.
#
# What this produces (7 corpora total):
#
#   1. naive_full              — naive baseline
#   2. hoprag_full             — official HopRAG indexer
#   3. ms_graphrag_full        — official MS GraphRAG (parquet outputs)
#   4. hyporeflect_full        — HypoReflect full pipeline
#   5. hyporeflect_no_table    — HypoReflect ablation: no table-to-text
#   6. hyporeflect_no_chunk    — HypoReflect ablation: no adaptive chunking
#   7. hyporeflect_no_summary  — HypoReflect ablation: no rolling summary
#
# Why no baseline ablations: RAG_ABLATION_TABLE/CHUNKING/SUMMARY are read only
# by HypoReflect (`models/hyporeflect/indexing/chunking.py`). The naive,
# hoprag, and ms_graphrag pipelines run their published code verbatim and
# ignore those flags — running them with `_no_*` corpus tags would produce
# indexes byte-identical to the `_full` variant. Paper Table 1 reports
# ablations on HypoReflect only.
#
# Concurrency notes:
#  - All 7 are dispatched in parallel by default; vLLM gen (port 28000) and
#    gen2 (port 28010) absorb the contention. With ~32 GB GPUs and Qwen3-4B
#    in each gen, the dominant bottleneck is MS GraphRAG's extract_graph
#    stage (~30h on the full ~33k text_unit corpus).
#  - When `VLLM_URL_2` is set, HypoReflect's `VLLMClient` round-robins between
#    gen and gen2. The MS GraphRAG official indexer installs a LiteLLM Router
#    across `RAG_MS_GEN_API_BASES` for the same effect.
#
# Usage:
#   ./run_all_indexing_parallel.sh                # sample (one company per sector), OCR corpus
#   ./run_all_indexing_parallel.sh --full         # full FinanceBench OCR corpus
#   ./run_all_indexing_parallel.sh --n 1          # first sample company only
#   ./run_all_indexing_parallel.sh --skip-baselines  # only HypoReflect family

set -e

SAMPLE_FLAG="--sample --ocr"
N_COMPANIES=""
SKIP_BASELINES="false"
LOG_DIR="logs/indexing_parallel"
mkdir -p "$LOG_DIR"

# Retrieval tuning defaults — override via env. These only affect HypoReflect /
# naive at index time; hoprag and ms_graphrag are insensitive.
export NEO4J_FULLTEXT_ANALYZER="${NEO4J_FULLTEXT_ANALYZER:-english}"
export RAG_RECREATE_TEXT_INDEX="${RAG_RECREATE_TEXT_INDEX:-False}"
export RAG_ENABLE_QUERY_REWRITE="${RAG_ENABLE_QUERY_REWRITE:-True}"
export RAG_QUERY_REWRITE_COUNT="${RAG_QUERY_REWRITE_COUNT:-2}"
export RAG_QUERY_REWRITE_WEIGHT="${RAG_QUERY_REWRITE_WEIGHT:-0.85}"
export RAG_META_BOOST_WEIGHT="${RAG_META_BOOST_WEIGHT:-0.50}"
export RAG_BOILERPLATE_PENALTY_WEIGHT="${RAG_BOILERPLATE_PENALTY_WEIGHT:-0.25}"

# MS GraphRAG knobs — when gen2 is up these widen the LiteLLM Router pool and
# raise the concurrent-request semaphore. Override per-host as needed.
export RAG_MS_GEN_API_BASES="${RAG_MS_GEN_API_BASES:-http://localhost:28000/v1,http://localhost:28010/v1}"
export RAG_MS_CONCURRENT_REQUESTS="${RAG_MS_CONCURRENT_REQUESTS:-48}"

while [ $# -gt 0 ]; do
    case $1 in
        --full)             SAMPLE_FLAG="";              shift ;;
        --n)                N_COMPANIES="--n $2";        shift 2 ;;
        --skip-baselines)   SKIP_BASELINES="true";       shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -n "$N_COMPANIES" ] && [ -z "$SAMPLE_FLAG" ]; then
    echo "INFO: --n provided with --full. Forcing sample subset mode."
    SAMPLE_FLAG="--sample --ocr"
fi

echo "=========================================================="
echo "   Parallel indexing — paper-aligned matrix (7 corpora)"
echo "   Mode:              ${SAMPLE_FLAG:-Full Dataset}"
[ -n "$N_COMPANIES" ] && echo "   Sample companies:  ${N_COMPANIES#--n }"
echo "   Skip baselines:    $SKIP_BASELINES"
echo "   Retrieval tune:    analyzer=${NEO4J_FULLTEXT_ANALYZER}, rewrite=${RAG_ENABLE_QUERY_REWRITE}"
echo "   MS GraphRAG:       bases=${RAG_MS_GEN_API_BASES}, concurrent=${RAG_MS_CONCURRENT_REQUESTS}"
echo "   Logs:              $LOG_DIR/"
echo "=========================================================="

# Pre-start required services once. gen2 starts if available; absence is
# tolerated and the LiteLLM Router falls back to gen only.
echo "Step 0: pre-starting services..."
./run_servers.sh neo4j
./run_servers.sh gen
./run_servers.sh gen2 || true
./run_servers.sh embed
./run_servers.sh rerank

wait_for_server() {
    local url=$1 name=$2 attempt=0 max=300
    echo "Waiting for $name ($url)..."
    while [ $attempt -lt $max ]; do
        if curl -s -o /dev/null -w "%{http_code}" --max-time 2 "$url" | grep -qE "200|401|405"; then
            echo " ✅ $name ready"
            return 0
        fi
        sleep 5; attempt=$((attempt + 1))
    done
    echo " ❌ $name never came up"
    return 1
}

wait_for_server "http://localhost:7474"            "Neo4j"
wait_for_server "http://localhost:28000/v1/models" "Gen (port 28000)"
wait_for_server "http://localhost:28010/v1/models" "Gen2 (port 28010)" || echo "   (optional — Router falls back to gen only)"
wait_for_server "http://localhost:18082/v1/models" "Embed"
wait_for_server "http://localhost:18083/health"    "Rerank"

run_task() {
    local name=$1 cmd=$2
    local log="$LOG_DIR/${name}.log"
    echo "  [STARTED] $name -> $log"
    eval "$cmd --skip-server" > "$log" 2>&1 && echo "  [COMPLETED] $name" || echo "  [FAILED] $name (check $log)"
}

# ---------- Baselines (full only) ----------
if [ "$SKIP_BASELINES" != "true" ]; then
    run_task "naive_full"        "./run_index.sh --model naive        --corpus-tag naive_full        $SAMPLE_FLAG $N_COMPANIES" &
    run_task "hoprag_full"       "./run_index.sh --model hoprag       --corpus-tag hoprag_full       $SAMPLE_FLAG $N_COMPANIES" &
    run_task "ms_graphrag_full"  "./run_index.sh --model ms_graphrag  --corpus-tag ms_graphrag_full  $SAMPLE_FLAG $N_COMPANIES" &
fi

# ---------- HypoReflect family ----------
run_task "hyporeflect_full"        "./run_index.sh --model hyporeflect --corpus-tag hyporeflect_full        $SAMPLE_FLAG $N_COMPANIES" &
run_task "hyporeflect_no_table"    "RAG_ABLATION_TABLE=False    ./run_index.sh --model hyporeflect --corpus-tag hyporeflect_no_table    $SAMPLE_FLAG $N_COMPANIES" &
run_task "hyporeflect_no_chunk"    "RAG_ABLATION_CHUNKING=False ./run_index.sh --model hyporeflect --corpus-tag hyporeflect_no_chunk    $SAMPLE_FLAG $N_COMPANIES" &
run_task "hyporeflect_no_summary"  "RAG_ABLATION_SUMMARY=False  ./run_index.sh --model hyporeflect --corpus-tag hyporeflect_no_summary  $SAMPLE_FLAG $N_COMPANIES" &

echo
echo "Waiting for all indexing tasks to finish..."
wait
echo
echo "All indexing tasks done. Inspect $LOG_DIR/ for per-task logs."
echo "Verify with: python scripts/check_indexes.py"
