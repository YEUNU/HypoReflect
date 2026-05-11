#!/bin/bash
#
# run_all_benchmark_parallel.sh — Run every benchmark needed for the paper.
#
# Matrix (12 runs total) — paper Table 1:
#
#   Baseline (agentic off):
#     1. naive_full
#     2. hoprag_full
#     3. ms_graphrag_full
#     4. hyporeflect_off          (HypoReflect, agentic off — paper §4.4)
#
#   HypoReflect main (agentic on, GPT reflection/refinement):
#     5. hyporeflect_full
#     6. hyporeflect_no_table     (ablation)
#     7. hyporeflect_no_chunk     (ablation)
#     8. hyporeflect_no_summary   (ablation)
#
#   Optional shared agentic-on (uses the agentic_core orchestrator on baselines):
#     9.  naive_agentic_on
#     10. hoprag_agentic_on
#     11. ms_graphrag_agentic_on
#     12. hyporeflect_local       (agentic on, all-local reflection/refinement)
#
# Baseline ablation runs (e.g., naive_no_table, hoprag_no_chunk) are NOT
# included: the index for those corpus tags is identical to the `_full`
# variant — `RAG_ABLATION_*` only affects HypoReflect's chunking pipeline.
#
# Expected indexes (built by run_all_indexing_parallel.sh):
#   naive_full / hoprag_full / ms_graphrag_full
#   hyporeflect_full / hyporeflect_no_table / hyporeflect_no_chunk
#   hyporeflect_no_summary
#
# All 12 runs are dispatched in parallel by default. They share local vLLM
# capacity; reflection/refinement run on OpenAI when keys are set (does not
# load the local GPUs).
#
# Usage:
#   ./run_all_benchmark_parallel.sh                  # sample matrix
#   ./run_all_benchmark_parallel.sh --full           # full FinanceBench
#   ./run_all_benchmark_parallel.sh --n 1            # one sample company
#   ./run_all_benchmark_parallel.sh --no-agentic-on  # skip the 4 agentic-on baseline runs

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$SCRIPT_DIR/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    else
        PYTHON_BIN="$(command -v python)"
    fi
fi

SAMPLE_FLAG="--sample"
N_COMPANIES=""
LOG_DIR="logs/benchmark_parallel"
mkdir -p "$LOG_DIR"
FAIL_MARKER="$LOG_DIR/.failed_tasks"
: > "$FAIL_MARKER"
RUN_AGENTIC_ON_BASELINES="${RAG_RUN_AGENTIC_ON_BASELINES:-True}"

# Retrieval tuning defaults (override via env when needed)
export NEO4J_FULLTEXT_ANALYZER="${NEO4J_FULLTEXT_ANALYZER:-english}"
export RAG_ENABLE_QUERY_REWRITE="${RAG_ENABLE_QUERY_REWRITE:-True}"
export RAG_QUERY_REWRITE_COUNT="${RAG_QUERY_REWRITE_COUNT:-2}"
export RAG_QUERY_REWRITE_WEIGHT="${RAG_QUERY_REWRITE_WEIGHT:-0.85}"
export RAG_META_BOOST_WEIGHT="${RAG_META_BOOST_WEIGHT:-0.50}"
export RAG_BOILERPLATE_PENALTY_WEIGHT="${RAG_BOILERPLATE_PENALTY_WEIGHT:-0.25}"

# MS GraphRAG knobs — only matter when the strategy itself is ms_graphrag.
export RAG_MS_GEN_API_BASES="${RAG_MS_GEN_API_BASES:-http://localhost:28000/v1,http://localhost:28010/v1}"
export RAG_MS_CONCURRENT_REQUESTS="${RAG_MS_CONCURRENT_REQUESTS:-48}"

# Unified result directory for this parallel run
export RAG_BENCHMARK_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

while [ $# -gt 0 ]; do
    case $1 in
        --full)              SAMPLE_FLAG="";                       shift ;;
        --n)                 N_COMPANIES="--n $2";                 shift 2 ;;
        --no-agentic-on)     RUN_AGENTIC_ON_BASELINES="False";     shift ;;
        # Legacy alias (older runs called it --no-agentic).
        --no-agentic)        RUN_AGENTIC_ON_BASELINES="False";     shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -n "$N_COMPANIES" ] && [ -z "$SAMPLE_FLAG" ]; then
    echo "INFO: --n provided with --full. Forcing sample subset mode."
    SAMPLE_FLAG="--sample"
fi

# Auto-pick the tagged queries file when available.
if [ "$SAMPLE_FLAG" = "--sample" ]; then
    if [ -f "data/financebench_queries_sample_tagged.json" ]; then
        QUERIES="data/financebench_queries_sample_tagged.json"
    else
        echo "WARN: data/financebench_queries_sample_tagged.json not found. Using data/financebench_queries.json"
        QUERIES="data/financebench_queries.json"
    fi
else
    if [ -f "data/financebench_queries_tagged.json" ]; then
        QUERIES="data/financebench_queries_tagged.json"
    else
        echo "WARN: data/financebench_queries_tagged.json not found. Using data/financebench_queries.json"
        QUERIES="data/financebench_queries.json"
    fi
fi

echo "=========================================================="
echo "   Parallel benchmark — paper Table 1 matrix (up to 12 runs)"
echo "   Mode:                 ${SAMPLE_FLAG:-Full Dataset}"
[ -n "$N_COMPANIES" ] && echo "   Sample companies:     ${N_COMPANIES#--n }"
echo "   Python:               $PYTHON_BIN"
echo "   Queries:              $QUERIES"
echo "   Run agentic-on row?:  $RUN_AGENTIC_ON_BASELINES"
echo "   Result dir:           data/results/$RAG_BENCHMARK_TIMESTAMP/"
echo "   Logs:                 $LOG_DIR/"
echo "=========================================================="

echo "Step -1: Python/Dependency preflight..."
if ! "$PYTHON_BIN" - <<'PY'
import importlib
import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
importlib.import_module("loguru")
importlib.import_module("typing_extensions")
from models.hoprag.hoprag_adapter import HopRAGAdapter  # noqa: F401
from models.ms_graphrag.ms_adapter import MSGraphRAGAdapter  # noqa: F401
print("Dependency preflight: OK")
PY
then
    echo "ERROR: Python preflight failed. Ensure this script runs with the project .venv."
    exit 1
fi

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
wait_for_server "http://localhost:28010/v1/models" "Gen2 (port 28010)" || echo "   (optional — VLLMClient falls back to gen only)"
wait_for_server "http://localhost:18082/v1/models" "Embed"
wait_for_server "http://localhost:18083/health"    "Rerank"

run_task() {
    local name=$1 cmd=$2
    local log="$LOG_DIR/${name}.log"
    echo "  [STARTED] $name -> $log"
    if eval "$cmd" > "$log" 2>&1; then
        echo "  [COMPLETED] $name"
    else
        echo "  [FAILED] $name"
        echo "$name" >> "$FAIL_MARKER"
    fi
}

# ---------- 1. Baselines (agentic off) ----------
run_task "1_naive" \
    "\"$PYTHON_BIN\" main.py --mode benchmark --strategy naive       --corpus-tag naive_full        --agentic off $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &
run_task "2_hoprag" \
    "\"$PYTHON_BIN\" main.py --mode benchmark --strategy hoprag      --corpus-tag hoprag_full       --agentic off $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &
run_task "3_ms_graphrag" \
    "\"$PYTHON_BIN\" main.py --mode benchmark --strategy ms_graphrag --corpus-tag ms_graphrag_full  --agentic off $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &

# ---------- 2. HypoReflect (agentic off — paper §4.4 reference) ----------
run_task "4_hyporeflect_off" \
    "RAG_ABLATION_TABLE=True RAG_ABLATION_CHUNKING=True RAG_ABLATION_SUMMARY=True RAG_ENABLE_REFLECTION=False \"$PYTHON_BIN\" main.py --mode benchmark --strategy hyporeflect --corpus-tag hyporeflect_full --agentic off $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &

# ---------- 3. HypoReflect main (agentic on) + 3 ablations ----------
run_task "5_hyporeflect_full" \
    "RAG_ABLATION_TABLE=True RAG_ABLATION_CHUNKING=True RAG_ABLATION_SUMMARY=True RAG_ENABLE_REFLECTION=True  \"$PYTHON_BIN\" main.py --mode benchmark --strategy hyporeflect --corpus-tag hyporeflect_full        --agentic on  $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &
run_task "6_hyporeflect_no_table" \
    "RAG_ABLATION_TABLE=False RAG_ENABLE_REFLECTION=True  \"$PYTHON_BIN\" main.py --mode benchmark --strategy hyporeflect --corpus-tag hyporeflect_no_table    --agentic on  $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &
run_task "7_hyporeflect_no_chunk" \
    "RAG_ABLATION_CHUNKING=False RAG_ENABLE_REFLECTION=True  \"$PYTHON_BIN\" main.py --mode benchmark --strategy hyporeflect --corpus-tag hyporeflect_no_chunk    --agentic on  $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &
run_task "8_hyporeflect_no_summary" \
    "RAG_ABLATION_SUMMARY=False RAG_ENABLE_REFLECTION=True  \"$PYTHON_BIN\" main.py --mode benchmark --strategy hyporeflect --corpus-tag hyporeflect_no_summary  --agentic on  $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &

# ---------- 4. Optional: shared agentic-on applied to baselines & all-local hyporeflect ----------
if [ "${RUN_AGENTIC_ON_BASELINES,,}" = "true" ]; then
    run_task "9_naive_agentic_on" \
        "REFLECTION_MODEL= REFINEMENT_MODEL= \"$PYTHON_BIN\" main.py --mode benchmark --strategy naive       --corpus-tag naive_full       --agentic on  $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &
    run_task "10_hoprag_agentic_on" \
        "REFLECTION_MODEL= REFINEMENT_MODEL= \"$PYTHON_BIN\" main.py --mode benchmark --strategy hoprag      --corpus-tag hoprag_full      --agentic on  $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &
    run_task "11_ms_graphrag_agentic_on" \
        "REFLECTION_MODEL= REFINEMENT_MODEL= \"$PYTHON_BIN\" main.py --mode benchmark --strategy ms_graphrag --corpus-tag ms_graphrag_full --agentic on  $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &
    run_task "12_hyporeflect_local" \
        "REFLECTION_MODEL= REFINEMENT_MODEL= RAG_ENABLE_REFLECTION=True \"$PYTHON_BIN\" main.py --mode benchmark --strategy hyporeflect --corpus-tag hyporeflect_full --agentic on $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &
fi

echo
echo "Waiting for all benchmark tasks to finish..."
wait

RUN_DIR="data/results/$RAG_BENCHMARK_TIMESTAMP"
if [ -d "$RUN_DIR" ]; then
    echo
    echo "[Step] Generating run report for $RUN_DIR ..."
    if [ "${RUN_AGENTIC_ON_BASELINES,,}" = "true" ]; then
        "$PYTHON_BIN" tools/benchmark_report.py generate --run-dir "$RUN_DIR" --coverage-profile parallel_all_agentic || true
    else
        "$PYTHON_BIN" tools/benchmark_report.py generate --run-dir "$RUN_DIR" --coverage-profile parallel_all || true
    fi
fi

if [ -s "$FAIL_MARKER" ]; then
    echo
    echo "Failed benchmark tasks:"
    sort -u "$FAIL_MARKER" | sed 's/^/  - /'
    echo "Check logs in $LOG_DIR/"
    exit 1
fi
echo
echo "All benchmark tasks done. Results in $RUN_DIR/"
