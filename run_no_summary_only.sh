#!/bin/bash
#
# run_no_summary_only.sh
# Run only no_summary ablation for HopRAG and MS-GraphRAG.
#

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

MODE="index"
SAMPLE_FLAG="--sample"
OCR_FLAG=""
N_COMPANIES=""
SKIP_SERVER=""
TARGET_MODEL="both"
LOG_DIR="logs/no_summary_only"
mkdir -p "$LOG_DIR"

# Retrieval tuning defaults (override via env when needed)
export NEO4J_FULLTEXT_ANALYZER="${NEO4J_FULLTEXT_ANALYZER:-english}"
export RAG_RECREATE_TEXT_INDEX="${RAG_RECREATE_TEXT_INDEX:-False}"
export RAG_ENABLE_QUERY_REWRITE="${RAG_ENABLE_QUERY_REWRITE:-True}"
export RAG_QUERY_REWRITE_COUNT="${RAG_QUERY_REWRITE_COUNT:-2}"
export RAG_QUERY_REWRITE_WEIGHT="${RAG_QUERY_REWRITE_WEIGHT:-0.85}"
export RAG_META_BOOST_WEIGHT="${RAG_META_BOOST_WEIGHT:-0.35}"
export RAG_BOILERPLATE_PENALTY_WEIGHT="${RAG_BOILERPLATE_PENALTY_WEIGHT:-0.25}"

while [ $# -gt 0 ]; do
    case $1 in
        --mode) MODE="$2"; shift 2 ;;
        --model) TARGET_MODEL="$2"; shift 2 ;;
        --full) SAMPLE_FLAG=""; OCR_FLAG=""; shift ;;
        --n) N_COMPANIES="--n $2"; shift 2 ;;
        --skip-server) SKIP_SERVER="true"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ "$MODE" != "index" ] && [ "$MODE" != "benchmark" ]; then
    echo "Invalid --mode: $MODE (use index|benchmark)"
    exit 1
fi

# Normalize model aliases
case "$TARGET_MODEL" in
    msgraphrag) TARGET_MODEL="ms_graphrag" ;;
    all) TARGET_MODEL="both" ;;
esac

if [ "$TARGET_MODEL" != "both" ] && [ "$TARGET_MODEL" != "hoprag" ] && [ "$TARGET_MODEL" != "ms_graphrag" ]; then
    echo "Invalid --model: $TARGET_MODEL (use hoprag|ms_graphrag|both)"
    exit 1
fi

# Keep behavior aligned with existing scripts:
# if --n is used with --full, force sample mode.
if [ -n "$N_COMPANIES" ] && [ -z "$SAMPLE_FLAG" ]; then
    echo "INFO: --n provided with --full. Forcing sample subset mode."
    SAMPLE_FLAG="--sample"
fi

if [ "$MODE" = "index" ] && [ -n "$SAMPLE_FLAG" ]; then
    OCR_FLAG="--ocr"
fi

if [ "$MODE" = "benchmark" ]; then
    # Unified timestamp for this run
    export RAG_BENCHMARK_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
fi

echo "=========================================================="
echo "   Run no_summary only (HopRAG + MS-GraphRAG)"
echo "   Mode: $MODE"
echo "   Target model: $TARGET_MODEL"
echo "   Python: $PYTHON_BIN"
echo "   Dataset mode: ${SAMPLE_FLAG:-Full Dataset}"
if [ -n "$N_COMPANIES" ]; then
    echo "   Sample companies: ${N_COMPANIES#--n }"
fi
echo "   Logs: $LOG_DIR/"
echo "=========================================================="

wait_for_server() {
    local url=$1
    local name=$2
    local max_attempts=300
    local attempt=0
    echo "Wait for $name ($url)..."
    while [ $attempt -lt $max_attempts ]; do
        if curl -s -o /dev/null -w "%{http_code}" --max-time 2 "$url" | grep -qE "200|401|405"; then
            echo " ✅ $name is Ready!"
            return 0
        fi
        sleep 5
        attempt=$((attempt + 1))
    done
    return 1
}

if [ "$SKIP_SERVER" != "true" ]; then
    echo "Step 0: Pre-starting required services (neo4j/gen/embed/rerank)..."
    ./run_servers.sh neo4j
    ./run_servers.sh gen
    ./run_servers.sh embed
    ./run_servers.sh rerank

    wait_for_server "http://localhost:7474" "Neo4j"
    wait_for_server "http://localhost:28000/v1/models" "Gen Server"
    wait_for_server "http://localhost:18082/v1/models" "Embed Service"
    wait_for_server "http://localhost:18083/health" "Rerank Service"
else
    echo "Step 0: Skipping server startup (Requested by caller)"
fi

if [ "$MODE" = "benchmark" ]; then
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
        echo "ERROR: Python preflight failed."
        exit 1
    fi
fi

run_task() {
    local name=$1
    local cmd=$2
    local log_file="$LOG_DIR/${name}.log"
    echo "  [STARTED] $name -> $log_file"
    eval "$cmd" > "$log_file" 2>&1
    if [ $? -eq 0 ]; then
        echo "  [COMPLETED] $name"
    else
        echo "  [FAILED] $name (Check $log_file)"
    fi
}

if [ "$MODE" = "index" ]; then
    if [ "$TARGET_MODEL" = "both" ] || [ "$TARGET_MODEL" = "hoprag" ]; then
        run_task "hoprag_no_summary" "RAG_ABLATION_SUMMARY=False ./run_index.sh --model hoprag --corpus-tag hoprag_no_summary $SAMPLE_FLAG $OCR_FLAG $N_COMPANIES --skip-server" &
    fi
    if [ "$TARGET_MODEL" = "both" ] || [ "$TARGET_MODEL" = "ms_graphrag" ]; then
        run_task "ms_graphrag_no_summary" "RAG_ABLATION_SUMMARY=False ./run_index.sh --model ms_graphrag --corpus-tag ms_graphrag_no_summary $SAMPLE_FLAG $OCR_FLAG $N_COMPANIES --skip-server" &
    fi
else
    if [ "$SAMPLE_FLAG" = "--sample" ]; then
        if [ -f "data/financebench_queries_sample_tagged.json" ]; then
            QUERIES="data/financebench_queries_sample_tagged.json"
        else
            QUERIES="data/financebench_queries.json"
        fi
    else
        if [ -f "data/financebench_queries_tagged.json" ]; then
            QUERIES="data/financebench_queries_tagged.json"
        else
            QUERIES="data/financebench_queries.json"
        fi
    fi

    if [ "$TARGET_MODEL" = "both" ] || [ "$TARGET_MODEL" = "hoprag" ]; then
        run_task "hoprag_no_summary" "RAG_ABLATION_TABLE=True RAG_ABLATION_CHUNKING=True RAG_ABLATION_SUMMARY=False RAG_ENABLE_REFLECTION=True \"$PYTHON_BIN\" main.py --mode benchmark --strategy hoprag --corpus-tag hoprag_no_summary $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &
    fi
    if [ "$TARGET_MODEL" = "both" ] || [ "$TARGET_MODEL" = "ms_graphrag" ]; then
        run_task "ms_graphrag_no_summary" "RAG_ABLATION_TABLE=True RAG_ABLATION_CHUNKING=True RAG_ABLATION_SUMMARY=False RAG_ENABLE_REFLECTION=True \"$PYTHON_BIN\" main.py --mode benchmark --strategy ms_graphrag --corpus-tag ms_graphrag_no_summary $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &
    fi
fi

echo -e "\nWaiting for all tasks to finish..."
wait
echo -e "\nDone."
