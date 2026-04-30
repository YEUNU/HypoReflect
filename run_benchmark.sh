#!/bin/bash
#
# run_benchmark.sh - Run benchmark evaluation
#

set -e

# [환경 설정]
export VLLM_API_KEY=EMPTY
export CUDA_VISIBLE_DEVICES=0
export NEO4J_FULLTEXT_ANALYZER="${NEO4J_FULLTEXT_ANALYZER:-english}"
export RAG_ENABLE_QUERY_REWRITE="${RAG_ENABLE_QUERY_REWRITE:-True}"
export RAG_QUERY_REWRITE_COUNT="${RAG_QUERY_REWRITE_COUNT:-2}"
export RAG_QUERY_REWRITE_WEIGHT="${RAG_QUERY_REWRITE_WEIGHT:-0.85}"
export RAG_META_BOOST_WEIGHT="${RAG_META_BOOST_WEIGHT:-0.35}"
export RAG_BOILERPLATE_PENALTY_WEIGHT="${RAG_BOILERPLATE_PENALTY_WEIGHT:-0.25}"
export RAG_BENCHMARK_TIMESTAMP="${RAG_BENCHMARK_TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

. "$SCRIPT_DIR/scripts/lib.sh"

PYTHON_BIN="${PYTHON_BIN:-$SCRIPT_DIR/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    else
        PYTHON_BIN="$(command -v python)"
    fi
fi

# Default values
QUERIES_FILE="data/financebench_queries.json"
MODEL="hyporeflect"
LLM="local"
N_COMPANIES=""
RUN_ALL=false
SAMPLE=""
OCR=""
CORPUS_TAG=""
AGENTIC=""

# Parse arguments
while [ $# -gt 0 ]; do
    case $1 in
        --queries) QUERIES_FILE="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --llm) LLM="$2"; shift 2 ;;
        --n) N_COMPANIES="--n $2"; shift 2 ;;
        --all) RUN_ALL=true; shift ;;
        --sample) SAMPLE="--sample"; shift 1 ;;
        --ocr) OCR="--ocr"; shift 1 ;;
        --corpus-tag) CORPUS_TAG="--corpus-tag $2"; shift 2 ;;
        --agentic) AGENTIC="--agentic $2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "========================================="
echo "     Benchmark Pre-flight Check          "
echo "========================================="
echo "Python: $PYTHON_BIN"
echo "Retrieval Tune: analyzer=${NEO4J_FULLTEXT_ANALYZER}, rewrite=${RAG_ENABLE_QUERY_REWRITE} (count=${RAG_QUERY_REWRITE_COUNT})"
if [ -n "$AGENTIC" ]; then
    echo "Agentic mode: ${AGENTIC#--agentic }"
fi

echo "Step 0: Python/Dependency preflight..."
if [ "$MODEL" = "hoprag" ] || [ "$MODEL" = "ms_graphrag" ] || [ "$RUN_ALL" = true ]; then
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

echo "Step 1: Checking benchmark services..."

# Start Neo4j
./run_servers.sh neo4j
if ! wait_for_server "http://localhost:7474" "Neo4j"; then exit 1; fi

# Start Generation Server
./run_servers.sh gen
if ! wait_for_server "http://localhost:28000/v1/models" "Generation Model"; then exit 1; fi

# Start Embedding Service
./run_servers.sh embed
if ! wait_for_server "http://localhost:18082/v1/models" "Embedding Model"; then exit 1; fi

# Start Reranker Service
./run_servers.sh rerank
if ! wait_for_server "http://localhost:18083/health" "Reranker Model"; then exit 1; fi

# [2] Run benchmark
echo ""
echo "[Step] Running benchmark..."
if [ "$RUN_ALL" = true ]; then
    "$PYTHON_BIN" main.py --mode benchmark_all --queries_file "$QUERIES_FILE" --model "$LLM" $N_COMPANIES $SAMPLE $OCR $CORPUS_TAG $AGENTIC
else
    "$PYTHON_BIN" main.py --mode benchmark --queries_file "$QUERIES_FILE" --strategy "$MODEL" --model "$LLM" $N_COMPANIES $SAMPLE $OCR $CORPUS_TAG $AGENTIC
fi

# [3] Generate human-readable run reports
RUN_DIR="data/results/$RAG_BENCHMARK_TIMESTAMP"
if [ ! -d "$RUN_DIR" ]; then
    RUN_DIR="$(ls -1dt data/results/* 2>/dev/null | head -n 1)"
fi
if [ -n "$RUN_DIR" ] && [ -d "$RUN_DIR" ]; then
    echo ""
    echo "[Step] Generating run report for $RUN_DIR ..."
    "$PYTHON_BIN" tools/benchmark_report.py generate --run-dir "$RUN_DIR" || true
fi
