#!/bin/bash
#
# run_all_benchmark_parallel.sh - 모든 케이스(교차 실험 포함)에 대해 벤치마크를 병렬로 실행
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

# 기본 설정
SAMPLE_FLAG="--sample"
N_COMPANIES=""
LOG_DIR="logs/benchmark_parallel"
mkdir -p "$LOG_DIR"
FAIL_MARKER="$LOG_DIR/.failed_tasks"
: > "$FAIL_MARKER"
RUN_AGENTIC_MATRIX="${RAG_RUN_AGENTIC_MATRIX:-True}"

# Retrieval tuning defaults (override via env when needed)
export NEO4J_FULLTEXT_ANALYZER="${NEO4J_FULLTEXT_ANALYZER:-english}"
export RAG_ENABLE_QUERY_REWRITE="${RAG_ENABLE_QUERY_REWRITE:-True}"
export RAG_QUERY_REWRITE_COUNT="${RAG_QUERY_REWRITE_COUNT:-2}"
export RAG_QUERY_REWRITE_WEIGHT="${RAG_QUERY_REWRITE_WEIGHT:-0.85}"
export RAG_META_BOOST_WEIGHT="${RAG_META_BOOST_WEIGHT:-0.35}"
export RAG_BOILERPLATE_PENALTY_WEIGHT="${RAG_BOILERPLATE_PENALTY_WEIGHT:-0.25}"

# [TIMESTAMP] Create a unified result directory for this parallel run
export RAG_BENCHMARK_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 인자 처리
while [ $# -gt 0 ]; do
    case $1 in
        --full) SAMPLE_FLAG=""; shift ;;
        --n) N_COMPANIES="--n $2"; shift 2 ;;
        --no-agentic) RUN_AGENTIC_MATRIX="False"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -n "$N_COMPANIES" ] && [ -z "$SAMPLE_FLAG" ]; then
    echo "INFO: --n provided with --full. Forcing sample subset mode."
    SAMPLE_FLAG="--sample"
fi

# 쿼리 파일 자동 선택 (fallback 지원)
if [ "$SAMPLE_FLAG" = "--sample" ]; then
    if [ -f "data/financebench_queries_sample_tagged.json" ]; then
        QUERIES="data/financebench_queries_sample_tagged.json"
    else
        echo "WARN: data/financebench_queries_sample_tagged.json not found. Falling back to data/financebench_queries.json"
        QUERIES="data/financebench_queries.json"
    fi
else
    if [ -f "data/financebench_queries_tagged.json" ]; then
        QUERIES="data/financebench_queries_tagged.json"
    else
        echo "WARN: data/financebench_queries_tagged.json not found. Falling back to data/financebench_queries.json"
        QUERIES="data/financebench_queries.json"
    fi
fi

echo "=========================================================="
echo "   Starting Universal Parallel Benchmarking"
echo "   Mode: ${SAMPLE_FLAG:-Full Dataset}"
if [ -n "$N_COMPANIES" ]; then
    echo "   Sample companies: ${N_COMPANIES#--n }"
fi
echo "   Python: $PYTHON_BIN"
echo "   Retrieval tune: analyzer=${NEO4J_FULLTEXT_ANALYZER}, rewrite=${RAG_ENABLE_QUERY_REWRITE} (count=${RAG_QUERY_REWRITE_COUNT})"
echo "   Agentic matrix (naive/hoprag/ms/hyporeflect): ${RUN_AGENTIC_MATRIX}"
echo "   Logs: $LOG_DIR/"
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

# 0. Start only required servers once (OCR server not needed for benchmark)
echo "Step 0: Pre-starting required services (neo4j/gen/embed/rerank)..."
./run_servers.sh neo4j
./run_servers.sh gen
./run_servers.sh embed
./run_servers.sh rerank

# Function to wait for server
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

wait_for_server "http://localhost:7474" "Neo4j"
wait_for_server "http://localhost:28000/v1/models" "Gen Server"
wait_for_server "http://localhost:18082/v1/models" "Embed Service"
wait_for_server "http://localhost:18083/health" "Rerank Service"

run_task() {
    local name=$1
    local cmd=$2
    local log_file="$LOG_DIR/${name}.log"
    echo "  [STARTED] $name -> $log_file"
    eval "$cmd" > "$log_file" 2>&1
    if [ $? -eq 0 ]; then
        echo "  [COMPLETED] $name"
    else
        echo "  [FAILED] $name"
        echo "$name" >> "$FAIL_MARKER"
    fi
}

# 1. Base & Full Strategies
run_task "naive" "RAG_ABLATION_TABLE=True RAG_ABLATION_CHUNKING=True RAG_ABLATION_SUMMARY=True RAG_ENABLE_REFLECTION=True \"$PYTHON_BIN\" main.py --mode benchmark --strategy naive --corpus-tag naive --agentic off $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &
run_task "hoprag_full" "RAG_ABLATION_TABLE=True RAG_ABLATION_CHUNKING=True RAG_ABLATION_SUMMARY=True RAG_ENABLE_REFLECTION=True \"$PYTHON_BIN\" main.py --mode benchmark --strategy hoprag --corpus-tag hoprag_full --agentic off $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &
run_task "ms_graphrag_full" "RAG_ABLATION_TABLE=True RAG_ABLATION_CHUNKING=True RAG_ABLATION_SUMMARY=True RAG_ENABLE_REFLECTION=True \"$PYTHON_BIN\" main.py --mode benchmark --strategy ms_graphrag --corpus-tag ms_graphrag_full --agentic off $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &
run_task "hyporeflect_full" "RAG_ABLATION_TABLE=True RAG_ABLATION_CHUNKING=True RAG_ABLATION_SUMMARY=True RAG_ENABLE_REFLECTION=True \"$PYTHON_BIN\" main.py --mode benchmark --strategy hyporeflect --corpus-tag full --agentic off $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &

# 2. Naive Ablations (Preprocessing Variants)
run_task "naive_no_table" "RAG_ABLATION_TABLE=False RAG_ABLATION_CHUNKING=True RAG_ABLATION_SUMMARY=True RAG_ENABLE_REFLECTION=True \"$PYTHON_BIN\" main.py --mode benchmark --strategy naive --corpus-tag naive_no_table --agentic off $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &
run_task "naive_no_chunk" "RAG_ABLATION_TABLE=True RAG_ABLATION_CHUNKING=False RAG_ABLATION_SUMMARY=True RAG_ENABLE_REFLECTION=True \"$PYTHON_BIN\" main.py --mode benchmark --strategy naive --corpus-tag naive_no_chunk --agentic off $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &
run_task "naive_no_summary" "RAG_ABLATION_TABLE=True RAG_ABLATION_CHUNKING=True RAG_ABLATION_SUMMARY=False RAG_ENABLE_REFLECTION=True \"$PYTHON_BIN\" main.py --mode benchmark --strategy naive --corpus-tag naive_no_summary --agentic off $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &

# 3. HypoReflect Ablations
run_task "hyporeflect_no_table" "RAG_ABLATION_TABLE=False RAG_ABLATION_CHUNKING=True RAG_ABLATION_SUMMARY=True RAG_ENABLE_REFLECTION=True \"$PYTHON_BIN\" main.py --mode benchmark --strategy hyporeflect --corpus-tag no_table --agentic off $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &
run_task "hyporeflect_no_chunk" "RAG_ABLATION_TABLE=True RAG_ABLATION_CHUNKING=False RAG_ABLATION_SUMMARY=True RAG_ENABLE_REFLECTION=True \"$PYTHON_BIN\" main.py --mode benchmark --strategy hyporeflect --corpus-tag no_chunk --agentic off $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &
run_task "hyporeflect_no_sum" "RAG_ABLATION_TABLE=True RAG_ABLATION_CHUNKING=True RAG_ABLATION_SUMMARY=False RAG_ENABLE_REFLECTION=True \"$PYTHON_BIN\" main.py --mode benchmark --strategy hyporeflect --corpus-tag no_summary --agentic off $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &

# 4. Cross-Ablation: HopRAG
run_task "hoprag_no_table" "RAG_ABLATION_TABLE=False RAG_ABLATION_CHUNKING=True RAG_ABLATION_SUMMARY=True RAG_ENABLE_REFLECTION=True \"$PYTHON_BIN\" main.py --mode benchmark --strategy hoprag --corpus-tag hoprag_no_table --agentic off $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &
run_task "hoprag_no_chunk" "RAG_ABLATION_TABLE=True RAG_ABLATION_CHUNKING=False RAG_ABLATION_SUMMARY=True RAG_ENABLE_REFLECTION=True \"$PYTHON_BIN\" main.py --mode benchmark --strategy hoprag --corpus-tag hoprag_no_chunk --agentic off $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &
run_task "hoprag_no_summary" "RAG_ABLATION_TABLE=True RAG_ABLATION_CHUNKING=True RAG_ABLATION_SUMMARY=False RAG_ENABLE_REFLECTION=True \"$PYTHON_BIN\" main.py --mode benchmark --strategy hoprag --corpus-tag hoprag_no_summary --agentic off $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &

# 5. Cross-Ablation: MS-GraphRAG
run_task "ms_graphrag_no_table" "RAG_ABLATION_TABLE=False RAG_ABLATION_CHUNKING=True RAG_ABLATION_SUMMARY=True RAG_ENABLE_REFLECTION=True \"$PYTHON_BIN\" main.py --mode benchmark --strategy ms_graphrag --corpus-tag ms_graphrag_no_table --agentic off $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &
run_task "ms_graphrag_no_chunk" "RAG_ABLATION_TABLE=True RAG_ABLATION_CHUNKING=False RAG_ABLATION_SUMMARY=True RAG_ENABLE_REFLECTION=True \"$PYTHON_BIN\" main.py --mode benchmark --strategy ms_graphrag --corpus-tag ms_graphrag_no_chunk --agentic off $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &
run_task "ms_graphrag_no_summary" "RAG_ABLATION_TABLE=True RAG_ABLATION_CHUNKING=True RAG_ABLATION_SUMMARY=False RAG_ENABLE_REFLECTION=True \"$PYTHON_BIN\" main.py --mode benchmark --strategy ms_graphrag --corpus-tag ms_graphrag_no_summary --agentic off $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &

if [ "${RUN_AGENTIC_MATRIX,,}" = "true" ]; then
    # 6. Agentic ON: Naive
    run_task "naive_agentic_on" "RAG_AGENTIC_PIPELINE=full RAG_ABLATION_TABLE=True RAG_ABLATION_CHUNKING=True RAG_ABLATION_SUMMARY=True RAG_ENABLE_REFLECTION=True \"$PYTHON_BIN\" main.py --mode benchmark --strategy naive --corpus-tag naive --agentic on $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &
    run_task "naive_no_table_agentic_on" "RAG_AGENTIC_PIPELINE=full RAG_ABLATION_TABLE=False RAG_ABLATION_CHUNKING=True RAG_ABLATION_SUMMARY=True RAG_ENABLE_REFLECTION=True \"$PYTHON_BIN\" main.py --mode benchmark --strategy naive --corpus-tag naive_no_table --agentic on $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &
    run_task "naive_no_chunk_agentic_on" "RAG_AGENTIC_PIPELINE=full RAG_ABLATION_TABLE=True RAG_ABLATION_CHUNKING=False RAG_ABLATION_SUMMARY=True RAG_ENABLE_REFLECTION=True \"$PYTHON_BIN\" main.py --mode benchmark --strategy naive --corpus-tag naive_no_chunk --agentic on $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &
    run_task "naive_no_summary_agentic_on" "RAG_AGENTIC_PIPELINE=full RAG_ABLATION_TABLE=True RAG_ABLATION_CHUNKING=True RAG_ABLATION_SUMMARY=False RAG_ENABLE_REFLECTION=True \"$PYTHON_BIN\" main.py --mode benchmark --strategy naive --corpus-tag naive_no_summary --agentic on $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &

    # 7. Agentic ON: HopRAG
    run_task "hoprag_full_agentic_on" "RAG_AGENTIC_PIPELINE=full RAG_ABLATION_TABLE=True RAG_ABLATION_CHUNKING=True RAG_ABLATION_SUMMARY=True RAG_ENABLE_REFLECTION=True \"$PYTHON_BIN\" main.py --mode benchmark --strategy hoprag --corpus-tag hoprag_full --agentic on $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &
    run_task "hoprag_no_table_agentic_on" "RAG_AGENTIC_PIPELINE=full RAG_ABLATION_TABLE=False RAG_ABLATION_CHUNKING=True RAG_ABLATION_SUMMARY=True RAG_ENABLE_REFLECTION=True \"$PYTHON_BIN\" main.py --mode benchmark --strategy hoprag --corpus-tag hoprag_no_table --agentic on $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &
    run_task "hoprag_no_chunk_agentic_on" "RAG_AGENTIC_PIPELINE=full RAG_ABLATION_TABLE=True RAG_ABLATION_CHUNKING=False RAG_ABLATION_SUMMARY=True RAG_ENABLE_REFLECTION=True \"$PYTHON_BIN\" main.py --mode benchmark --strategy hoprag --corpus-tag hoprag_no_chunk --agentic on $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &
    run_task "hoprag_no_summary_agentic_on" "RAG_AGENTIC_PIPELINE=full RAG_ABLATION_TABLE=True RAG_ABLATION_CHUNKING=True RAG_ABLATION_SUMMARY=False RAG_ENABLE_REFLECTION=True \"$PYTHON_BIN\" main.py --mode benchmark --strategy hoprag --corpus-tag hoprag_no_summary --agentic on $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &

    # 8. Agentic ON: MS-GraphRAG
    run_task "ms_graphrag_full_agentic_on" "RAG_AGENTIC_PIPELINE=full RAG_ABLATION_TABLE=True RAG_ABLATION_CHUNKING=True RAG_ABLATION_SUMMARY=True RAG_ENABLE_REFLECTION=True \"$PYTHON_BIN\" main.py --mode benchmark --strategy ms_graphrag --corpus-tag ms_graphrag_full --agentic on $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &
    run_task "ms_graphrag_no_table_agentic_on" "RAG_AGENTIC_PIPELINE=full RAG_ABLATION_TABLE=False RAG_ABLATION_CHUNKING=True RAG_ABLATION_SUMMARY=True RAG_ENABLE_REFLECTION=True \"$PYTHON_BIN\" main.py --mode benchmark --strategy ms_graphrag --corpus-tag ms_graphrag_no_table --agentic on $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &
    run_task "ms_graphrag_no_chunk_agentic_on" "RAG_AGENTIC_PIPELINE=full RAG_ABLATION_TABLE=True RAG_ABLATION_CHUNKING=False RAG_ABLATION_SUMMARY=True RAG_ENABLE_REFLECTION=True \"$PYTHON_BIN\" main.py --mode benchmark --strategy ms_graphrag --corpus-tag ms_graphrag_no_chunk --agentic on $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &
    run_task "ms_graphrag_no_summary_agentic_on" "RAG_AGENTIC_PIPELINE=full RAG_ABLATION_TABLE=True RAG_ABLATION_CHUNKING=True RAG_ABLATION_SUMMARY=False RAG_ENABLE_REFLECTION=True \"$PYTHON_BIN\" main.py --mode benchmark --strategy ms_graphrag --corpus-tag ms_graphrag_no_summary --agentic on $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &

    # 9. Agentic ON: HypoReflect
    run_task "hyporeflect_full_agentic_on" "RAG_ABLATION_TABLE=True RAG_ABLATION_CHUNKING=True RAG_ABLATION_SUMMARY=True RAG_ENABLE_REFLECTION=True \"$PYTHON_BIN\" main.py --mode benchmark --strategy hyporeflect --corpus-tag full --agentic on $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &
    run_task "hyporeflect_no_table_agentic_on" "RAG_ABLATION_TABLE=False RAG_ABLATION_CHUNKING=True RAG_ABLATION_SUMMARY=True RAG_ENABLE_REFLECTION=True \"$PYTHON_BIN\" main.py --mode benchmark --strategy hyporeflect --corpus-tag no_table --agentic on $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &
    run_task "hyporeflect_no_chunk_agentic_on" "RAG_ABLATION_TABLE=True RAG_ABLATION_CHUNKING=False RAG_ABLATION_SUMMARY=True RAG_ENABLE_REFLECTION=True \"$PYTHON_BIN\" main.py --mode benchmark --strategy hyporeflect --corpus-tag no_chunk --agentic on $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &
    run_task "hyporeflect_no_summary_agentic_on" "RAG_ABLATION_TABLE=True RAG_ABLATION_CHUNKING=True RAG_ABLATION_SUMMARY=False RAG_ENABLE_REFLECTION=True \"$PYTHON_BIN\" main.py --mode benchmark --strategy hyporeflect --corpus-tag no_summary --agentic on $SAMPLE_FLAG $N_COMPANIES --queries_file $QUERIES" &
fi

echo -e "\nWaiting for all tasks to finish..."
wait

RUN_DIR="data/results/$RAG_BENCHMARK_TIMESTAMP"
if [ -d "$RUN_DIR" ]; then
    echo ""
    echo "[Step] Generating run report for $RUN_DIR ..."
    if [ "${RUN_AGENTIC_MATRIX,,}" = "true" ]; then
        "$PYTHON_BIN" tools/benchmark_report.py generate --run-dir "$RUN_DIR" --coverage-profile parallel_all_agentic || true
    else
        "$PYTHON_BIN" tools/benchmark_report.py generate --run-dir "$RUN_DIR" --coverage-profile parallel_all || true
    fi
fi

if [ -s "$FAIL_MARKER" ]; then
    echo ""
    echo "Failed benchmark tasks:"
    sort -u "$FAIL_MARKER" | sed 's/^/  - /'
    echo "Check logs in $LOG_DIR/"
    exit 1
fi
echo -e "\nAll benchmarking tasks completed! Results in data/results/"
