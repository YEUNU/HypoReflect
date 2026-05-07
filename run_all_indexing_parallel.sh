#!/bin/bash
#
# run_all_indexing_parallel.sh - 모든 케이스(교차 실험 포함)를 병렬로 실행
#

set -e

# 기본 설정
SAMPLE_FLAG="--sample --ocr"
N_COMPANIES=""
LOG_DIR="logs/indexing_parallel"
mkdir -p "$LOG_DIR"

# Retrieval tuning defaults (override via env when needed)
export NEO4J_FULLTEXT_ANALYZER="${NEO4J_FULLTEXT_ANALYZER:-english}"
export RAG_RECREATE_TEXT_INDEX="${RAG_RECREATE_TEXT_INDEX:-False}"
export RAG_ENABLE_QUERY_REWRITE="${RAG_ENABLE_QUERY_REWRITE:-True}"
export RAG_QUERY_REWRITE_COUNT="${RAG_QUERY_REWRITE_COUNT:-2}"
export RAG_QUERY_REWRITE_WEIGHT="${RAG_QUERY_REWRITE_WEIGHT:-0.85}"
export RAG_META_BOOST_WEIGHT="${RAG_META_BOOST_WEIGHT:-0.35}"
export RAG_BOILERPLATE_PENALTY_WEIGHT="${RAG_BOILERPLATE_PENALTY_WEIGHT:-0.25}"

# 인자 처리
while [ $# -gt 0 ]; do
    case $1 in
        --full)
            SAMPLE_FLAG=""
            shift
            ;;
        --n) N_COMPANIES="--n $2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -n "$N_COMPANIES" ] && [ -z "$SAMPLE_FLAG" ]; then
    echo "INFO: --n provided with --full. Forcing sample subset mode."
    SAMPLE_FLAG="--sample --ocr"
fi

echo "=========================================================="
echo "   Starting Universal Parallel Indexing (Cross-Ablation)"
echo "   Mode: ${SAMPLE_FLAG:-Full Dataset}"
if [ -n "$N_COMPANIES" ]; then
    echo "   Sample companies: ${N_COMPANIES#--n }"
fi
echo "   Retrieval tune: analyzer=${NEO4J_FULLTEXT_ANALYZER}, recreate_text_index=${RAG_RECREATE_TEXT_INDEX}, rewrite=${RAG_ENABLE_QUERY_REWRITE} (count=${RAG_QUERY_REWRITE_COUNT})"
if [ "${RAG_RECREATE_TEXT_INDEX}" = "True" ]; then
    echo "   WARN: recreate_text_index=True with parallel indexing may increase index churn."
fi
echo "   Logs: $LOG_DIR/"
echo "=========================================================="

# 0. Start only required servers once (OCR server not needed for indexing from text corpus)
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
    # Add --skip-server to the command
    eval "$cmd --skip-server" > "$log_file" 2>&1
    if [ $? -eq 0 ]; then
        echo "  [COMPLETED] $name"
    else
        echo "  [FAILED] $name (Check $log_file)"
    fi
}

# 1. Baseline & Full Versions
run_task "naive" "RAG_ABLATION_TABLE=True RAG_ABLATION_CHUNKING=True RAG_ABLATION_SUMMARY=True ./run_index.sh --model naive --corpus-tag naive $SAMPLE_FLAG $N_COMPANIES" &
run_task "hoprag_full" "./run_index.sh --model hoprag --corpus-tag hoprag_full $SAMPLE_FLAG $N_COMPANIES" &
run_task "ms_graphrag_full" "./run_index.sh --model ms_graphrag --corpus-tag ms_graphrag_full $SAMPLE_FLAG $N_COMPANIES" &
run_task "hyporeflect_full" "./run_index.sh --model hyporeflect --corpus-tag full $SAMPLE_FLAG $N_COMPANIES" &

# 2. Naive Ablations (Preprocessing Variants)
run_task "naive_no_table" "RAG_ABLATION_TABLE=False RAG_ABLATION_CHUNKING=True RAG_ABLATION_SUMMARY=True ./run_index.sh --model naive --corpus-tag naive_no_table $SAMPLE_FLAG $N_COMPANIES" &
run_task "naive_no_chunk" "RAG_ABLATION_TABLE=True RAG_ABLATION_CHUNKING=False RAG_ABLATION_SUMMARY=True ./run_index.sh --model naive --corpus-tag naive_no_chunk $SAMPLE_FLAG $N_COMPANIES" &
run_task "naive_no_summary" "RAG_ABLATION_TABLE=True RAG_ABLATION_CHUNKING=True RAG_ABLATION_SUMMARY=False ./run_index.sh --model naive --corpus-tag naive_no_summary $SAMPLE_FLAG $N_COMPANIES" &

# 3. HypoReflect Ablations
run_task "hyporeflect_no_table" "RAG_ABLATION_TABLE=False ./run_index.sh --model hyporeflect --corpus-tag no_table $SAMPLE_FLAG $N_COMPANIES" &
run_task "hyporeflect_no_chunk" "RAG_ABLATION_CHUNKING=False ./run_index.sh --model hyporeflect --corpus-tag no_chunk $SAMPLE_FLAG $N_COMPANIES" &
run_task "hyporeflect_no_summary" "RAG_ABLATION_SUMMARY=False ./run_index.sh --model hyporeflect --corpus-tag no_summary $SAMPLE_FLAG $N_COMPANIES" &

# 4. Cross-Ablation: HopRAG
run_task "hoprag_no_table" "RAG_ABLATION_TABLE=False ./run_index.sh --model hoprag --corpus-tag hoprag_no_table $SAMPLE_FLAG $N_COMPANIES" &
run_task "hoprag_no_chunk" "RAG_ABLATION_CHUNKING=False ./run_index.sh --model hoprag --corpus-tag hoprag_no_chunk $SAMPLE_FLAG $N_COMPANIES" &
run_task "hoprag_no_summary" "RAG_ABLATION_SUMMARY=False ./run_index.sh --model hoprag --corpus-tag hoprag_no_summary $SAMPLE_FLAG $N_COMPANIES" &

# 5. Cross-Ablation: MS-GraphRAG
run_task "ms_graphrag_no_table" "RAG_ABLATION_TABLE=False ./run_index.sh --model ms_graphrag --corpus-tag ms_graphrag_no_table $SAMPLE_FLAG $N_COMPANIES" &
run_task "ms_graphrag_no_chunk" "RAG_ABLATION_CHUNKING=False ./run_index.sh --model ms_graphrag --corpus-tag ms_graphrag_no_chunk $SAMPLE_FLAG $N_COMPANIES" &
run_task "ms_graphrag_no_summary" "RAG_ABLATION_SUMMARY=False ./run_index.sh --model ms_graphrag --corpus-tag ms_graphrag_no_summary $SAMPLE_FLAG $N_COMPANIES" &

echo -e "\nWaiting for all indexing tasks to finish..."
wait
echo -e "\nAll indexing tasks completed! Check results with 'python scripts/check_indexes.py'"
