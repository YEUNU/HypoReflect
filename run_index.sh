#!/bin/bash
#
# run_index.sh - Index text files into Neo4j graph
#

set -e

# [환경 설정]
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export CUDA_VISIBLE_DEVICES=0
export NEO4J_VECTOR_DIMENSIONS="${NEO4J_VECTOR_DIMENSIONS:-1024}"
export MAX_EMBEDDING_LENGTH="${MAX_EMBEDDING_LENGTH:-16384}"
export NEO4J_FULLTEXT_ANALYZER="${NEO4J_FULLTEXT_ANALYZER:-english}"
export RAG_RECREATE_TEXT_INDEX="${RAG_RECREATE_TEXT_INDEX:-False}"
export RAG_ENABLE_QUERY_REWRITE="${RAG_ENABLE_QUERY_REWRITE:-True}"
export RAG_QUERY_REWRITE_COUNT="${RAG_QUERY_REWRITE_COUNT:-2}"
export RAG_QUERY_REWRITE_WEIGHT="${RAG_QUERY_REWRITE_WEIGHT:-0.85}"
export RAG_META_BOOST_WEIGHT="${RAG_META_BOOST_WEIGHT:-0.35}"
export RAG_BOILERPLATE_PENALTY_WEIGHT="${RAG_BOILERPLATE_PENALTY_WEIGHT:-0.25}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

. "$SCRIPT_DIR/scripts/lib.sh"

# Default values
MODEL="hyporeflect"
LLM="local"
DATASET=""
N_COMPANIES=""
RAW_OCR=""
CLEAR_GRAPH=""
CORPUS_TAG=""
SAVE_INTERMEDIATE=""
SAMPLE=""
SAVE_TO=""
OCR=""

SKIP_SERVER=""

# Parse arguments
while [ $# -gt 0 ]; do
    case $1 in
        --dataset) DATASET="--dataset $2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --llm) LLM="$2"; shift 2 ;;
        --n) N_COMPANIES="--n $2"; shift 2 ;;
        --raw-ocr) RAW_OCR="--raw-ocr"; shift 1 ;;
        --clear-graph) CLEAR_GRAPH="--clear-graph"; shift 1 ;;
        --corpus-tag) CORPUS_TAG="--corpus-tag $2"; shift 2 ;;
        --save-intermediate) SAVE_INTERMEDIATE="--save-intermediate"; shift 1 ;;
        --sample) SAMPLE="--sample"; shift 1 ;;
        --save-to) SAVE_TO="--save-to $2"; shift 2 ;;
        --ocr) OCR="--ocr"; shift 1 ;;
        --skip-server) SKIP_SERVER="true"; shift 1 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "========================================="
echo "     Indexing Pre-flight Check           "
echo "========================================="
echo "Retrieval Tune: analyzer=${NEO4J_FULLTEXT_ANALYZER}, recreate_text_index=${RAG_RECREATE_TEXT_INDEX}, rewrite=${RAG_ENABLE_QUERY_REWRITE} (count=${RAG_QUERY_REWRITE_COUNT})"

# [0] Check Arch (skip if running all models)
if [ "$MODEL" != "all" ] && [ ! -d "models/$MODEL" ]; then
    echo "❌ Arch model '$MODEL' not found in models/ folder."
    exit 1
fi

if [ "$SKIP_SERVER" != "true" ]; then
    echo "Step 1: Checking indexing services..."

    # Start Neo4j
    ./run_servers.sh neo4j
    if ! wait_for_server "http://localhost:7474" "Neo4j"; then
        echo "Fatal: Neo4j failed." >&2
        exit 1
    fi

    # Start Generation Server
    ./run_servers.sh gen
    if ! wait_for_server "http://localhost:28000/v1/models" "Generation Model"; then
        echo "Fatal: Generation model failed." >&2
        exit 1
    fi

    # Start Embedding Service
    ./run_servers.sh embed
    if ! wait_for_server "http://localhost:18082/v1/models" "Embedding Model"; then
        echo "Fatal: Embedding service failed." >&2
        exit 1
    fi

    # Start Reranker Service (Required for Rank-based Edge Pruning)
    ./run_servers.sh rerank
    if ! wait_for_server "http://localhost:18083/health" "Reranker Model"; then
        echo "Fatal: Reranker service failed." >&2
        exit 1
    fi
else
    echo "Step 1: Skipping server startup (Requested by caller)"
fi

# [3] Run indexing
echo ""
echo "[Step] Running indexing..."

if [ "$MODEL" = "all" ]; then
    echo "🚀 Running ALL models in parallel..."
    MODELS=("hyporeflect" "hoprag" "naive" "ms_graphrag")
    PIDS=()
    
    for m in "${MODELS[@]}"; do
        echo "  Starting $m..."
        python main.py --mode index $DATASET --strategy "$m" --model "$LLM" $N_COMPANIES $RAW_OCR $CLEAR_GRAPH $CORPUS_TAG $SAVE_INTERMEDIATE $SAMPLE $SAVE_TO $OCR > "logs/index_${m}.log" 2>&1 &
        PIDS+=($!)
    done
    
    echo "  Waiting for all models to complete..."
    FAILED=0
    for i in "${!PIDS[@]}"; do
        if wait ${PIDS[$i]}; then
            echo "  ✅ ${MODELS[$i]} completed successfully"
        else
            echo "  ❌ ${MODELS[$i]} failed"
            FAILED=1
        fi
    done
    
    if [ $FAILED -eq 1 ]; then
        echo "Some models failed. Check logs/index_*.log for details."
        exit 1
    fi
    echo "✅ All models indexed successfully!"
else
    python main.py --mode index $DATASET --strategy "$MODEL" --model "$LLM" $N_COMPANIES $RAW_OCR $CLEAR_GRAPH $CORPUS_TAG $SAVE_INTERMEDIATE $SAMPLE $SAVE_TO $OCR
fi
