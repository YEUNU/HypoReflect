#!/bin/bash
#
# run_servers.sh - Centralized service manager for HypoReflect
# Usage: ./run_servers.sh {neo4j|gen|ocr|embed|rerank|all}
#

set -e

# [환경 설정]
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

SERVICE=$1

# Optional Java setup (for local/custom Neo4j distributions)
if [ -n "${JAVA_HOME:-}" ]; then
    export PATH="$JAVA_HOME/bin:$PATH"
fi

resolve_neo4j_cmd() {
    if [ -n "${NEO4J_BIN:-}" ] && [ -x "${NEO4J_BIN}" ]; then
        echo "${NEO4J_BIN}"
        return 0
    fi
    if [ -n "${NEO4J_HOME:-}" ] && [ -x "${NEO4J_HOME}/bin/neo4j" ]; then
        echo "${NEO4J_HOME}/bin/neo4j"
        return 0
    fi
    if command -v neo4j >/dev/null 2>&1; then
        command -v neo4j
        return 0
    fi
    return 1
}

curl_with_auth() {
    local url="$1"
    if [ -n "$VLLM_API_KEY" ] && [ "$VLLM_API_KEY" != "EMPTY" ]; then
        curl -s --max-time 1 -H "Authorization: Bearer ${VLLM_API_KEY}" "$url"
    else
        curl -s --max-time 1 "$url"
    fi
}

is_vllm_server_up() {
    local port="$1"
    # Prefer lightweight unauthenticated probe to avoid auth noise.
    curl -s --max-time 1 "http://localhost:${port}/health" > /dev/null 2>&1 && return 0

    # Fallback: authenticated model listing when API key is configured.
    if [ -n "$VLLM_API_KEY" ] && [ "$VLLM_API_KEY" != "EMPTY" ]; then
        curl_with_auth "http://localhost:${port}/v1/models" > /dev/null 2>&1 && return 0
    fi

    is_port_in_use "$port"
}

is_port_in_use() {
    local port="$1"
    fuser "${port}/tcp" > /dev/null 2>&1
}

apply_neo4j_docker_limits() {
    local container_name="$1"
    local neo4j_docker_cpus="${NEO4J_DOCKER_CPUS:-12}"
    local neo4j_docker_cpuset="${NEO4J_DOCKER_CPUSET:-}"
    local update_args=()

    if [ -n "${neo4j_docker_cpus}" ]; then
        update_args+=(--cpus "${neo4j_docker_cpus}")
    fi

    if [ -n "${neo4j_docker_cpuset}" ]; then
        update_args+=(--cpuset-cpus "${neo4j_docker_cpuset}")
    fi

    if [ ${#update_args[@]} -gt 0 ]; then
        docker update "${update_args[@]}" "${container_name}" > /dev/null
    fi

    if [ -n "${neo4j_docker_cpus}" ]; then
        echo "Applied Neo4j Docker CPU limit: ${neo4j_docker_cpus}"
    fi

    if [ -n "${neo4j_docker_cpuset}" ]; then
        echo "Applied Neo4j Docker CPU pinning: ${neo4j_docker_cpuset}"
    fi
}

start_neo4j_docker() {
    if ! command -v docker >/dev/null 2>&1; then
        return 1
    fi

    local container_name="${NEO4J_CONTAINER_NAME:-hyporeflect-neo4j}"
    local neo4j_user="${NEO4J_USER:-neo4j}"
    local neo4j_password="${NEO4J_PASSWORD:-1q2w3e4r}"
    local neo4j_docker_cpus="${NEO4J_DOCKER_CPUS:-12}"
    local neo4j_docker_cpuset="${NEO4J_DOCKER_CPUSET:-}"
    local docker_args=()

    if [ -n "${neo4j_docker_cpus}" ]; then
        docker_args+=(--cpus "${neo4j_docker_cpus}")
    fi

    if [ -n "${neo4j_docker_cpuset}" ]; then
        docker_args+=(--cpuset-cpus "${neo4j_docker_cpuset}")
    fi

    if docker ps --format '{{.Names}}' | grep -Fxq "${container_name}"; then
        apply_neo4j_docker_limits "${container_name}"
        echo "✅ Neo4j Docker container is already UP (${container_name})"
        return 0
    fi

    if docker ps -a --format '{{.Names}}' | grep -Fxq "${container_name}"; then
        echo "Starting Neo4j Docker container (${container_name})..."
        docker start "${container_name}" > /dev/null
        apply_neo4j_docker_limits "${container_name}"
        return 0
    fi

    local neo4j_data_dir="${NEO4J_DATA_DIR:-${SCRIPT_DIR:-.}/neo4j_data}"
    mkdir -p "${neo4j_data_dir}"

    echo "Starting Neo4j Docker container (${container_name})..."
    docker run -d \
        --name "${container_name}" \
        "${docker_args[@]}" \
        -p 7474:7474 \
        -p 7687:7687 \
        -v "${neo4j_data_dir}:/data" \
        -e NEO4J_AUTH="${neo4j_user}/${neo4j_password}" \
        -e NEO4J_server_memory_pagecache_size=4g \
        -e NEO4J_server_memory_heap_initial__size=2g \
        -e NEO4J_server_memory_heap_max__size=4g \
        neo4j:5-community > /dev/null
}

start_neo4j() {
    if ! curl -s --max-time 1 http://localhost:7474 > /dev/null 2>&1; then
        local neo4j_cmd
        if ! neo4j_cmd="$(resolve_neo4j_cmd)"; then
            echo "Neo4j local binary not found. Trying Docker fallback..."
            if start_neo4j_docker; then
                return 0
            fi
            echo "❌ Neo4j not found. Set NEO4J_BIN/NEO4J_HOME, install neo4j on PATH, or install Docker."
            return 1
        fi
        echo "Starting Neo4j..."
        nohup "${neo4j_cmd}" start > neo4j.log 2>&1 &
    else
        echo "✅ Neo4j is already UP"
    fi
}

start_gen() {
    if is_vllm_server_up 28000; then
        echo "✅ Generation Server is already UP"
        return 0
    fi

    if is_port_in_use 28000; then
        echo "✅ Generation Server is already running (port 28000 in use)"
        return 0
    fi

    echo "Starting Generation Server (Port 28000)..."
    CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/vllm serve Qwen/Qwen3-4B-Instruct-2507 \
        --served-model-name generation-model \
        --host 0.0.0.0 \
        --port 28000 \
        --gpu-memory-utilization 0.8 \
        --max-model-len 65536 \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_xml \
        --attention-backend FLASHINFER \
        --trust-remote-code > vllm_gen.log 2>&1 &
}

start_ocr() {
    if is_vllm_server_up 28001; then
        echo "✅ OCR Server is already UP"
        return 0
    fi

    if is_port_in_use 28001; then
        echo "✅ OCR Server is already running (port 28001 in use)"
        return 0
    fi

    echo "Starting OCR Server (Port 28001)..."
    MALLOC_TRIM_THRESHOLD_=100000 \
    CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/vllm serve lightonai/LightOnOCR-1B-1025 \
        --served-model-name ocr-model \
        --host 0.0.0.0 \
        --port 28001 \
        --gpu-memory-utilization 0.4 \
        --max-model-len 8192 \
        --attention-backend FLASHINFER \
        --trust-remote-code \
        --limit-mm-per-prompt '{"image":1}' \
        --mm-processor-cache-gb 0.5 > vllm_ocr.log 2>&1 &
}

start_embed() {
    if is_vllm_server_up 18082; then
        echo "✅ Embedding Server is already UP"
        return 0
    fi

    if is_port_in_use 18082; then
        echo "✅ Embedding Server is already running (port 18082 in use)"
        return 0
    fi

    echo "Starting Embedding Server (Port 18082)..."
    CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/vllm serve Qwen/Qwen3-Embedding-0.6B \
        --served-model-name embedding-model \
        --host 0.0.0.0 \
        --port 18082 \
        --gpu-memory-utilization 0.45 \
        --max-model-len 16384 \
        --attention-backend FLASHINFER \
        --trust-remote-code > embedding.log 2>&1 &
}

start_rerank() {
    if curl -s --max-time 1 http://localhost:18083/health > /dev/null 2>&1; then
        echo "✅ Reranker Service is already UP"
        return 0
    fi

    if is_port_in_use 18083; then
        echo "✅ Reranker Service is already running (port 18083 in use)"
        return 0
    fi

    echo "Starting Reranker Service (Port 18083)..."
    export RERANKER_MODEL_ID="Qwen/Qwen3-Reranker-0.6B"
    export RERANKER_GPU_ID=1
    export RERANKER_GPU_UTIL=0.4
    export RERANKER_MAX_MODEL_LEN=4096
    export RERANKER_ATTENTION_BACKEND="FLASHINFER"
    nohup .venv/bin/uvicorn third_party.backend_reranker.main:app --host 0.0.0.0 --port 18083 > reranker.log 2>&1 &
}

case $SERVICE in
    neo4j)  start_neo4j ;;
    gen)    start_gen ;;
    ocr)    start_ocr ;;
    embed)  start_embed ;;
    rerank) start_rerank ;;
    all)
        start_neo4j
        start_gen
        start_ocr
        start_embed
        start_rerank
        ;;
    *)
        echo "Usage: $0 {neo4j|gen|ocr|embed|rerank|all}"
        exit 1
        ;;
esac
