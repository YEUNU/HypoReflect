#!/bin/bash

# Optional Java setup (for local/custom Neo4j distributions)
if [ -n "${JAVA_HOME:-}" ]; then
    export PATH="$JAVA_HOME/bin:$PATH"
fi

SERVICE="${1:-all}"
NEO4J_CONTAINER_NAME="${NEO4J_CONTAINER_NAME:-hyporeflect-neo4j}"

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

is_docker_available() {
    command -v docker >/dev/null 2>&1
}

is_container_running() {
    local name="$1"
    docker ps --format '{{.Names}}' 2>/dev/null | grep -Fxq "${name}"
}

kill_matching_processes() {
    local pattern="$1"
    local pid

    while read -r pid; do
        [ -z "${pid}" ] && continue
        [ "${pid}" = "$$" ] && continue
        [ "${pid}" = "${PPID}" ] && continue
        kill -9 "${pid}" 2>/dev/null || true
    done < <(pgrep -f -- "${pattern}" 2>/dev/null || true)
}

kill_port() {
    local port="$1"
    fuser -k -9 "${port}/tcp" >/dev/null 2>&1 || true
}

show_port_status() {
    local port
    echo ""
    echo "========================================="
    echo "Port Status Check:"
    echo "========================================="
    for port in "$@"; do
        if fuser "${port}/tcp" >/dev/null 2>&1; then
            echo "⚠️  Port ${port} still in use!"
        else
            echo "✅ Port ${port} is free"
        fi
    done
}

stop_neo4j() {
    local neo4j_cmd
    echo "[Neo4j] Stopping..."

    neo4j_cmd="$(resolve_neo4j_cmd || true)"
    if [ -n "${neo4j_cmd}" ]; then
        "${neo4j_cmd}" stop >/dev/null 2>&1 || true
    fi

    if is_docker_available && is_container_running "${NEO4J_CONTAINER_NAME}"; then
        docker stop "${NEO4J_CONTAINER_NAME}" >/dev/null || true
    fi

    sleep 1
    kill_port 7474
    kill_port 7687
    kill_matching_processes "org\\.neo4j"
    kill_matching_processes "neo4j console"
}

stop_gen() {
    echo "[Generation] Stopping (port 28000)..."
    kill_port 28000
    kill_matching_processes "served-model-name generation-model"
}

stop_ocr() {
    echo "[OCR] Stopping (port 28001)..."
    kill_port 28001
    kill_matching_processes "served-model-name ocr-model"
}

stop_embed() {
    echo "[Embedding] Stopping (port 18082)..."
    kill_port 18082
    kill_matching_processes "served-model-name embedding-model"
    kill_matching_processes "backend_embedding"
}

stop_rerank() {
    echo "[Reranker] Stopping (port 18083)..."
    kill_port 18083
    kill_matching_processes "backend_reranker"
    kill_matching_processes "uvicorn third_party\\.backend_reranker\\.main:app"
}

stop_cleanup() {
    echo "[Cleanup] Stopping remaining indexing/uvicorn processes..."
    kill_matching_processes "main\\.py --mode index"
    kill_matching_processes "uvicorn third_party"
}

echo "========================================="
echo "     Aggressive Server Shutdown          "
echo "========================================="

case "${SERVICE}" in
    neo4j)
        stop_neo4j
        show_port_status 7474 7687
        ;;
    gen)
        stop_gen
        show_port_status 28000
        ;;
    ocr)
        stop_ocr
        show_port_status 28001
        ;;
    embed)
        stop_embed
        show_port_status 18082
        ;;
    rerank)
        stop_rerank
        show_port_status 18083
        ;;
    all)
        stop_neo4j
        stop_gen
        stop_ocr
        stop_embed
        stop_rerank
        stop_cleanup
        show_port_status 28000 28001 18082 18083 7474 7687
        ;;
    *)
        echo "Usage: $0 {neo4j|gen|ocr|embed|rerank|all}"
        exit 1
        ;;
esac

echo ""
echo "========================================="
echo "     Requested shutdown completed.       "
echo "========================================="
