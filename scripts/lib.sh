#!/bin/bash

# Shared shell helpers for local service orchestration.

wait_for_server() {
    local url="$1"
    local name="$2"
    local success_codes="${3:-200|401|405}"
    local max_attempts="${4:-300}"
    local attempt=0
    local check_url="$url"
    local curl_args=(-s -o /dev/null -w "%{http_code}" --max-time 2)

    if [[ "$url" == */v1/models ]]; then
        # For local model readiness checks, prefer the lightweight health endpoint.
        check_url="${url%/v1/models}/health"
    elif [ -n "${VLLM_API_KEY:-}" ] && [ "${VLLM_API_KEY}" != "EMPTY" ]; then
        curl_args+=(-H "Authorization: Bearer ${VLLM_API_KEY}")
    fi

    echo "Wait for $name ($check_url)..."
    while [ "$attempt" -lt "$max_attempts" ]; do
        if curl "${curl_args[@]}" "$check_url" | grep -qE "$success_codes"; then
            echo " ✅ $name is Ready!"
            return 0
        fi
        printf "."
        sleep 5
        attempt=$((attempt + 1))

        if [ $((attempt % 12)) -eq 0 ]; then
            echo " (Waiting for ${name}... $((attempt * 5))s elapsed)"
        fi
    done

    echo ""
    echo " ❌ ERROR: $name failed to start after $((max_attempts * 5 / 60)) minutes."
    return 1
}
