#!/usr/bin/env bash
# ci-mlx-smoke-test.sh — start mesh-llm with a tiny MLX model on macOS,
# run one inference request, verify template selection, then shut down.
#
# Usage:
#   scripts/ci-mlx-smoke-test.sh <mesh-llm-binary> <mlx-model-dir-or-repo> [expected-template-source]

set -euo pipefail

MESH_LLM="$1"
MODEL_SPEC="$2"
EXPECTED_TEMPLATE_SOURCE="${3:-}"
API_PORT=9337
CONSOLE_PORT=3131
MAX_WAIT=300
LOG=/tmp/mesh-llm-ci-mlx.log

echo "=== CI MLX Smoke Test ==="
echo "  mesh-llm:  $MESH_LLM"
echo "  model:     $MODEL_SPEC"
echo "  api port:  $API_PORT"
echo "  os:        $(uname -s)"

if [ "$(uname -s)" != "Darwin" ]; then
    echo "❌ MLX smoke test only supports macOS"
    exit 1
fi

if [ ! -f "$MESH_LLM" ]; then
    echo "❌ Missing mesh-llm binary: $MESH_LLM"
    exit 1
fi

echo "Starting mesh-llm..."
if [ -d "$MODEL_SPEC" ]; then
    RUST_LOG=info "$MESH_LLM" \
        --mlx-file "$MODEL_SPEC" \
        --no-draft \
        --port "$API_PORT" \
        --console "$CONSOLE_PORT" \
        > "$LOG" 2>&1 &
else
    RUST_LOG=info "$MESH_LLM" \
        --model "$MODEL_SPEC" \
        --mlx \
        --no-draft \
        --port "$API_PORT" \
        --console "$CONSOLE_PORT" \
        > "$LOG" 2>&1 &
fi
MESH_PID=$!
echo "  PID: $MESH_PID"

cleanup() {
    echo "Shutting down mesh-llm (PID $MESH_PID)..."
    kill "$MESH_PID" 2>/dev/null || true
    pkill -P "$MESH_PID" 2>/dev/null || true
    sleep 2
    kill -9 "$MESH_PID" 2>/dev/null || true
    wait "$MESH_PID" 2>/dev/null || true
    echo "Cleanup done."
}
trap cleanup EXIT

echo "Waiting for model to load (up to ${MAX_WAIT}s)..."
for i in $(seq 1 "$MAX_WAIT"); do
    if ! kill -0 "$MESH_PID" 2>/dev/null; then
        echo "❌ mesh-llm exited unexpectedly"
        echo "--- Log tail ---"
        tail -80 "$LOG" || true
        exit 1
    fi

    READY=$(curl -sf "http://localhost:${CONSOLE_PORT}/api/status" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('llama_ready', False))" 2>/dev/null || echo "False")
    if [ "$READY" = "True" ]; then
        echo "✅ Model loaded in ${i}s"
        break
    fi

    if [ "$i" -eq "$MAX_WAIT" ]; then
        echo "❌ Model failed to load within ${MAX_WAIT}s"
        echo "--- Log tail ---"
        tail -80 "$LOG" || true
        exit 1
    fi

    if [ $((i % 15)) -eq 0 ]; then
        echo "  Still waiting... (${i}s)"
    fi
    sleep 1
done

if [ -n "$EXPECTED_TEMPLATE_SOURCE" ]; then
    if ! grep -F "MLX prompt template: loaded HF template from $EXPECTED_TEMPLATE_SOURCE" "$LOG" >/dev/null 2>&1; then
        echo "❌ Expected template source not found in log: $EXPECTED_TEMPLATE_SOURCE"
        echo "--- Log tail ---"
        tail -120 "$LOG" || true
        exit 1
    fi
    echo "✅ Template source matched: $EXPECTED_TEMPLATE_SOURCE"
fi

echo "Testing /v1/chat/completions..."
RESPONSE=$(curl -sf "http://localhost:${API_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "any",
        "messages": [{"role": "user", "content": "What is 2+2? Reply with one word only."}],
        "max_tokens": 32,
        "temperature": 0,
        "enable_thinking": false
    }' 2>&1)

if [ $? -ne 0 ]; then
    echo "❌ Inference request failed"
    echo "$RESPONSE"
    echo "--- Log tail ---"
    tail -80 "$LOG" || true
    exit 1
fi

CONTENT=$(echo "$RESPONSE" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'])" 2>/dev/null || echo "")
if [ -z "$CONTENT" ]; then
    echo "❌ Empty response from inference"
    echo "Raw response: $RESPONSE"
    exit 1
fi

FINISH_REASON=$(echo "$RESPONSE" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0].get('finish_reason',''))" 2>/dev/null || echo "")
if [ -z "$FINISH_REASON" ]; then
    echo "❌ Missing finish_reason in response"
    echo "Raw response: $RESPONSE"
    exit 1
fi

echo "✅ Inference response: $CONTENT"

echo "Testing /v1/models..."
MODELS=$(curl -sf "http://localhost:${API_PORT}/v1/models" 2>&1)
MODEL_COUNT=$(echo "$MODELS" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('data',[])))" 2>/dev/null || echo "0")
if [ "$MODEL_COUNT" -eq 0 ]; then
    echo "❌ No models in /v1/models"
    echo "$MODELS"
    exit 1
fi
echo "✅ /v1/models returned $MODEL_COUNT model(s)"

echo ""
echo "=== MLX smoke test passed ==="
