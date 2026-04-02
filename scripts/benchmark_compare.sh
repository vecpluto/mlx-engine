#!/bin/bash
# MLX-Engine vs Ollama Benchmark Comparison
# Usage: ./scripts/benchmark_compare.sh

set -euo pipefail

PROMPT="Write a short story about a robot learning to paint."
NUM_TOKENS=128
RUNS=3

echo "╔══════════════════════════════════════════════════╗"
echo "║     MLX-Engine vs Ollama Benchmark Comparison    ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║  Prompt: \"${PROMPT:0:45}...\"                    "
echo "║  Tokens: $NUM_TOKENS | Runs: $RUNS              "
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── MLX-Engine Benchmark ───────────────────��──────
echo "=== MLX-Engine (Qwen3-4B-4bit, mlx-rs) ==="
MLX_BIN="$(dirname "$0")/../target/release/mlx-engine"
MLX_MODEL="mlx-community/Qwen3-4B-4bit"

if [ ! -f "$MLX_BIN" ]; then
    echo "Building mlx-engine..."
    (cd "$(dirname "$0")/.." && source "$HOME/.cargo/env" && cargo build --release 2>&1 | tail -1)
fi

for i in $(seq 1 $RUNS); do
    echo ""
    echo "--- Run $i/$RUNS ---"
    "$MLX_BIN" bench --model "$MLX_MODEL" --num-tokens "$NUM_TOKENS" --prompt "$PROMPT" 2>&1
done

echo ""
echo ""

# ── Ollama Benchmark ──────────────────────────────
echo "=== Ollama (qwen3:4b, GGUF Q4_K_M) ==="

if ! command -v ollama &>/dev/null; then
    echo "Ollama not installed, skipping."
    exit 0
fi

# Check if model exists
if ! ollama list 2>/dev/null | grep -q "qwen3:4b"; then
    echo "Model qwen3:4b not available. Pull with: ollama pull qwen3:4b"
    exit 0
fi

for i in $(seq 1 $RUNS); do
    echo ""
    echo "--- Run $i/$RUNS ---"

    START_NS=$(python3 -c 'import time; print(int(time.time_ns()))')

    # Use Ollama API for precise measurement
    RESPONSE=$(curl -s http://localhost:11434/api/generate \
        -d "{\"model\":\"qwen3:4b\",\"prompt\":\"$PROMPT\",\"options\":{\"num_predict\":$NUM_TOKENS,\"temperature\":0},\"stream\":false}" \
        2>/dev/null)

    END_NS=$(python3 -c 'import time; print(int(time.time_ns()))')

    # Extract metrics from Ollama response
    TOTAL_DURATION=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('total_duration',0))" 2>/dev/null || echo "0")
    LOAD_DURATION=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('load_duration',0))" 2>/dev/null || echo "0")
    PROMPT_EVAL_DURATION=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('prompt_eval_duration',0))" 2>/dev/null || echo "0")
    EVAL_DURATION=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('eval_duration',0))" 2>/dev/null || echo "0")
    PROMPT_EVAL_COUNT=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('prompt_eval_count',0))" 2>/dev/null || echo "0")
    EVAL_COUNT=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('eval_count',0))" 2>/dev/null || echo "0")

    # Calculate TPS
    if [ "$EVAL_DURATION" -gt 0 ]; then
        DECODE_TPS=$(python3 -c "print(f'{$EVAL_COUNT / ($EVAL_DURATION / 1e9):.1f}')")
    else
        DECODE_TPS="N/A"
    fi

    if [ "$PROMPT_EVAL_DURATION" -gt 0 ]; then
        PREFILL_TPS=$(python3 -c "print(f'{$PROMPT_EVAL_COUNT / ($PROMPT_EVAL_DURATION / 1e9):.1f}')")
    else
        PREFILL_TPS="N/A"
    fi

    TTFT_S=$(python3 -c "print(f'{$PROMPT_EVAL_DURATION / 1e9:.3f}')")
    DECODE_S=$(python3 -c "print(f'{$EVAL_DURATION / 1e9:.3f}')")
    TOTAL_S=$(python3 -c "print(f'{$TOTAL_DURATION / 1e9:.3f}')")

    echo "╔══════════════════════════════════════════╗"
    echo "║          Ollama Benchmark Results        ║"
    echo "╠══════════════════════════════════════════╣"
    printf "║  Prompt tokens:     %6s               ║\n" "$PROMPT_EVAL_COUNT"
    printf "║  Generated tokens:  %6s               ║\n" "$EVAL_COUNT"
    echo "║                                          ║"
    printf "║  TTFT (prefill):    %8ss             ║\n" "$TTFT_S"
    printf "║  Prefill speed:     %8s tok/s       ║\n" "$PREFILL_TPS"
    printf "║  Decode time:       %8ss             ║\n" "$DECODE_S"
    printf "║  Decode speed:      %8s tok/s       ║\n" "$DECODE_TPS"
    printf "║  Total time:        %8ss             ║\n" "$TOTAL_S"
    echo "╚══════════════════════════════════════════╝"
done

echo ""
echo "Done. Compare decode speed (tok/s) across engines."
