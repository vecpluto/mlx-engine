# mlx-engine

Native Apple Silicon LLM inference engine powered by [MLX](https://github.com/ml-explore/mlx). Single binary, zero Python dependency.

**120+ tok/s** decode speed on Qwen3-4B-4bit — **faster than Ollama** on Apple Silicon.

## Benchmark: mlx-engine vs Ollama

Qwen3-4B, 128 tokens generated, MacBook Pro M3 Pro. Best of 3 warm runs:

| Metric | mlx-engine | Ollama 0.12 (llama.cpp) |
|--------|:----------:|:-----------------------:|
| **Decode speed** | **119.9 tok/s** | 114.7 tok/s |
| TTFT (warm) | 0.059s | 0.009s |
| Total time | 1.12s | 1.19s |
| Binary size | Single Rust | Go + llama.cpp C++ |
| MLX native | Yes | Partial (v0.19 preview) |
| Python dependency | None | None |

mlx-engine achieves **~5% faster decode** with native MLX Metal acceleration.

### Raw benchmark output

```
╔══════════════════════════════════════════╗
║        MLX-Engine Benchmark Results      ║
╠══════════════════════════════════════════╣
║  Prompt tokens:         11               ║
║  Generated tokens:     128               ║
║                                          ║
║  TTFT (prefill):       0.059s             ║
║  Prefill speed:        187.9 tok/s       ║
║  Decode time:          1.060s             ║
║  Decode speed:         119.9 tok/s       ║
║  Total time:           1.119s             ║
╚══════════════════════════════════════════╝
```

## Quick Start

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4/M5)
- Rust 1.85+ (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- CMake (`brew install cmake`)
- Xcode Command Line Tools (`xcode-select --install`)
- Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain`)

### Build

```bash
git clone https://github.com/vecpluto/mlx-engine.git
cd mlx-engine
cargo build --release
```

### Run

```bash
# Interactive chat
./target/release/mlx-engine chat --model mlx-community/Qwen3-4B-4bit

# Single prompt
./target/release/mlx-engine generate \
  --model mlx-community/Qwen3-4B-4bit \
  --prompt "Explain quantum computing in simple terms." \
  --temp 0.7

# Benchmark
./target/release/mlx-engine bench \
  --model mlx-community/Qwen3-4B-4bit \
  --num-tokens 128

# OpenAI-compatible API server
./target/release/mlx-engine serve \
  --model mlx-community/Qwen3-4B-4bit \
  --port 8080

# Non-streaming request
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3","messages":[{"role":"user","content":"Hello!"}],"max_tokens":100}'

# Streaming request (per-token SSE)
curl --no-buffer http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3","messages":[{"role":"user","content":"Hello!"}],"max_tokens":100,"stream":true}'
```

Models are automatically resolved from the HuggingFace cache (`~/.cache/huggingface/hub/`). Download models first with:

```bash
pip install huggingface-hub
huggingface-cli download mlx-community/Qwen3-4B-4bit
```

## Supported Models

Architecture is auto-detected from `config.json`'s `"architectures"` field. No manual configuration needed.

### Qwen3

`Qwen3ForCausalLM` architecture — includes quantized 4-bit variants from [mlx-community](https://huggingface.co/mlx-community):

- `mlx-community/Qwen3-4B-4bit`
- `mlx-community/Qwen3-1.7B-4bit`

### Llama

`LlamaForCausalLM` architecture — Llama 3.x instruct models and quantized variants:

- `mlx-community/Llama-3.2-3B-4bit`
- Any local or HuggingFace model with `"LlamaForCausalLM"` in `config.json`

## Streaming

The `serve` command implements true per-token SSE streaming. Each decoded token is sent to the client immediately as it is produced — no buffering.

Enable streaming by setting `"stream": true` in the request body, and pass `--no-buffer` to curl to see tokens arrive in real time:

```bash
curl --no-buffer http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3",
    "messages": [{"role": "user", "content": "Count to ten."}],
    "max_tokens": 200,
    "stream": true
  }'
```

Response chunks follow the OpenAI SSE format (`data: {...}\n\n`), terminated by `data: [DONE]\n\n`. Any OpenAI-compatible client library works without modification.

## Architecture

Built on top of [mlx-rs](https://github.com/oxiglade/mlx-rs) (Rust bindings for Apple's MLX framework) and its `mlx-lm` crate for model implementations.

Key technical details:

- **Multi-model support via `ModelWrapper` enum**: `ModelWrapper::Qwen3` and `ModelWrapper::Llama` variants share a single inference path. Adding a new architecture requires only a new enum variant and a string match in `detect_architecture()`.
- **Architecture auto-detection**: `config.json`'s `"architectures"` array is read at load time. Supported values: `Qwen3ForCausalLM`, `LlamaForCausalLM`. An unsupported value produces a clear error listing what was found.
- **Pre-quantized model loading**: Handles the correct load order (quantize structure first, then load pre-quantized weights) with key remapping for `QuantizedLinear`'s `.inner.weight` paths.
- **QuantizedEmbedding workaround**: Direct field patching for mlx-rs v0.25.3's missing `#[param]` attributes.
- **Custom forward step**: Replaces the library's `Generate` iterator for proper KV cache handling and shape management.
- **Thread isolation for SSE streaming**: `mlx_rs::Array` is `!Send`. The inference loop runs on a dedicated OS thread for its entire lifetime; Axum handlers communicate via `std::sync::mpsc` (requests) and `tokio::sync::mpsc` (per-token stream events).

## Commands

| Command | Description |
|---------|-------------|
| `chat` | Interactive multi-turn conversation |
| `serve` | OpenAI-compatible HTTP API with true per-token SSE streaming |
| `generate` | Single prompt text generation |
| `bench` | Performance benchmark (TTFT, TPOT, TPS) |

## Why mlx-engine?

| | mlx-engine | Ollama | llama.cpp | Python mlx-lm |
|---|:---:|:---:|:---:|:---:|
| Apple MLX native | Yes | Partial | No | Yes |
| Zero Python | Yes | Yes | Yes | No |
| Single binary | Yes | Yes | Yes | No |
| Rust memory safety | Yes | No (Go) | No (C++) | No |
| Pre-quantized 4-bit | Yes | Yes | Yes (GGUF) | Yes |
| Model support | Qwen3 + Llama | Many | Many (GGUF) | Many |

## License

MIT
