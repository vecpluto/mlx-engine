# mlx-engine

Native Apple Silicon LLM inference engine powered by [MLX](https://github.com/ml-explore/mlx). Single binary, zero Python dependency.

**124+ tok/s** decode speed on Qwen3-4B-4bit (Apple Silicon).

## Benchmark

```
╔══════════════════════════════════════════╗
║        MLX-Engine Benchmark Results      ║
╠══════════════════════════════════════════╣
║  Prompt tokens:         11               ║
║  Generated tokens:     128               ║
║                                          ║
║  TTFT (prefill):       0.109s             ║
║  Prefill speed:        100.8 tok/s       ║
║  Decode time:          1.021s             ║
║  Decode speed:         124.4 tok/s       ║
║  Total time:           1.130s             ║
╚══════════════════════════════════════════╝
```

Tested on MacBook Pro M3 Pro, Qwen3-4B-4bit.

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
```

Models are automatically resolved from the HuggingFace cache (`~/.cache/huggingface/hub/`). Download models first with:

```bash
pip install huggingface-hub
huggingface-cli download mlx-community/Qwen3-4B-4bit
```

## Supported Models

Currently supports **Qwen3** architecture (including quantized 4-bit variants from [mlx-community](https://huggingface.co/mlx-community)):

- `mlx-community/Qwen3-4B-4bit`
- `mlx-community/Qwen3-1.7B-4bit`

Llama support coming soon (already implemented in upstream mlx-lm).

## Architecture

Built on top of [mlx-rs](https://github.com/oxiglade/mlx-rs) (Rust bindings for Apple's MLX framework) and its `mlx-lm` crate for model implementations.

Key technical details:
- **Pre-quantized model loading**: Handles the correct load order (quantize structure first, then load pre-quantized weights) with key remapping for `QuantizedLinear`'s `.inner.weight` paths
- **QuantizedEmbedding workaround**: Direct field patching for mlx-rs v0.25.3's missing `#[param]` attributes
- **Custom forward step**: Replaces the library's `Generate` iterator for proper KV cache handling and shape management

## Commands

| Command | Description |
|---------|-------------|
| `chat` | Interactive multi-turn conversation |
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

## License

MIT
