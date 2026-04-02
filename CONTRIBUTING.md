# Contributing to mlx-engine

## Development Setup

### Prerequisites

- macOS with Apple Silicon (M1+)
- Rust 1.85+
- CMake (`brew install cmake`)
- Xcode Command Line Tools
- Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain`)

### Build

```bash
git clone https://github.com/vecpluto/mlx-engine.git
cd mlx-engine
cargo build --release
```

### Test

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --all -- --check
```

## Project Structure

- `src/main.rs` — CLI entry point and command dispatch
- `src/model.rs` — Model loading, quantization, weight remapping, architecture detection
- `src/inference.rs` — Core inference loop, token generation with streaming callback
- `src/server.rs` — OpenAI-compatible HTTP API with true SSE streaming
- `src/bench.rs` — Performance benchmarking

## Adding a New Model

1. Ensure the model is implemented in upstream `mlx-lm` crate
2. Add variant to `ModelFamily` and `ModelWrapper` enums in `src/model.rs`
3. Add architecture string detection in `detect_architecture()`
4. Add chat template formatter in `src/server.rs`
5. Add EOS token fallback in `load_model_and_tokenizer()`
6. Update tests and README

## Code Standards

- Zero clippy warnings (`cargo clippy --all-targets -- -D warnings`)
- All tests passing (`cargo test`)
- Formatted with rustfmt (`cargo fmt`)
- No hardcoded token IDs — read from config.json

## Pull Requests

- One feature per PR
- Include tests for new functionality
- Update README if adding user-facing features
- Run the full check suite before submitting
