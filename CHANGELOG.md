# Changelog

## [0.1.0] - 2026-04-02

### Added

- Initial release
- Qwen3 model support (including 4-bit quantized from mlx-community)
- Llama model support with auto-detection from config.json
- Interactive chat mode (`chat` command)
- Single-prompt generation (`generate` command)
- Performance benchmark with TTFT/TPOT/TPS metrics (`bench` command)
- OpenAI-compatible HTTP API server (`serve` command)
  - POST /v1/chat/completions
  - True per-token SSE streaming
  - Non-streaming JSON response
- Pre-quantized 4-bit model loading with QuantizedEmbedding workaround
- Dynamic EOS token detection from model config
- HuggingFace cache auto-resolution
- 120+ tok/s decode speed on Qwen3-4B-4bit (M3 Pro)
