use std::collections::HashMap;
use std::io::{self, Write};
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use mlx_lm::cache::ConcatKeyValueCache;
use mlx_lm::models::qwen3::{
    get_qwen3_model_args, load_qwen3_tokenizer, Model, ModelInput, WeightMap, sample,
};
use mlx_rs::module::{Module, ModuleParameters, ModuleParametersExt};
use mlx_rs::nn::QuantizedEmbedding;
use mlx_rs::ops::indexing::{IndexOp, NewAxis};
use mlx_rs::quantization::{MaybeQuantized, Quantizable};
use mlx_rs::transforms::eval;
use mlx_rs::Array;

// ── HTTP server imports ────────────────────────────────────────────────────────
use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tower_http::cors::{Any, CorsLayer};
use uuid::Uuid;

// ── CLI definition ─────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "mlx-engine")]
#[command(about = "Native Apple Silicon LLM inference. Single binary, zero Python.")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Interactive chat with a model
    Chat {
        /// Path to model directory (HuggingFace cache or local)
        #[arg(short, long)]
        model: String,

        /// Sampling temperature (0.0 = greedy)
        #[arg(short, long, default_value = "0.7")]
        temp: f32,

        /// Maximum tokens to generate per turn
        #[arg(long, default_value = "512")]
        max_tokens: usize,
    },
    /// Generate text from a prompt (non-interactive)
    Generate {
        /// Path to model directory
        #[arg(short, long)]
        model: String,

        /// Input prompt
        #[arg(short, long)]
        prompt: String,

        /// Sampling temperature
        #[arg(short, long, default_value = "0.7")]
        temp: f32,

        /// Maximum tokens to generate
        #[arg(long, default_value = "256")]
        max_tokens: usize,
    },
    /// Run benchmark and report TPS metrics
    Bench {
        /// Path to model directory
        #[arg(short, long)]
        model: String,

        /// Prompt for benchmarking
        #[arg(short, long, default_value = "Write a short story about a robot learning to paint.")]
        prompt: String,

        /// Number of tokens to generate
        #[arg(short, long, default_value = "128")]
        num_tokens: usize,
    },
    /// Start an OpenAI-compatible HTTP API server
    Serve {
        /// Path to model directory (HuggingFace cache or local)
        #[arg(short, long)]
        model: String,

        /// Port to listen on
        #[arg(short, long, default_value = "8080")]
        port: u16,

        /// Host address to bind
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
    },
}

// ── Model path resolution ──────────────────────────────────────────────────────

fn resolve_model_path(model: &str) -> Result<PathBuf> {
    let path = PathBuf::from(model);
    if path.exists() {
        return Ok(path);
    }

    // Try HuggingFace cache
    let hf_cache = dirs::home_dir()
        .context("cannot find home dir")?
        .join(".cache/huggingface/hub");

    let model_dir_name = format!("models--{}", model.replace('/', "--"));
    let model_cache = hf_cache.join(&model_dir_name);

    if model_cache.exists() {
        // Find latest snapshot
        let snapshots_dir = model_cache.join("snapshots");
        if snapshots_dir.exists() {
            let mut entries: Vec<_> = std::fs::read_dir(&snapshots_dir)?
                .filter_map(|e| e.ok())
                .collect();
            entries.sort_by_key(|e| e.metadata().ok().and_then(|m| m.modified().ok()));
            if let Some(latest) = entries.last() {
                return Ok(latest.path());
            }
        }
    }

    anyhow::bail!(
        "Model not found: '{}'. Provide a local path or HuggingFace model ID (e.g., mlx-community/Qwen3-4B-4bit)",
        model
    )
}

// ── Weight loading helpers ─────────────────────────────────────────────────────

/// Collect all safetensors weights from a model directory into a single flat map.
fn collect_safetensors_weights(path: &PathBuf) -> Result<HashMap<String, Array>> {
    let weights_index_path = path.join("model.safetensors.index.json");
    let weights_path = path.join("model.safetensors");

    let mut all_weights: HashMap<String, Array> = HashMap::new();

    if weights_index_path.exists() {
        let json = std::fs::read_to_string(&weights_index_path)?;
        let weight_map: WeightMap = serde_json::from_str(&json)?;
        let weight_files: std::collections::HashSet<&String> =
            weight_map.weight_map.values().collect();
        for weight_file in weight_files {
            let wf = path.join(weight_file);
            let tensors = Array::load_safetensors(&wf)
                .map_err(|e| anyhow::anyhow!("Failed to load {}: {:?}", weight_file, e))?;
            all_weights.extend(tensors);
        }
    } else if weights_path.exists() {
        let tensors = Array::load_safetensors(&weights_path)
            .map_err(|e| anyhow::anyhow!("Failed to load model.safetensors: {:?}", e))?;
        all_weights.extend(tensors);
    } else {
        anyhow::bail!("No safetensors weights found in {}", path.display());
    }

    Ok(all_weights)
}

/// Directly patch a `QuantizedEmbedding`'s weight arrays from the safetensors map.
///
/// In mlx-rs v0.25.3, `QuantizedEmbedding` fields lack `#[param]` attributes, so
/// `parameters_mut().flatten()` returns nothing for this module. We must write the
/// three tensors (weight, scales, biases) directly into the struct fields.
///
/// The prefix is e.g. `"model.embed_tokens"`.  Matched keys are drained from the map.
fn patch_quantized_embedding(
    qe: &mut QuantizedEmbedding,
    prefix: &str,
    weights: &mut HashMap<String, Array>,
) {
    let weight_key = format!("{prefix}.weight");
    let scales_key = format!("{prefix}.scales");
    let biases_key = format!("{prefix}.biases");

    if let Some(w) = weights.remove(&weight_key) {
        qe.inner.weight.value = w;
    }
    if let Some(s) = weights.remove(&scales_key) {
        qe.scales.value = s;
    }
    if let Some(b) = weights.remove(&biases_key) {
        qe.biases.value = b;
    }
}

/// Load pre-quantized safetensors weights into an already-quantized model.
///
/// Two-phase approach:
///
/// Phase 1 — `QuantizedEmbedding` direct patch: in mlx-rs v0.25.3 `QuantizedEmbedding`
/// exposes no `#[param]`-annotated fields, so its parameters are invisible to
/// `parameters_mut().flatten()`.  We reach into `model.model.embed_tokens` directly
/// and assign weight/scales/biases by field access.
///
/// Phase 2 — `QuantizedLinear` via parameter API: for every remaining key the
/// safetensors stores `<prefix>.weight` while the quantized model stores it as
/// `<prefix>.inner.weight`.  We remap the key and call `update_flattened`.
fn load_prequantized_weights(
    model: &mut Model,
    mut safetensors_weights: HashMap<String, Array>,
) -> Result<()> {
    // Phase 1: directly patch embed_tokens, which uses QuantizedEmbedding (no #[param] fields
    // in v0.25.3 — invisible to the standard parameter API).
    let embed_prefix = "model.embed_tokens";
    match &mut model.model.embed_tokens {
        MaybeQuantized::Quantized(qe) => {
            patch_quantized_embedding(qe, embed_prefix, &mut safetensors_weights);
            eprintln!("  Patched embed_tokens directly (QuantizedEmbedding).");
        }
        MaybeQuantized::Original(_) => {
            // Non-quantized embedding: standard path handles it in phase 2.
        }
    }

    // Phase 2: load all remaining keys through the standard parameter API.
    // QuantizedLinear exposes `inner.weight`, `scales`, `biases` via #[param].
    // Safetensors stores `.weight` without the `.inner` prefix for quantized layers.
    let model_keys: std::collections::HashSet<String> = model
        .parameters_mut()
        .flatten()
        .keys()
        .map(|k| k.to_string())
        .collect();

    let mut remapped: HashMap<Rc<str>, Array> = HashMap::new();
    let mut unmatched_count = 0usize;

    for (st_key, array) in safetensors_weights {
        // Direct match (non-quantized parameters like rms_norm.weight).
        if model_keys.contains(&st_key) {
            remapped.insert(Rc::from(st_key.as_str()), array);
            continue;
        }

        // Quantized linear layers: remap `<prefix>.weight` → `<prefix>.inner.weight`.
        if st_key.ends_with(".weight") {
            let inner_key = format!(
                "{}.inner.weight",
                &st_key[..st_key.len() - ".weight".len()]
            );
            if model_keys.contains(&inner_key) {
                remapped.insert(Rc::from(inner_key.as_str()), array);
                continue;
            }
        }

        unmatched_count += 1;
        eprintln!("  [warn] unmatched safetensors key: {}", st_key);
    }

    eprintln!(
        "  Weight mapping: {} matched via parameter API, {} unmatched",
        remapped.len(),
        unmatched_count
    );

    model.update_flattened(remapped);
    model.eval().map_err(|e| anyhow::anyhow!("{:?}", e))?;

    Ok(())
}

fn load_model_and_tokenizer(
    model_path: &str,
) -> Result<(Model, tokenizers::Tokenizer, PathBuf)> {
    let path = resolve_model_path(model_path)?;
    eprintln!("Loading model from: {}", path.display());

    let start = Instant::now();

    // Read config to detect quantization settings.
    let model_args = get_qwen3_model_args(&path).context("Failed to read config.json")?;
    let config_path = path.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)?;
    let config_json: serde_json::Value = serde_json::from_str(&config_str)?;

    let quantization = config_json.get("quantization");
    let is_quantized = quantization.is_some();

    if is_quantized {
        let q = quantization.unwrap();
        let gs = q.get("group_size").and_then(|v| v.as_i64()).unwrap_or(64);
        let b = q.get("bits").and_then(|v| v.as_i64()).unwrap_or(4);
        eprintln!("Detected pre-quantized model: {}-bit, group_size={}", b, gs);
    }

    let mut model = Model::new(model_args).context("Failed to create model")?;

    if is_quantized {
        // For pre-quantized models the correct loading order is:
        //   1. Quantize the model structure first — this creates QuantizedLinear /
        //      QuantizedEmbedding shells whose flattened parameter keys include `.inner.weight`.
        //   2. Collect all safetensors tensors into memory.
        //   3. Remap `<prefix>.weight` → `<prefix>.inner.weight` for every quantized layer
        //      and apply the weights via update_flattened.
        //
        // Loading non-quantized weights into a non-quantized model first then re-quantizing
        // is WRONG: it would re-quantize the already-quantized (int4) data as if it were
        // float weights, producing garbage output.
        let q = quantization.unwrap();
        let gs = q.get("group_size").and_then(|v| v.as_i64()).unwrap_or(64) as i32;
        let b = q.get("bits").and_then(|v| v.as_i64()).unwrap_or(4) as i32;

        eprintln!("Quantizing model structure ({}-bit, group_size={})...", b, gs);
        model = model
            .try_into_quantized(gs, b)
            .map_err(|e| anyhow::anyhow!("{:?}", e))
            .context("Failed to create quantized model structure")?;

        eprintln!("Loading pre-quantized weights...");
        let weights = collect_safetensors_weights(&path)?;
        load_prequantized_weights(&mut model, weights)?;
    } else {
        // Non-quantized model: standard load — keys match directly.
        let weights_index_path = path.join("model.safetensors.index.json");
        let weights_path = path.join("model.safetensors");

        if weights_index_path.exists() {
            let json = std::fs::read_to_string(&weights_index_path)?;
            let weight_map: WeightMap = serde_json::from_str(&json)?;
            let weight_files: std::collections::HashSet<&String> =
                weight_map.weight_map.values().collect();
            for weight_file in weight_files {
                let wf = path.join(weight_file);
                model
                    .load_safetensors(&wf)
                    .map_err(|e| anyhow::anyhow!("{:?}", e))?;
            }
        } else {
            model
                .load_safetensors(&weights_path)
                .map_err(|e| anyhow::anyhow!("{:?}", e))?;
        }
    }

    let tokenizer = load_qwen3_tokenizer(&path).context("Failed to load tokenizer")?;
    let load_time = start.elapsed();
    eprintln!("Model loaded in {:.2}s", load_time.as_secs_f64());

    Ok((model, tokenizer, path))
}

// ── Inference core ─────────────────────────────────────────────────────────────

/// Single-step forward pass returning shape `(batch, vocab_size)` logits for the last token.
///
/// The library's `Generate` iterator has a decode-mode shape bug: `argmax_axis` on
/// `(batch, 1, vocab_size)` produces `(batch, 1)`, causing the next input to be
/// `(batch, 1, 1)` and growing unboundedly.  We implement our own loop to avoid this.
fn forward_step(
    model: &mut Model,
    inputs: &Array,
    cache: &mut Vec<Option<ConcatKeyValueCache>>,
) -> Result<Array> {
    let input = ModelInput {
        inputs,
        mask: None,
        cache,
    };
    let logits = model
        .forward(input)
        .map_err(|e| anyhow::anyhow!("{:?}", e))?;

    // logits shape: (batch, seq_len, vocab_size) — select last position.
    // For seq_len=1 (decode step) this is a no-op in terms of semantics but keeps
    // shape consistent at (batch, vocab_size) for the argmax / categorical call.
    let last_logits = logits.index((.., -1i32, ..));
    Ok(last_logits)
}

fn generate_tokens(
    model: &mut Model,
    tokenizer: &tokenizers::Tokenizer,
    prompt: &str,
    temp: f32,
    max_tokens: usize,
    show_stats: bool,
) -> Result<String> {
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    let num_prompt_tokens = encoding.get_ids().len();

    // Pre-initialize every layer's KV cache so the prefill step stores keys/values.
    // Without pre-initialization the cache entries are None, Attention.forward skips
    // the KV storage branch, and decode steps run with no cached context.
    let num_layers = model.args.num_hidden_layers as usize;
    let mut cache: Vec<Option<ConcatKeyValueCache>> =
        (0..num_layers).map(|_| Some(ConcatKeyValueCache::new())).collect();

    let mut tokens: Vec<Array> = Vec::new();
    let mut decoded = String::new();
    let gen_start = Instant::now();
    let mut token_count = 0usize;

    // --- Prefill ---
    let prompt_tokens = Array::from(encoding.get_ids()).index(NewAxis);
    let logits = forward_step(model, &prompt_tokens, &mut cache)?;

    // Sample first token; logits shape is (batch, vocab_size).
    let mut y = sample(&logits, temp).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    // y shape: (batch,) after argmax with keep_dims=false

    eval(&[y.clone()]).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let ttft = Some(gen_start.elapsed().as_secs_f64());

    let first_id: u32 = y.item();
    tokens.push(y.clone());
    token_count += 1;

    // Check EOS immediately
    if first_id == 151645 || first_id == 151643 {
        // flush and return
        let slice: Vec<u32> = tokens.drain(..).map(|t| t.item::<u32>()).collect();
        let chunk = tokenizer.decode(&slice, true).map_err(|e| anyhow::anyhow!("{}", e))?;
        if !show_stats { print!("{chunk}"); io::stdout().flush()?; }
        decoded.push_str(&chunk);
        print_perf_stats(num_prompt_tokens, ttft, gen_start.elapsed().as_secs_f64(), token_count);
        if !show_stats { println!(); }
        return Ok(decoded);
    }

    // --- Decode loop ---
    // y always has shape (batch,) = (1,) so y.index(NewAxis) gives (1, 1).
    loop {
        if token_count >= max_tokens {
            break;
        }

        // Decode step: single-token input, shape (batch, 1).
        let inputs = y.index((.., NewAxis));
        let logits = forward_step(model, &inputs, &mut cache)?;

        // logits shape: (batch, vocab_size) — safe to sample directly.
        y = sample(&logits, temp).map_err(|e| anyhow::anyhow!("{:?}", e))?;
        // y shape: (batch,)

        tokens.push(y.clone());
        token_count += 1;

        // Periodically eval and stream output.
        if tokens.len() % 20 == 0 {
            eval(&tokens).map_err(|e| anyhow::anyhow!("{:?}", e))?;
            let slice: Vec<u32> = tokens.drain(..).map(|t| t.item::<u32>()).collect();
            let chunk = tokenizer
                .decode(&slice, true)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            if !show_stats {
                print!("{chunk}");
                io::stdout().flush()?;
            }
            decoded.push_str(&chunk);
        }

        let token_id: u32 = y.item();
        if token_id == 151645 || token_id == 151643 {
            break;
        }
    }

    // Flush remaining tokens.
    if !tokens.is_empty() {
        eval(&tokens).map_err(|e| anyhow::anyhow!("{:?}", e))?;
        let slice: Vec<u32> = tokens.drain(..).map(|t| t.item::<u32>()).collect();
        let chunk = tokenizer
            .decode(&slice, true)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        if !show_stats {
            print!("{chunk}");
            io::stdout().flush()?;
        }
        decoded.push_str(&chunk);
    }

    let total_time = gen_start.elapsed().as_secs_f64();
    print_perf_stats(num_prompt_tokens, ttft, total_time, token_count);

    if !show_stats {
        println!();
    }

    Ok(decoded)
}

fn print_perf_stats(
    num_prompt_tokens: usize,
    ttft: Option<f64>,
    total_time: f64,
    token_count: usize,
) {
    eprintln!();
    eprintln!("─── Performance ───────────────────────");
    eprintln!("  Prompt tokens:  {}", num_prompt_tokens);
    if let Some(t) = ttft {
        eprintln!("  TTFT:           {:.3}s", t);
        let decode_time = total_time - t;
        if decode_time > 0.0 && token_count > 1 {
            let tps = (token_count - 1) as f64 / decode_time;
            eprintln!("  Decode tokens:  {}", token_count);
            eprintln!("  Decode time:    {:.3}s", decode_time);
            eprintln!("  Decode speed:   {:.1} tok/s", tps);
        }
    }
    eprintln!("  Total time:     {:.3}s", total_time);
    eprintln!("───────────────────────────────────────");
}

// ── Subcommand runners ─────────────────────────────────────────────────────────

fn run_chat(model_path: &str, temp: f32, max_tokens: usize) -> Result<()> {
    let (mut model, tokenizer, _path) = load_model_and_tokenizer(model_path)?;

    eprintln!("\nmlx-engine v{} — Type 'quit' to exit\n", env!("CARGO_PKG_VERSION"));

    loop {
        print!("You: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }
        if input == "quit" || input == "exit" {
            break;
        }

        // Simple prompt formatting for Qwen3
        let prompt = format!("<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n");

        print!("Assistant: ");
        io::stdout().flush()?;

        generate_tokens(&mut model, &tokenizer, &prompt, temp, max_tokens, false)?;

        // Reset cache between turns for simplicity in MVP
        // TODO: maintain conversation context
    }

    Ok(())
}

fn run_generate(model_path: &str, prompt: &str, temp: f32, max_tokens: usize) -> Result<()> {
    let (mut model, tokenizer, _path) = load_model_and_tokenizer(model_path)?;
    generate_tokens(&mut model, &tokenizer, prompt, temp, max_tokens, false)?;
    Ok(())
}

fn run_bench(model_path: &str, prompt: &str, num_tokens: usize) -> Result<()> {
    let (mut model, tokenizer, _path) = load_model_and_tokenizer(model_path)?;

    eprintln!("\n=== MLX-Engine Benchmark ===\n");
    eprintln!("Prompt: \"{}\"", &prompt[..prompt.len().min(80)]);
    eprintln!("Target tokens: {}\n", num_tokens);

    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    let prompt_tokens = Array::from(encoding.get_ids()).index(NewAxis);
    let num_prompt_tokens = encoding.get_ids().len();

    let num_layers = model.args.num_hidden_layers as usize;
    let mut cache: Vec<Option<ConcatKeyValueCache>> =
        (0..num_layers).map(|_| Some(ConcatKeyValueCache::new())).collect();

    let mut token_count = 0usize;
    let gen_start = Instant::now();

    // Prefill
    let logits = forward_step(&mut model, &prompt_tokens, &mut cache)?;
    let mut y = sample(&logits, 0.0).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    eval(&[y.clone()]).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let ttft = Some(gen_start.elapsed().as_secs_f64());
    token_count += 1;

    let mut batch: Vec<Array> = vec![y.clone()];

    // Decode loop
    loop {
        if token_count >= num_tokens {
            break;
        }
        let token_id: u32 = y.item();
        if token_id == 151645 || token_id == 151643 {
            break;
        }

        let inputs = y.index((.., NewAxis));
        let logits = forward_step(&mut model, &inputs, &mut cache)?;
        y = sample(&logits, 0.0).map_err(|e| anyhow::anyhow!("{:?}", e))?;
        token_count += 1;
        batch.push(y.clone());

        if batch.len() % 20 == 0 {
            eval(&batch).map_err(|e| anyhow::anyhow!("{:?}", e))?;
            batch.clear();
        }
    }

    if !batch.is_empty() {
        eval(&batch).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    }

    let total_time = gen_start.elapsed().as_secs_f64();
    let ttft_val = ttft.unwrap_or(0.0);
    let decode_time = total_time - ttft_val;
    let decode_tokens = if token_count > 1 { token_count - 1 } else { 1 };
    let tps = decode_tokens as f64 / decode_time;
    let prompt_tps = if ttft_val > 0.0 { num_prompt_tokens as f64 / ttft_val } else { 0.0 };

    println!();
    println!("╔══════════════════════════════════════════╗");
    println!("║        MLX-Engine Benchmark Results      ║");
    println!("╠══════════════════════════════════════════╣");
    println!("║  Prompt tokens:     {:>6}               ║", num_prompt_tokens);
    println!("║  Generated tokens:  {:>6}               ║", token_count);
    println!("║                                          ║");
    println!("║  TTFT (prefill):    {:>8.3}s             ║", ttft_val);
    println!("║  Prefill speed:     {:>8.1} tok/s       ║", prompt_tps);
    println!("║  Decode time:       {:>8.3}s             ║", decode_time);
    println!("║  Decode speed:      {:>8.1} tok/s       ║", tps);
    println!("║  Total time:        {:>8.3}s             ║", total_time);
    println!("╚══════════════════════════════════════════╝");

    Ok(())
}

// ── HTTP server ────────────────────────────────────────────────────────────────
//
// mlx-rs Array / Model are !Send (they rely on thread-local MLX stream state).
// We cannot pass them across thread boundaries.
//
// Architecture:
//   - One dedicated OS thread owns Model + Tokenizer for its entire lifetime.
//   - Axum handlers communicate with that thread via a std::sync::mpsc channel.
//   - Each request packages its parameters into an InferRequest and attaches a
//     one-shot reply channel; the inference thread sends back InferResponse.
//   - The Axum handler blocks its spawn_blocking worker while waiting for the reply.
//   - A Mutex<Sender<InferRequest>> in shared state serialises concurrent requests
//     (one at a time, MVP).

/// A single inference job sent from an HTTP handler to the model thread.
struct InferRequest {
    prompt: String,
    temp: f32,
    max_tokens: usize,
    reply: std::sync::mpsc::SyncSender<InferResponse>,
}

/// Result of a single inference job.
struct InferResponse {
    text: Result<GeneratedText>,
}

struct GeneratedText {
    content: String,
    prompt_tokens: usize,
    completion_tokens: usize,
}

/// Axum shared state: a mutex-protected sender to the inference thread.
#[derive(Clone)]
struct AppState {
    tx: Arc<Mutex<std::sync::mpsc::SyncSender<InferRequest>>>,
    model_name: Arc<String>,
}

// ── OpenAI-compatible request/response types ───────────────────────────────────

#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    #[allow(dead_code)]
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    stream: bool,
}

fn default_temperature() -> f32 { 0.7 }
fn default_max_tokens() -> usize { 256 }

#[derive(Debug, Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: UsageInfo,
}

#[derive(Debug, Serialize)]
struct ChatChoice {
    index: u32,
    message: AssistantMessage,
    finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
struct AssistantMessage {
    role: &'static str,
    content: String,
}

#[derive(Debug, Serialize)]
struct UsageInfo {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

// SSE streaming types
#[derive(Debug, Serialize)]
struct ChatCompletionChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChunkChoice>,
}

#[derive(Debug, Serialize)]
struct ChunkChoice {
    index: u32,
    delta: ChunkDelta,
    finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

// ── Chat template ──────────────────────────────────────────────────────────────

/// Format a list of chat messages into Qwen3's instruct prompt format.
///
/// ```text
/// <|im_start|>system
/// You are a helpful assistant.<|im_end|>
/// <|im_start|>user
/// Hello<|im_end|>
/// <|im_start|>assistant
/// ```
fn format_qwen3_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        prompt.push_str(&format!(
            "<|im_start|>{}\n{}<|im_end|>\n",
            msg.role, msg.content
        ));
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

// ── Axum error type ────────────────────────────────────────────────────────────

struct ApiError(anyhow::Error);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = serde_json::json!({
            "error": {
                "message": self.0.to_string(),
                "type": "internal_server_error",
            }
        });
        (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
    }
}

impl<E: Into<anyhow::Error>> From<E> for ApiError {
    fn from(e: E) -> Self {
        ApiError(e.into())
    }
}

// ── Handler ────────────────────────────────────────────────────────────────────

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    let prompt = format_qwen3_prompt(&req.messages);
    let temp = req.temperature;
    let max_tokens = req.max_tokens;
    let model_name = (*state.model_name).clone();
    let request_id = format!("chatcmpl-{}", Uuid::new_v4().simple());
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Clone the sender before moving into spawn_blocking.
    let tx = Arc::clone(&state.tx);

    // Run the blocking wait on a dedicated thread so we don't block the async runtime.
    let gen = tokio::task::spawn_blocking(move || {
        // One-shot synchronous channel for this request's reply.
        let (reply_tx, reply_rx) = std::sync::mpsc::sync_channel::<InferResponse>(1);

        let infer_req = InferRequest {
            prompt,
            temp,
            max_tokens,
            reply: reply_tx,
        };

        // Acquire the mutex only long enough to enqueue the request.
        {
            let sender = tx.lock().map_err(|_| anyhow::anyhow!("inference thread is gone"))?;
            sender
                .send(infer_req)
                .map_err(|_| anyhow::anyhow!("inference thread channel closed"))?;
        }

        // Wait for the inference result (blocks this spawn_blocking thread).
        reply_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("inference thread dropped reply channel"))
    })
    .await
    .map_err(|e| anyhow::anyhow!("spawn_blocking join error: {e}"))??;

    let gt = gen.text?;

    if req.stream {
        // --- SSE streaming response ---
        // We already have the full text; simulate streaming by sending it as a
        // single content chunk followed by [DONE].  A future version can wire
        // the inference loop to produce per-token SSE frames.
        use axum::body::Body;
        use axum::http::header;

        let id = request_id.clone();
        let chunk = ChatCompletionChunk {
            id: id.clone(),
            object: "chat.completion.chunk",
            created,
            model: model_name.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: Some("assistant"),
                    content: Some(gt.content.clone()),
                },
                finish_reason: None,
            }],
        };
        let done_chunk = ChatCompletionChunk {
            id,
            object: "chat.completion.chunk",
            created,
            model: model_name,
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta { role: None, content: None },
                finish_reason: Some("stop"),
            }],
        };

        let sse_body = format!(
            "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
            serde_json::to_string(&chunk)?,
            serde_json::to_string(&done_chunk)?
        );

        let response = axum::response::Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache")
            .header("X-Accel-Buffering", "no")
            .body(Body::from(sse_body))
            .map_err(|e| anyhow::anyhow!("failed to build SSE response: {e}"))?;

        return Ok(response);
    }

    // --- Non-streaming JSON response ---
    let response = ChatCompletionResponse {
        id: request_id,
        object: "chat.completion",
        created,
        model: model_name,
        choices: vec![ChatChoice {
            index: 0,
            message: AssistantMessage {
                role: "assistant",
                content: gt.content,
            },
            finish_reason: "stop",
        }],
        usage: UsageInfo {
            prompt_tokens: gt.prompt_tokens,
            completion_tokens: gt.completion_tokens,
            total_tokens: gt.prompt_tokens + gt.completion_tokens,
        },
    };

    Ok(Json(response).into_response())
}

// ── Inference thread loop ──────────────────────────────────────────────────────

/// Run the model inference loop on the calling thread.
///
/// This function never returns under normal operation — it blocks processing
/// requests until the channel sender side is dropped (server shutdown).
fn inference_loop(
    mut model: Model,
    tokenizer: tokenizers::Tokenizer,
    rx: std::sync::mpsc::Receiver<InferRequest>,
) {
    eprintln!("[serve] inference thread ready");

    for req in rx {
        let InferRequest { prompt, temp, max_tokens, reply } = req;

        // Count prompt tokens for usage reporting.
        let prompt_token_count = tokenizer
            .encode(prompt.as_str(), true)
            .map(|e| e.get_ids().len())
            .unwrap_or(0);

        let result = generate_tokens_server(&mut model, &tokenizer, &prompt, temp, max_tokens);

        let response = match result {
            Ok((text, completion_tokens)) => InferResponse {
                text: Ok(GeneratedText {
                    content: text,
                    prompt_tokens: prompt_token_count,
                    completion_tokens,
                }),
            },
            Err(e) => InferResponse { text: Err(e) },
        };

        // Best-effort send; if the HTTP handler timed out, we ignore the error.
        let _ = reply.send(response);
    }

    eprintln!("[serve] inference channel closed, thread exiting");
}

/// Like `generate_tokens` but returns `(text, completion_token_count)` and never
/// writes to stdout — suitable for server-side use where output is JSON.
fn generate_tokens_server(
    model: &mut Model,
    tokenizer: &tokenizers::Tokenizer,
    prompt: &str,
    temp: f32,
    max_tokens: usize,
) -> Result<(String, usize)> {
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    let num_layers = model.args.num_hidden_layers as usize;
    let mut cache: Vec<Option<ConcatKeyValueCache>> =
        (0..num_layers).map(|_| Some(ConcatKeyValueCache::new())).collect();

    let mut tokens: Vec<Array> = Vec::new();
    let mut decoded = String::new();
    let mut token_count = 0usize;

    // Prefill
    let prompt_tokens = Array::from(encoding.get_ids()).index(NewAxis);
    let logits = forward_step(model, &prompt_tokens, &mut cache)?;
    let mut y = sample(&logits, temp).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    eval(&[y.clone()]).map_err(|e| anyhow::anyhow!("{:?}", e))?;

    let first_id: u32 = y.item();
    tokens.push(y.clone());
    token_count += 1;

    if first_id == 151645 || first_id == 151643 {
        let slice: Vec<u32> = tokens.drain(..).map(|t| t.item::<u32>()).collect();
        let chunk = tokenizer.decode(&slice, true).map_err(|e| anyhow::anyhow!("{}", e))?;
        decoded.push_str(&chunk);
        return Ok((decoded, token_count));
    }

    // Decode loop
    loop {
        if token_count >= max_tokens {
            break;
        }

        let inputs = y.index((.., NewAxis));
        let logits = forward_step(model, &inputs, &mut cache)?;
        y = sample(&logits, temp).map_err(|e| anyhow::anyhow!("{:?}", e))?;
        tokens.push(y.clone());
        token_count += 1;

        if tokens.len() % 20 == 0 {
            eval(&tokens).map_err(|e| anyhow::anyhow!("{:?}", e))?;
            let slice: Vec<u32> = tokens.drain(..).map(|t| t.item::<u32>()).collect();
            let chunk = tokenizer
                .decode(&slice, true)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            decoded.push_str(&chunk);
        }

        let token_id: u32 = y.item();
        if token_id == 151645 || token_id == 151643 {
            break;
        }
    }

    if !tokens.is_empty() {
        eval(&tokens).map_err(|e| anyhow::anyhow!("{:?}", e))?;
        let slice: Vec<u32> = tokens.drain(..).map(|t| t.item::<u32>()).collect();
        let chunk = tokenizer
            .decode(&slice, true)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        decoded.push_str(&chunk);
    }

    Ok((decoded, token_count))
}

// ── serve subcommand entrypoint ────────────────────────────────────────────────

#[tokio::main]
async fn run_serve(model_path: &str, host: &str, port: u16) -> Result<()> {
    let (model, tokenizer, _path) = load_model_and_tokenizer(model_path)?;

    // Bounded channel so slow clients apply back-pressure.
    let (tx, rx) = std::sync::mpsc::sync_channel::<InferRequest>(4);

    // Spin up the dedicated inference thread.  It owns `model` and `tokenizer`
    // for its entire lifetime — they never cross thread boundaries.
    std::thread::Builder::new()
        .name("mlx-inference".into())
        .spawn(move || inference_loop(model, tokenizer, rx))
        .context("failed to spawn inference thread")?;

    let state = AppState {
        tx: Arc::new(Mutex::new(tx)),
        model_name: Arc::new(model_path.to_string()),
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
        .layer(cors);

    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("failed to bind to {addr}"))?;

    eprintln!(
        "[serve] mlx-engine v{} listening on http://{}",
        env!("CARGO_PKG_VERSION"),
        listener.local_addr()?
    );
    eprintln!("[serve] POST /v1/chat/completions");
    eprintln!("[serve] Press Ctrl+C to stop");

    axum::serve(listener, app).await.context("server error")?;

    Ok(())
}

// ── main ───────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Chat {
            model,
            temp,
            max_tokens,
        } => run_chat(&model, temp, max_tokens),
        Commands::Generate {
            model,
            prompt,
            temp,
            max_tokens,
        } => run_generate(&model, &prompt, temp, max_tokens),
        Commands::Bench {
            model,
            prompt,
            num_tokens,
        } => run_bench(&model, &prompt, num_tokens),
        Commands::Serve { model, host, port } => run_serve(&model, &host, port),
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── format_qwen3_prompt ──────────────────────────────────────────────────

    #[test]
    fn test_format_single_user_message() {
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "Hello".into(),
        }];
        let prompt = format_qwen3_prompt(&messages);
        assert_eq!(
            prompt,
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn test_format_system_plus_user() {
        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: "You are a helpful assistant.".into(),
            },
            ChatMessage {
                role: "user".into(),
                content: "What is 2+2?".into(),
            },
        ];
        let prompt = format_qwen3_prompt(&messages);
        assert!(prompt.starts_with("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"));
        assert!(prompt.contains("<|im_start|>user\nWhat is 2+2?<|im_end|>\n"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_format_multi_turn() {
        let messages = vec![
            ChatMessage { role: "user".into(), content: "Hi".into() },
            ChatMessage { role: "assistant".into(), content: "Hello!".into() },
            ChatMessage { role: "user".into(), content: "How are you?".into() },
        ];
        let prompt = format_qwen3_prompt(&messages);
        // All three turns should appear in order.
        let user_pos = prompt.find("<|im_start|>user\nHi").unwrap();
        let assistant_pos = prompt.find("<|im_start|>assistant\nHello!").unwrap();
        let user2_pos = prompt.find("<|im_start|>user\nHow are you?").unwrap();
        assert!(user_pos < assistant_pos);
        assert!(assistant_pos < user2_pos);
        // Must end with the open assistant turn.
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_format_empty_messages() {
        let messages: Vec<ChatMessage> = vec![];
        let prompt = format_qwen3_prompt(&messages);
        assert_eq!(prompt, "<|im_start|>assistant\n");
    }

    #[test]
    fn test_format_special_characters() {
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "What's 5 * 7?\n(show work)".into(),
        }];
        let prompt = format_qwen3_prompt(&messages);
        assert!(prompt.contains("What's 5 * 7?\n(show work)"));
    }

    // ── ChatCompletionRequest deserialization ────────────────────────────────

    #[test]
    fn test_request_defaults() {
        let json = r#"{"messages":[{"role":"user","content":"hi"}]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!((req.temperature - 0.7).abs() < f32::EPSILON);
        assert_eq!(req.max_tokens, 256);
        assert!(!req.stream);
        assert!(req.model.is_none());
    }

    #[test]
    fn test_request_explicit_fields() {
        let json = r#"{
            "model": "qwen3-4b",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.3,
            "max_tokens": 128,
            "stream": true
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model.as_deref(), Some("qwen3-4b"));
        assert!((req.temperature - 0.3).abs() < f32::EPSILON);
        assert_eq!(req.max_tokens, 128);
        assert!(req.stream);
        assert_eq!(req.messages[0].role, "user");
        assert_eq!(req.messages[0].content, "Hello");
    }

    // ── ChatCompletionResponse serialization ─────────────────────────────────

    #[test]
    fn test_response_serialization() {
        let response = ChatCompletionResponse {
            id: "chatcmpl-abc".into(),
            object: "chat.completion",
            created: 1_700_000_000,
            model: "qwen3-4b".into(),
            choices: vec![ChatChoice {
                index: 0,
                message: AssistantMessage {
                    role: "assistant",
                    content: "4".into(),
                },
                finish_reason: "stop",
            }],
            usage: UsageInfo {
                prompt_tokens: 10,
                completion_tokens: 1,
                total_tokens: 11,
            },
        };
        let json = serde_json::to_string(&response).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["message"]["role"], "assistant");
        assert_eq!(v["choices"][0]["message"]["content"], "4");
        assert_eq!(v["usage"]["total_tokens"], 11);
    }

    // ── resolve_model_path ───────────────────────────────────────────────────

    #[test]
    fn test_resolve_existing_path() {
        // The project root always exists; resolving it as a model path should
        // succeed and return that path verbatim.
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let result = resolve_model_path(manifest_dir);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), PathBuf::from(manifest_dir));
    }

    #[test]
    fn test_resolve_nonexistent_path() {
        let result = resolve_model_path("/absolutely/does/not/exist/model-xyz");
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("Model not found"));
    }

    // ── UsageInfo arithmetic ─────────────────────────────────────────────────

    #[test]
    fn test_usage_total() {
        let usage = UsageInfo {
            prompt_tokens: 42,
            completion_tokens: 13,
            total_tokens: 42 + 13,
        };
        assert_eq!(usage.total_tokens, 55);
    }

    // ── ChunkDelta skip_serializing_if ───────────────────────────────────────

    #[test]
    fn test_chunk_delta_omits_none_fields() {
        let delta = ChunkDelta { role: None, content: None };
        let json = serde_json::to_string(&delta).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(v.get("role").is_none(), "None role should be omitted");
        assert!(v.get("content").is_none(), "None content should be omitted");
    }

    #[test]
    fn test_chunk_delta_includes_some_fields() {
        let delta = ChunkDelta {
            role: Some("assistant"),
            content: Some("hello".into()),
        };
        let json = serde_json::to_string(&delta).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["role"], "assistant");
        assert_eq!(v["content"], "hello");
    }
}
