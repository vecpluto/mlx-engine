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
}

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
    let mut ttft: Option<f64> = None;
    let mut token_count = 0usize;

    // --- Prefill ---
    let prompt_tokens = Array::from(encoding.get_ids()).index(NewAxis);
    let logits = forward_step(model, &prompt_tokens, &mut cache)?;

    // Sample first token; logits shape is (batch, vocab_size).
    let mut y = sample(&logits, temp).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    // y shape: (batch,) after argmax with keep_dims=false

    eval(&[y.clone()]).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    ttft = Some(gen_start.elapsed().as_secs_f64());

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
    let mut ttft: Option<f64> = None;

    // Prefill
    let logits = forward_step(&mut model, &prompt_tokens, &mut cache)?;
    let mut y = sample(&logits, 0.0).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    eval(&[y.clone()]).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    ttft = Some(gen_start.elapsed().as_secs_f64());
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
    }
}
