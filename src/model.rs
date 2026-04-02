//! Model loading: path resolution, architecture detection, weight loading, quantization.
//!
//! Supports Qwen3 and Llama architectures, auto-detected from `config.json`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Instant;

use anyhow::{Context, Result};
use mlx_lm::cache::ConcatKeyValueCache;
use mlx_lm::models::llama::{self as llama, get_llama_model_args, load_llama_tokenizer};
use mlx_lm::models::qwen3::{self as qwen3, get_qwen3_model_args, load_qwen3_tokenizer};
use mlx_rs::module::{Module, ModuleParameters, ModuleParametersExt};
use mlx_rs::nn::QuantizedEmbedding;
use mlx_rs::quantization::{MaybeQuantized, Quantizable};
use mlx_rs::Array;
use tokenizers::Tokenizer;

// ── Architecture detection ─────────────────────────────────────────────────────

/// Which model family was loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    Qwen3,
    Llama,
}

/// Detect model architecture from `config.json`'s `"architectures"` array.
///
/// # Errors
///
/// Returns an error if the field is missing or contains no recognized architecture.
///
/// # Examples
///
/// ```no_run
/// let family = detect_architecture(&config_json)?;
/// ```
pub fn detect_architecture(config: &serde_json::Value) -> Result<ModelFamily> {
    let archs = config
        .get("architectures")
        .and_then(|v| v.as_array())
        .context("config.json missing 'architectures' field")?;

    for arch in archs {
        if let Some(s) = arch.as_str() {
            if s == "Qwen3ForCausalLM" {
                return Ok(ModelFamily::Qwen3);
            }
            if s == "LlamaForCausalLM" {
                return Ok(ModelFamily::Llama);
            }
        }
    }

    anyhow::bail!(
        "Unsupported architecture in config.json. Supported: Qwen3ForCausalLM, LlamaForCausalLM. Found: {:?}",
        archs
    )
}

// ── ModelWrapper ───────────────────────────────────────────────────────────────

/// Unified wrapper over all supported model types.
///
/// Methods mirror those called by `inference.rs` and `bench.rs`.
pub enum ModelWrapper {
    Qwen3(qwen3::Model),
    Llama(llama::Model),
}

impl ModelWrapper {
    /// Number of transformer layers (used to size the KV cache vector).
    pub fn num_hidden_layers(&self) -> usize {
        match self {
            ModelWrapper::Qwen3(m) => m.args.num_hidden_layers as usize,
            ModelWrapper::Llama(m) => m.args.num_hidden_layers as usize,
        }
    }

    /// Single-step forward pass returning `(batch, vocab_size)` logits for the last token.
    pub fn forward_step(
        &mut self,
        inputs: &Array,
        cache: &mut Vec<Option<ConcatKeyValueCache>>,
    ) -> Result<Array> {
        use mlx_rs::ops::indexing::IndexOp;

        let logits = match self {
            ModelWrapper::Qwen3(m) => {
                let input = qwen3::ModelInput {
                    inputs,
                    mask: None,
                    cache,
                };
                m.forward(input).map_err(|e| anyhow::anyhow!("{:?}", e))?
            }
            ModelWrapper::Llama(m) => {
                let input = llama::ModelInput {
                    inputs,
                    mask: None,
                    cache,
                };
                m.forward(input).map_err(|e| anyhow::anyhow!("{:?}", e))?
            }
        };
        // logits shape: (batch, seq_len, vocab_size) — select last position.
        Ok(logits.index((.., -1i32, ..)))
    }

    /// Quantize the model in-place (consumes and replaces self).
    pub fn try_into_quantized(self, group_size: i32, bits: i32) -> Result<Self> {
        match self {
            ModelWrapper::Qwen3(m) => {
                let q = m
                    .try_into_quantized(group_size, bits)
                    .map_err(|e| anyhow::anyhow!("{:?}", e))?;
                Ok(ModelWrapper::Qwen3(q))
            }
            ModelWrapper::Llama(m) => {
                let q = m
                    .try_into_quantized(group_size, bits)
                    .map_err(|e| anyhow::anyhow!("{:?}", e))?;
                Ok(ModelWrapper::Llama(q))
            }
        }
    }

    /// Load safetensors weights from a file (non-quantized path).
    pub fn load_safetensors(&mut self, path: &Path) -> Result<()> {
        match self {
            ModelWrapper::Qwen3(m) => m
                .load_safetensors(path)
                .map_err(|e| anyhow::anyhow!("{:?}", e)),
            ModelWrapper::Llama(m) => m
                .load_safetensors(path)
                .map_err(|e| anyhow::anyhow!("{:?}", e)),
        }
    }
}

// ── Model path resolution ──────────────────────────────────────────────────────

pub fn resolve_model_path(model: &str) -> Result<PathBuf> {
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
pub fn collect_safetensors_weights(path: &Path) -> Result<HashMap<String, Array>> {
    let weights_index_path = path.join("model.safetensors.index.json");
    let weights_path = path.join("model.safetensors");

    let mut all_weights: HashMap<String, Array> = HashMap::new();

    if weights_index_path.exists() {
        let json = std::fs::read_to_string(&weights_index_path)?;
        // Use serde_json::Value to parse the weight map generically.
        let parsed: serde_json::Value = serde_json::from_str(&json)?;
        let weight_files: std::collections::HashSet<String> = parsed
            .get("weight_map")
            .and_then(|v| v.as_object())
            .map(|m| {
                m.values()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        for weight_file in weight_files {
            let wf = path.join(&weight_file);
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
pub fn patch_quantized_embedding(
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

/// Load pre-quantized safetensors weights into an already-quantized Qwen3 model.
///
/// See inline comments for the two-phase approach.
pub fn load_prequantized_weights_qwen3(
    model: &mut qwen3::Model,
    mut safetensors_weights: HashMap<String, Array>,
) -> Result<()> {
    load_prequantized_weights_inner(model, "model.embed_tokens", &mut safetensors_weights, |m| {
        match &mut m.model.embed_tokens {
            MaybeQuantized::Quantized(qe) => Some(qe),
            MaybeQuantized::Original(_) => None,
        }
    })
}

/// Load pre-quantized safetensors weights into an already-quantized Llama model.
pub fn load_prequantized_weights_llama(
    model: &mut llama::Model,
    mut safetensors_weights: HashMap<String, Array>,
) -> Result<()> {
    load_prequantized_weights_inner(model, "model.embed_tokens", &mut safetensors_weights, |m| {
        match &mut m.model.embed_tokens {
            MaybeQuantized::Quantized(qe) => Some(qe),
            MaybeQuantized::Original(_) => None,
        }
    })
}

/// Generic inner implementation shared by both model families.
///
/// # Type parameters
///
/// - `M`: model type implementing `ModuleParameters + ModuleParametersExt + mlx_rs::module::Module<_>`
/// - `F`: closure to access the `QuantizedEmbedding` field (returns `None` for non-quantized)
fn load_prequantized_weights_inner<M, F>(
    model: &mut M,
    embed_prefix: &str,
    safetensors_weights: &mut HashMap<String, Array>,
    get_embed: F,
) -> Result<()>
where
    M: ModuleParameters + ModuleParametersExt,
    F: FnOnce(&mut M) -> Option<&mut QuantizedEmbedding>,
{
    // Phase 1: patch embed_tokens directly (QuantizedEmbedding has no #[param] in v0.25.3).
    if let Some(qe) = get_embed(model) {
        patch_quantized_embedding(qe, embed_prefix, safetensors_weights);
        eprintln!("  Patched embed_tokens directly (QuantizedEmbedding).");
    }

    // Phase 2: load remaining keys via the standard parameter API.
    let model_keys: std::collections::HashSet<String> = model
        .parameters_mut()
        .flatten()
        .keys()
        .map(|k| k.to_string())
        .collect();

    let mut remapped: HashMap<Rc<str>, Array> = HashMap::new();
    let mut unmatched_count = 0usize;

    for (st_key, array) in safetensors_weights.drain() {
        if model_keys.contains(&st_key) {
            remapped.insert(Rc::from(st_key.as_str()), array);
            continue;
        }

        // Quantized linear layers: remap `<prefix>.weight` → `<prefix>.inner.weight`.
        if st_key.ends_with(".weight") {
            let inner_key = format!("{}.inner.weight", &st_key[..st_key.len() - ".weight".len()]);
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

// ── Main loading entry point ───────────────────────────────────────────────────

/// Load model, tokenizer, and return EOS token IDs read from config.json.
///
/// Returns `(model, tokenizer, eos_tokens, model_path)`.
/// Architecture is auto-detected from `config.json`'s `"architectures"` field.
/// Supported: `Qwen3ForCausalLM`, `LlamaForCausalLM`.
pub fn load_model_and_tokenizer(
    model_path: &str,
) -> Result<(ModelWrapper, Tokenizer, Vec<u32>, PathBuf)> {
    let path = resolve_model_path(model_path)?;
    eprintln!("Loading model from: {}", path.display());

    let start = Instant::now();

    // Read config for architecture detection, quantization, and EOS tokens.
    let config_path = path.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)?;
    let config_json: serde_json::Value = serde_json::from_str(&config_str)?;

    let family = detect_architecture(&config_json)?;
    eprintln!("Detected architecture: {:?}", family);

    let eos_tokens = collect_eos_tokens(&config_json, family);

    let quantization = config_json.get("quantization");
    let is_quantized = quantization.is_some();

    if is_quantized {
        let q = quantization.unwrap();
        let gs = q.get("group_size").and_then(|v| v.as_i64()).unwrap_or(64);
        let b = q.get("bits").and_then(|v| v.as_i64()).unwrap_or(4);
        eprintln!("Detected pre-quantized model: {}-bit, group_size={}", b, gs);
    }

    let mut model_wrapper = match family {
        ModelFamily::Qwen3 => {
            let model_args =
                get_qwen3_model_args(&path).context("Failed to read Qwen3 config.json")?;
            let m = qwen3::Model::new(model_args).context("Failed to create Qwen3 model")?;
            ModelWrapper::Qwen3(m)
        }
        ModelFamily::Llama => {
            let model_args =
                get_llama_model_args(&path).context("Failed to read Llama config.json")?;
            let m = llama::Model::new(model_args).context("Failed to create Llama model")?;
            ModelWrapper::Llama(m)
        }
    };

    if is_quantized {
        let q = quantization.unwrap();
        let gs = q.get("group_size").and_then(|v| v.as_i64()).unwrap_or(64) as i32;
        let b = q.get("bits").and_then(|v| v.as_i64()).unwrap_or(4) as i32;

        eprintln!(
            "Quantizing model structure ({}-bit, group_size={})...",
            b, gs
        );
        model_wrapper = model_wrapper
            .try_into_quantized(gs, b)
            .context("Failed to create quantized model structure")?;

        eprintln!("Loading pre-quantized weights...");
        let weights = collect_safetensors_weights(&path)?;

        match &mut model_wrapper {
            ModelWrapper::Qwen3(m) => load_prequantized_weights_qwen3(m, weights)?,
            ModelWrapper::Llama(m) => load_prequantized_weights_llama(m, weights)?,
        }
    } else {
        // Non-quantized model: standard load.
        let weights_index_path = path.join("model.safetensors.index.json");
        let weights_path = path.join("model.safetensors");

        let weight_files: Vec<PathBuf> = if weights_index_path.exists() {
            let json = std::fs::read_to_string(&weights_index_path)?;
            let parsed: serde_json::Value = serde_json::from_str(&json)?;
            let files: std::collections::HashSet<String> = parsed
                .get("weight_map")
                .and_then(|v| v.as_object())
                .map(|m| {
                    m.values()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            files.into_iter().map(|f| path.join(f)).collect()
        } else {
            vec![weights_path]
        };

        for wf in weight_files {
            model_wrapper.load_safetensors(&wf)?;
        }
    }

    let tokenizer = match family {
        ModelFamily::Qwen3 => {
            load_qwen3_tokenizer(&path).context("Failed to load Qwen3 tokenizer")?
        }
        ModelFamily::Llama => {
            load_llama_tokenizer(&path).context("Failed to load Llama tokenizer")?
        }
    };

    let load_time = start.elapsed();
    eprintln!("Model loaded in {:.2}s", load_time.as_secs_f64());

    Ok((model_wrapper, tokenizer, eos_tokens, path))
}

/// Extract EOS token IDs from config.json.
///
/// The `eos_token_id` field can be a single integer or an array of integers.
/// Falls back to model-family-specific well-known values if the field is absent.
fn collect_eos_tokens(config: &serde_json::Value, family: ModelFamily) -> Vec<u32> {
    if let Some(eos) = config.get("eos_token_id") {
        if let Some(id) = eos.as_u64() {
            return vec![id as u32];
        }
        if let Some(arr) = eos.as_array() {
            let ids: Vec<u32> = arr
                .iter()
                .filter_map(|v| v.as_u64().map(|n| n as u32))
                .collect();
            if !ids.is_empty() {
                return ids;
            }
        }
    }
    // Fallback to well-known EOS tokens per family.
    match family {
        ModelFamily::Qwen3 => vec![151645, 151643],
        // Llama 3 EOS token IDs.
        ModelFamily::Llama => vec![128001, 128009],
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_existing_path() {
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

    #[test]
    fn test_detect_qwen3_architecture() {
        let config = serde_json::json!({ "architectures": ["Qwen3ForCausalLM"] });
        let family = detect_architecture(&config).unwrap();
        assert_eq!(family, ModelFamily::Qwen3);
    }

    #[test]
    fn test_detect_llama_architecture() {
        let config = serde_json::json!({ "architectures": ["LlamaForCausalLM"] });
        let family = detect_architecture(&config).unwrap();
        assert_eq!(family, ModelFamily::Llama);
    }

    #[test]
    fn test_detect_unsupported_architecture() {
        let config = serde_json::json!({ "architectures": ["GPT2LMHeadModel"] });
        let result = detect_architecture(&config);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("Unsupported architecture"));
    }

    #[test]
    fn test_detect_missing_architectures_field() {
        let config = serde_json::json!({ "model_type": "qwen" });
        let result = detect_architecture(&config);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("missing 'architectures'"));
    }

    #[test]
    fn test_collect_eos_tokens_single() {
        let config = serde_json::json!({ "eos_token_id": 151645 });
        let tokens = collect_eos_tokens(&config, ModelFamily::Qwen3);
        assert_eq!(tokens, vec![151645u32]);
    }

    #[test]
    fn test_collect_eos_tokens_array() {
        let config = serde_json::json!({ "eos_token_id": [151643, 151645] });
        let tokens = collect_eos_tokens(&config, ModelFamily::Qwen3);
        assert_eq!(tokens, vec![151643u32, 151645u32]);
    }

    #[test]
    fn test_collect_eos_tokens_fallback_qwen3() {
        let config = serde_json::json!({});
        let tokens = collect_eos_tokens(&config, ModelFamily::Qwen3);
        assert!(tokens.contains(&151645));
        assert!(tokens.contains(&151643));
    }

    #[test]
    fn test_collect_eos_tokens_fallback_llama() {
        let config = serde_json::json!({});
        let tokens = collect_eos_tokens(&config, ModelFamily::Llama);
        assert!(tokens.contains(&128001));
        assert!(tokens.contains(&128009));
    }

    #[test]
    fn test_detect_first_matching_architecture() {
        // If multiple architectures are listed, first recognized one wins.
        let config = serde_json::json!({ "architectures": ["UnknownArch", "LlamaForCausalLM"] });
        let family = detect_architecture(&config).unwrap();
        assert_eq!(family, ModelFamily::Llama);
    }
}
