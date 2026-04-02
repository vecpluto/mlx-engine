//! Model loading: path resolution, weight loading, quantization handling.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Instant;

use anyhow::{Context, Result};
use mlx_lm::models::qwen3::{get_qwen3_model_args, load_qwen3_tokenizer, Model, WeightMap};
use mlx_rs::module::{ModuleParameters, ModuleParametersExt};
use mlx_rs::nn::QuantizedEmbedding;
use mlx_rs::quantization::{MaybeQuantized, Quantizable};
use mlx_rs::Array;
use tokenizers::Tokenizer;

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
pub fn load_prequantized_weights(
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

/// Load model, tokenizer, and return EOS token IDs read from config.json.
///
/// Returns `(model, tokenizer, eos_tokens, model_path)`.
/// `eos_tokens` is populated from the `eos_token_id` field in config.json;
/// it may contain one or more token IDs depending on the model config.
pub fn load_model_and_tokenizer(model_path: &str) -> Result<(Model, Tokenizer, Vec<u32>, PathBuf)> {
    let path = resolve_model_path(model_path)?;
    eprintln!("Loading model from: {}", path.display());

    let start = Instant::now();

    // Read config to detect quantization settings and EOS tokens.
    let model_args = get_qwen3_model_args(&path).context("Failed to read config.json")?;
    let config_path = path.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)?;
    let config_json: serde_json::Value = serde_json::from_str(&config_str)?;

    // Collect EOS token IDs from config.
    let eos_tokens = collect_eos_tokens(&config_json);

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

    Ok((model, tokenizer, eos_tokens, path))
}

/// Extract EOS token IDs from config.json.
///
/// The `eos_token_id` field can be a single integer or an array of integers.
/// Falls back to Qwen3's well-known values `[151645, 151643]` if the field is absent.
fn collect_eos_tokens(config: &serde_json::Value) -> Vec<u32> {
    if let Some(eos) = config.get("eos_token_id") {
        if let Some(id) = eos.as_u64() {
            return vec![id as u32];
        }
        if let Some(arr) = eos.as_array() {
            let ids: Vec<u32> = arr.iter().filter_map(|v| v.as_u64().map(|n| n as u32)).collect();
            if !ids.is_empty() {
                return ids;
            }
        }
    }
    // Fallback to Qwen3 well-known EOS tokens.
    vec![151645, 151643]
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
    fn test_collect_eos_tokens_single() {
        let config = serde_json::json!({ "eos_token_id": 151645 });
        let tokens = collect_eos_tokens(&config);
        assert_eq!(tokens, vec![151645u32]);
    }

    #[test]
    fn test_collect_eos_tokens_array() {
        let config = serde_json::json!({ "eos_token_id": [151643, 151645] });
        let tokens = collect_eos_tokens(&config);
        assert_eq!(tokens, vec![151643u32, 151645u32]);
    }

    #[test]
    fn test_collect_eos_tokens_fallback() {
        let config = serde_json::json!({});
        let tokens = collect_eos_tokens(&config);
        // Should fall back to Qwen3 defaults.
        assert!(tokens.contains(&151645));
        assert!(tokens.contains(&151643));
    }
}
