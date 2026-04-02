//! Unified token generation: forward pass, decode loop, performance reporting.

use std::io::{self, Write};
use std::time::Instant;

use anyhow::Result;
use mlx_lm::cache::ConcatKeyValueCache;
use mlx_lm::models::qwen3::sample;
use mlx_rs::transforms::eval;
use mlx_rs::Array;
use tokenizers::Tokenizer;

use crate::model::ModelWrapper;

// ── Public output type ─────────────────────────────────────────────────────────

/// Statistics and decoded text returned by [`generate`].
pub struct GenerateOutput {
    /// Decoded text produced by the model.
    pub text: String,
    /// Number of tokens in the input prompt.
    pub prompt_tokens: usize,
    /// Number of tokens generated (including the first token from prefill).
    pub completion_tokens: usize,
    /// Time-to-first-token in seconds (prefill latency).
    pub ttft: Option<f64>,
    /// Wall-clock time for the full generation in seconds.
    pub total_time: f64,
}

// ── Forward pass ───────────────────────────────────────────────────────────────

/// Single-step forward pass returning shape `(batch, vocab_size)` logits for the last token.
///
/// Delegates to [`ModelWrapper::forward_step`] which handles the shape slicing.
pub fn forward_step(
    model: &mut ModelWrapper,
    inputs: &Array,
    cache: &mut Vec<Option<ConcatKeyValueCache>>,
) -> Result<Array> {
    model.forward_step(inputs, cache)
}

// ── Unified generation ─────────────────────────────────────────────────────────

/// Generate tokens from `prompt`, calling `on_chunk` for each decoded token.
///
/// This single function covers both interactive (streaming to stdout) and server
/// (silent, collect into string) use cases via the `on_chunk` callback.
///
/// # Examples
///
/// Streaming to stdout:
/// ```no_run
/// # use mlx_engine::inference::{generate, GenerateOutput};
/// let out = generate(&mut model, &tokenizer, "Hello", 0.7, 256, &[151645, 151643], |chunk| {
///     print!("{chunk}");
///     true
/// })?;
/// println!("\n{:.1} tok/s", out.completion_tokens as f64 / out.total_time);
/// ```
///
/// Silent (server mode):
/// ```no_run
/// let out = generate(&mut model, &tokenizer, &prompt, temp, max_tokens, &eos_tokens, |_| true)?;
/// ```
pub fn generate(
    model: &mut ModelWrapper,
    tokenizer: &Tokenizer,
    prompt: &str,
    temp: f32,
    max_tokens: usize,
    eos_tokens: &[u32],
    mut on_chunk: impl FnMut(&str) -> bool,
) -> Result<GenerateOutput> {
    use mlx_rs::ops::indexing::{IndexOp, NewAxis};

    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    let num_prompt_tokens = encoding.get_ids().len();

    if max_tokens == 0 {
        return Ok(GenerateOutput {
            text: String::new(),
            prompt_tokens: num_prompt_tokens,
            completion_tokens: 0,
            ttft: None,
            total_time: 0.0,
        });
    }

    let num_layers = model.num_hidden_layers();
    let mut cache: Vec<Option<ConcatKeyValueCache>> = (0..num_layers)
        .map(|_| Some(ConcatKeyValueCache::new()))
        .collect();

    let mut pending_eval: Vec<Array> = Vec::new();
    let mut decoded = String::new();
    let gen_start = Instant::now();
    let mut token_count = 0usize;

    // --- Prefill ---
    let prompt_tokens = Array::from(encoding.get_ids()).index(NewAxis);
    let logits = forward_step(model, &prompt_tokens, &mut cache)?;

    let mut y = sample(&logits, temp).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let first_id: u32 = y.item();
    let ttft = Some(gen_start.elapsed().as_secs_f64());
    pending_eval.push(y.clone());
    token_count += 1;
    let chunk = decode_token(tokenizer, first_id)?;
    if !on_chunk(&chunk) {
        decoded.push_str(&chunk);
        return Ok(GenerateOutput {
            text: decoded,
            prompt_tokens: num_prompt_tokens,
            completion_tokens: token_count,
            ttft,
            total_time: gen_start.elapsed().as_secs_f64(),
        });
    }
    decoded.push_str(&chunk);

    // Check EOS immediately after prefill.
    if eos_tokens.contains(&first_id) {
        eval_pending(&mut pending_eval)?;
        return Ok(GenerateOutput {
            text: decoded,
            prompt_tokens: num_prompt_tokens,
            completion_tokens: token_count,
            ttft,
            total_time: gen_start.elapsed().as_secs_f64(),
        });
    }

    // --- Decode loop ---
    loop {
        if token_count >= max_tokens {
            break;
        }

        let inputs = y.index((.., NewAxis));
        let logits = forward_step(model, &inputs, &mut cache)?;
        y = sample(&logits, temp).map_err(|e| anyhow::anyhow!("{:?}", e))?;

        let token_id: u32 = y.item();
        pending_eval.push(y.clone());
        token_count += 1;
        let chunk = decode_token(tokenizer, token_id)?;
        if !on_chunk(&chunk) {
            decoded.push_str(&chunk);
            return Ok(GenerateOutput {
                text: decoded,
                prompt_tokens: num_prompt_tokens,
                completion_tokens: token_count,
                ttft,
                total_time: gen_start.elapsed().as_secs_f64(),
            });
        }
        decoded.push_str(&chunk);

        // Periodically force MLX lazy evaluation while still streaming each token.
        if pending_eval.len().is_multiple_of(20) {
            eval_pending(&mut pending_eval)?;
        }

        if eos_tokens.contains(&token_id) {
            break;
        }
    }

    if !pending_eval.is_empty() {
        eval_pending(&mut pending_eval)?;
    }

    Ok(GenerateOutput {
        text: decoded,
        prompt_tokens: num_prompt_tokens,
        completion_tokens: token_count,
        ttft,
        total_time: gen_start.elapsed().as_secs_f64(),
    })
}

// ── Helpers ────────────────────────────────────────────────────────────────────

/// Force evaluation of pending token arrays, draining `tokens` in place.
fn eval_pending(tokens: &mut Vec<Array>) -> Result<()> {
    // eval requires &Array items; collect refs first.
    let refs: Vec<&Array> = tokens.iter().collect();
    eval(refs).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    tokens.clear();
    Ok(())
}

fn decode_token(tokenizer: &Tokenizer, token_id: u32) -> Result<String> {
    tokenizer
        .decode(&[token_id], true)
        .map_err(|e| anyhow::anyhow!("{}", e))
}

// ── Performance reporting ──────────────────────────────────────────────────────

/// Print a compact performance summary to stderr.
pub fn print_perf_stats(out: &GenerateOutput) {
    eprintln!();
    eprintln!("─── Performance ───────────────────────");
    eprintln!("  Prompt tokens:  {}", out.prompt_tokens);
    if let Some(t) = out.ttft {
        eprintln!("  TTFT:           {:.3}s", t);
        let decode_time = out.total_time - t;
        if decode_time > 0.0 && out.completion_tokens > 1 {
            let tps = (out.completion_tokens - 1) as f64 / decode_time;
            eprintln!("  Decode tokens:  {}", out.completion_tokens);
            eprintln!("  Decode time:    {:.3}s", decode_time);
            eprintln!("  Decode speed:   {:.1} tok/s", tps);
        }
    }
    eprintln!("  Total time:     {:.3}s", out.total_time);
    eprintln!("───────────────────────────────────────");
}

/// Generate tokens with streaming to stdout and a trailing newline.
///
/// Wraps [`generate`] with an `on_chunk` that writes to stdout and flushes.
pub fn generate_streaming(
    model: &mut ModelWrapper,
    tokenizer: &Tokenizer,
    prompt: &str,
    temp: f32,
    max_tokens: usize,
    eos_tokens: &[u32],
) -> Result<GenerateOutput> {
    let out = generate(
        model,
        tokenizer,
        prompt,
        temp,
        max_tokens,
        eos_tokens,
        |chunk| {
            print!("{chunk}");
            let _ = io::stdout().flush();
            true
        },
    )?;
    print_perf_stats(&out);
    println!();
    Ok(out)
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    // Integration tests that run model inference require downloaded weights and
    // are too slow for CI.  The logic covered here is EOS token checking and
    // per-token decoding, both of which need an actual tokenizer to test
    // meaningfully.  See the server module tests for pure-logic unit tests.
}
