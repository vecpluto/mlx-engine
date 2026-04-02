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

/// Generate tokens from `prompt`, calling `on_chunk` for each decoded text chunk.
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
/// })?;
/// println!("\n{:.1} tok/s", out.completion_tokens as f64 / out.total_time);
/// ```
///
/// Silent (server mode):
/// ```no_run
/// let out = generate(&mut model, &tokenizer, &prompt, temp, max_tokens, &eos_tokens, |_| {})?;
/// ```
pub fn generate(
    model: &mut ModelWrapper,
    tokenizer: &Tokenizer,
    prompt: &str,
    temp: f32,
    max_tokens: usize,
    eos_tokens: &[u32],
    mut on_chunk: impl FnMut(&str),
) -> Result<GenerateOutput> {
    use mlx_rs::ops::indexing::{IndexOp, NewAxis};

    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    let num_prompt_tokens = encoding.get_ids().len();

    let num_layers = model.num_hidden_layers();
    let mut cache: Vec<Option<ConcatKeyValueCache>> =
        (0..num_layers).map(|_| Some(ConcatKeyValueCache::new())).collect();

    let mut pending: Vec<Array> = Vec::new();
    let mut decoded = String::new();
    let gen_start = Instant::now();
    let mut token_count = 0usize;

    // --- Prefill ---
    let prompt_tokens = Array::from(encoding.get_ids()).index(NewAxis);
    let logits = forward_step(model, &prompt_tokens, &mut cache)?;

    let mut y = sample(&logits, temp).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    eval(&[y.clone()]).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let ttft = Some(gen_start.elapsed().as_secs_f64());

    let first_id: u32 = y.item();
    pending.push(y.clone());
    token_count += 1;

    // Check EOS immediately after prefill.
    if eos_tokens.contains(&first_id) {
        let chunk = flush_tokens(tokenizer, &mut pending)?;
        on_chunk(&chunk);
        decoded.push_str(&chunk);
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

        pending.push(y.clone());
        token_count += 1;

        // Periodically eval and stream decoded chunks to the caller.
        if pending.len().is_multiple_of(20) {
            let chunk = flush_tokens(tokenizer, &mut pending)?;
            on_chunk(&chunk);
            decoded.push_str(&chunk);
        }

        let token_id: u32 = y.item();
        if eos_tokens.contains(&token_id) {
            break;
        }
    }

    // Flush any remaining tokens.
    if !pending.is_empty() {
        let chunk = flush_tokens(tokenizer, &mut pending)?;
        on_chunk(&chunk);
        decoded.push_str(&chunk);
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

/// Eval and decode a batch of token arrays, draining `tokens` in place.
fn flush_tokens(tokenizer: &Tokenizer, tokens: &mut Vec<Array>) -> Result<String> {
    // eval requires &Array items; collect refs first.
    let refs: Vec<&Array> = tokens.iter().collect();
    eval(refs).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let ids: Vec<u32> = tokens.drain(..).map(|t| t.item::<u32>()).collect();
    tokenizer.decode(&ids, true).map_err(|e| anyhow::anyhow!("{}", e))
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
    let out = generate(model, tokenizer, prompt, temp, max_tokens, eos_tokens, |chunk| {
        print!("{chunk}");
        let _ = io::stdout().flush();
    })?;
    print_perf_stats(&out);
    println!();
    Ok(out)
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    // Integration tests that run model inference require downloaded weights and
    // are too slow for CI.  The logic covered here is EOS token checking and
    // the flush_tokens helper, both of which need an actual tokenizer to test
    // meaningfully.  See the server module tests for pure-logic unit tests.
}
