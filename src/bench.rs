//! Benchmark subcommand: measures prefill and decode throughput.

use anyhow::Result;
use mlx_lm::cache::ConcatKeyValueCache;
use mlx_lm::models::qwen3::sample;
use mlx_rs::ops::indexing::{IndexOp, NewAxis};
use mlx_rs::transforms::eval;
use mlx_rs::Array;
use std::time::Instant;
use tokenizers::Tokenizer;

use crate::inference::forward_step;
use crate::model::ModelWrapper;

/// Run a benchmark and print a formatted results table.
pub fn run_bench(
    model: &mut ModelWrapper,
    tokenizer: &Tokenizer,
    prompt: &str,
    num_tokens: usize,
    eos_tokens: &[u32],
) -> Result<()> {
    eprintln!("\n=== MLX-Engine Benchmark ===\n");
    eprintln!("Prompt: \"{}\"", &prompt[..prompt.len().min(80)]);
    eprintln!("Target tokens: {}\n", num_tokens);

    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    let prompt_array = Array::from(encoding.get_ids()).index(NewAxis);
    let num_prompt_tokens = encoding.get_ids().len();

    let num_layers = model.num_hidden_layers();
    let mut cache: Vec<Option<ConcatKeyValueCache>> = (0..num_layers)
        .map(|_| Some(ConcatKeyValueCache::new()))
        .collect();

    let mut token_count = 0usize;
    let gen_start = Instant::now();

    // Prefill
    let logits = forward_step(model, &prompt_array, &mut cache)?;
    let mut y = sample(&logits, 0.0).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    eval(&[y.clone()]).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let ttft_val = gen_start.elapsed().as_secs_f64();
    token_count += 1;

    let mut batch: Vec<Array> = vec![y.clone()];

    // Decode loop
    loop {
        if token_count >= num_tokens {
            break;
        }
        let token_id: u32 = y.item();
        if eos_tokens.contains(&token_id) {
            break;
        }

        let inputs = y.index((.., NewAxis));
        let logits = forward_step(model, &inputs, &mut cache)?;
        y = sample(&logits, 0.0).map_err(|e| anyhow::anyhow!("{:?}", e))?;
        token_count += 1;
        batch.push(y.clone());

        if batch.len().is_multiple_of(20) {
            eval(&batch).map_err(|e| anyhow::anyhow!("{:?}", e))?;
            batch.clear();
        }
    }

    if !batch.is_empty() {
        eval(&batch).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    }

    let total_time = gen_start.elapsed().as_secs_f64();
    let decode_time = total_time - ttft_val;
    let decode_tokens = if token_count > 1 { token_count - 1 } else { 1 };
    let tps = decode_tokens as f64 / decode_time;
    let prompt_tps = if ttft_val > 0.0 {
        num_prompt_tokens as f64 / ttft_val
    } else {
        0.0
    };

    println!();
    println!("╔══════════════════════════════════════════╗");
    println!("║        MLX-Engine Benchmark Results      ║");
    println!("╠══════════════════════════════════════════╣");
    println!(
        "║  Prompt tokens:     {:>6}               ║",
        num_prompt_tokens
    );
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

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    // Benchmark tests require downloaded model weights and are exercised via
    // the runtime verification step (`cargo run -- bench ...`).
}
