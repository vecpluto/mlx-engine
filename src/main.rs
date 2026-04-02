//! mlx-engine — Native Apple Silicon LLM inference. Single binary, zero Python.
//!
//! Entry point: CLI parsing and subcommand dispatch only.

mod bench;
mod inference;
mod model;
mod server;

use std::io::{self, Write};

use anyhow::Result;
use clap::{Parser, Subcommand};

use inference::generate_streaming;
use model::load_model_and_tokenizer;

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
        #[arg(
            short,
            long,
            default_value = "Write a short story about a robot learning to paint."
        )]
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

// ── Subcommand runners ─────────────────────────────────────────────────────────

fn run_chat(model_path: &str, temp: f32, max_tokens: usize) -> Result<()> {
    let (mut model, tokenizer, eos_tokens, _path) = load_model_and_tokenizer(model_path)?;

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

        let prompt = format!("<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n");

        print!("Assistant: ");
        io::stdout().flush()?;

        generate_streaming(&mut model, &tokenizer, &prompt, temp, max_tokens, &eos_tokens)?;
    }

    Ok(())
}

fn run_generate(model_path: &str, prompt: &str, temp: f32, max_tokens: usize) -> Result<()> {
    let (mut model, tokenizer, eos_tokens, _path) = load_model_and_tokenizer(model_path)?;
    generate_streaming(&mut model, &tokenizer, prompt, temp, max_tokens, &eos_tokens)?;
    Ok(())
}

fn run_bench_cmd(model_path: &str, prompt: &str, num_tokens: usize) -> Result<()> {
    let (mut model, tokenizer, eos_tokens, _path) = load_model_and_tokenizer(model_path)?;
    bench::run_bench(&mut model, &tokenizer, prompt, num_tokens, &eos_tokens)
}

// ── main ───────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Chat { model, temp, max_tokens } => run_chat(&model, temp, max_tokens),
        Commands::Generate { model, prompt, temp, max_tokens } => {
            run_generate(&model, &prompt, temp, max_tokens)
        }
        Commands::Bench { model, prompt, num_tokens } => run_bench_cmd(&model, &prompt, num_tokens),
        Commands::Serve { model, host, port } => {
            let (mdl, tokenizer, eos_tokens, _path) = load_model_and_tokenizer(&model)?;
            // Determine model family for prompt formatting in the server.
            let family = match &mdl {
                model::ModelWrapper::Qwen3(_) => model::ModelFamily::Qwen3,
                model::ModelWrapper::Llama(_) => model::ModelFamily::Llama,
            };
            tokio::runtime::Runtime::new()?.block_on(async {
                server::run_serve(mdl, tokenizer, eos_tokens, model, &host, port, family).await
            })
        }
    }
}
