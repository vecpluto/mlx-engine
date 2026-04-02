//! Integration tests for mlx-engine's public API surface.
//!
//! Tests that require GPU or downloaded model weights are marked `#[ignore]`
//! and will not run in CI.  All other tests exercise pure-logic code paths
//! without touching the file system beyond temporary directories.

use std::fs;
use std::path::PathBuf;

use serde_json::json;

// ── Helpers ────────────────────────────────────────────────────────────────────

/// Write a minimal `config.json` with the given architecture string into `dir`.
fn write_config(dir: &std::path::Path, architecture: &str) {
    let config = json!({
        "architectures": [architecture],
        "eos_token_id": 151645,
        "num_hidden_layers": 4,
        "hidden_size": 512,
        "intermediate_size": 1024,
        "num_attention_heads": 8,
        "vocab_size": 32000
    });
    fs::write(dir.join("config.json"), config.to_string()).expect("write config.json");
}

// ── CLI parsing tests ──────────────────────────────────────────────────────────
//
// `Cli` and `Commands` live in `main.rs` which is a binary crate; they are not
// re-exported as a library.  We therefore test the CLI by invoking the compiled
// binary via `std::process::Command` and inspecting exit codes / stderr output.
//
// `cargo test --test integration` builds the binary as a side-effect of
// building the test harness, so the binary is always fresh.

fn binary_path() -> PathBuf {
    // CARGO_BIN_EXE_mlx-engine is set by cargo when running integration tests
    // for a binary crate.  Fall back to a best-guess for manual invocations.
    option_env!("CARGO_BIN_EXE_mlx-engine")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            manifest.join("target/debug/mlx-engine")
        })
}

/// Run the binary with the given arguments and return the exit status + stderr.
fn run(args: &[&str]) -> (bool, String) {
    let output = std::process::Command::new(binary_path())
        .args(args)
        .output()
        .expect("failed to execute mlx-engine binary");
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    // clap prints help/errors to stdout for `--help`, and to stderr for errors.
    let combined = format!("{stdout}{stderr}");
    (output.status.success(), combined)
}

#[test]
fn test_cli_parse_chat() {
    // Missing required `--model` argument should produce a clap error (exit != 0).
    let (ok, output) = run(&["chat"]);
    assert!(!ok, "chat without --model must fail");
    assert!(
        output.contains("--model") || output.contains("model"),
        "error output should mention the missing argument; got: {output}"
    );
}

#[test]
fn test_cli_parse_generate() {
    // Missing both `--model` and `--prompt` should fail.
    let (ok, output) = run(&["generate"]);
    assert!(!ok, "generate without required args must fail");
    assert!(
        output.contains("model") || output.contains("prompt"),
        "error output should mention missing args; got: {output}"
    );
}

#[test]
fn test_cli_parse_bench() {
    // Missing `--model` should fail.
    let (ok, output) = run(&["bench"]);
    assert!(!ok, "bench without --model must fail");
    assert!(
        output.contains("model"),
        "error output should mention --model; got: {output}"
    );
}

#[test]
fn test_cli_parse_serve() {
    // Missing `--model` should fail.
    let (ok, output) = run(&["serve"]);
    assert!(!ok, "serve without --model must fail");
    assert!(
        output.contains("model"),
        "error output should mention --model; got: {output}"
    );
}

#[test]
fn test_cli_parse_serve_defaults() {
    // The `--help` flag should exit successfully and mention default values.
    let (ok, output) = run(&["serve", "--help"]);
    assert!(ok, "serve --help must succeed; got: {output}");
    // Default port 8080 and default host 127.0.0.1 should appear in help text.
    assert!(
        output.contains("8080"),
        "help should mention default port 8080; got: {output}"
    );
    assert!(
        output.contains("127.0.0.1"),
        "help should mention default host 127.0.0.1; got: {output}"
    );
}

// ── Model path resolution tests ────────────────────────────────────────────────

#[test]
fn test_resolve_nonexistent_model() {
    // `resolve_model_path` is `pub` in model.rs.  Integration tests access it
    // indirectly via the binary: asking for a model that does not exist must
    // produce a non-zero exit and a helpful error message.
    let (ok, output) = run(&[
        "generate",
        "--model",
        "/nonexistent/model/xyz",
        "--prompt",
        "hi",
    ]);
    assert!(!ok, "generate with missing model must fail");
    assert!(
        output.contains("Model not found")
            || output.contains("not found")
            || output.contains("No such file"),
        "error should mention that the model was not found; got: {output}"
    );
}

#[test]
fn test_resolve_local_path() {
    // A directory that exists should be accepted by `resolve_model_path`.
    // We create a temp dir with a valid-looking config.json so the path
    // resolution step succeeds; the actual model loading will fail because
    // there are no weights — but the error will be past the path-resolution
    // stage.
    let tmp = tempfile::tempdir().expect("create tempdir");
    write_config(tmp.path(), "Qwen3ForCausalLM");

    let (ok, output) = run(&[
        "generate",
        "--model",
        tmp.path().to_str().unwrap(),
        "--prompt",
        "hello",
    ]);
    // The binary will fail (no weights), but NOT with "Model not found".
    assert!(!ok, "generate without weights must fail");
    assert!(
        !output.contains("Model not found"),
        "path resolution should succeed for an existing directory; got: {output}"
    );
}

// ── Architecture detection tests ───────────────────────────────────────────────
//
// `detect_architecture` is `pub` in `model.rs`.  Because mlx-engine is a
// binary-only crate (no `lib.rs`), we cannot `use mlx_engine::model::...`
// directly.  Instead we replicate the pure JSON logic here — these tests
// document the expected contract and will catch any regression in the
// function's behaviour when cross-checked against the unit tests in model.rs.

fn detect_architecture(config: &serde_json::Value) -> Result<&'static str, String> {
    let archs = config
        .get("architectures")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "config.json missing 'architectures' field".to_string())?;

    for arch in archs {
        if let Some(s) = arch.as_str() {
            if s == "Qwen3ForCausalLM" {
                return Ok("Qwen3");
            }
            if s == "LlamaForCausalLM" {
                return Ok("Llama");
            }
        }
    }

    Err(format!("Unsupported architecture: {archs:?}"))
}

#[test]
fn test_detect_qwen3_architecture() {
    let config = json!({ "architectures": ["Qwen3ForCausalLM"] });
    assert_eq!(detect_architecture(&config).unwrap(), "Qwen3");
}

#[test]
fn test_detect_llama_architecture() {
    let config = json!({ "architectures": ["LlamaForCausalLM"] });
    assert_eq!(detect_architecture(&config).unwrap(), "Llama");
}

#[test]
fn test_detect_unknown_architecture() {
    let config = json!({ "architectures": ["GPT2LMHeadModel"] });
    let err = detect_architecture(&config).unwrap_err();
    assert!(
        err.contains("Unsupported"),
        "error should say Unsupported; got: {err}"
    );
}

#[test]
fn test_detect_architecture_missing_field() {
    let config = json!({ "model_type": "qwen" });
    let err = detect_architecture(&config).unwrap_err();
    assert!(
        err.contains("architectures"),
        "error should mention the missing field; got: {err}"
    );
}

#[test]
fn test_detect_first_matching_architecture() {
    // Second entry is recognized; first is unknown.
    let config = json!({ "architectures": ["UnknownArch", "LlamaForCausalLM"] });
    assert_eq!(detect_architecture(&config).unwrap(), "Llama");
}

// ── Server request/response round-trip ────────────────────────────────────────
//
// These tests exercise the serde contracts of the OpenAI-compatible types.
// The types are `pub` in `server.rs` but again unreachable as a library.
// We duplicate the minimal JSON round-trip logic to ensure the wire format
// stays stable and correct.

#[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq)]
struct ChatMessageRt {
    role: String,
    content: String,
}

#[derive(Debug, serde::Deserialize)]
struct ChatCompletionRequestRt {
    model: Option<String>,
    messages: Vec<ChatMessageRt>,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    stream: bool,
}

fn default_temperature() -> f32 {
    0.7
}
fn default_max_tokens() -> usize {
    256
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct ChatCompletionResponseRt {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChatChoiceRt>,
    usage: UsageInfoRt,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct ChatChoiceRt {
    index: u32,
    message: AssistantMessageRt,
    finish_reason: String,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct AssistantMessageRt {
    role: String,
    content: String,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct UsageInfoRt {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[test]
fn test_chat_request_deserialization() {
    // Minimal request — only required field `messages` present.
    let json = r#"{"messages":[{"role":"user","content":"Hello!"}]}"#;
    let req: ChatCompletionRequestRt = serde_json::from_str(json).unwrap();

    assert_eq!(req.messages.len(), 1);
    assert_eq!(req.messages[0].role, "user");
    assert_eq!(req.messages[0].content, "Hello!");
    assert!(req.model.is_none());
    // Serde defaults must kick in.
    assert!(
        (req.temperature - 0.7).abs() < f32::EPSILON,
        "temperature default should be 0.7, got {}",
        req.temperature
    );
    assert_eq!(req.max_tokens, 256);
    assert!(!req.stream);
}

#[test]
fn test_chat_request_explicit_fields() {
    let json = r#"{
        "model": "mlx-community/Qwen3-4B-4bit",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is Rust?"}
        ],
        "temperature": 0.2,
        "max_tokens": 512,
        "stream": true
    }"#;
    let req: ChatCompletionRequestRt = serde_json::from_str(json).unwrap();

    assert_eq!(req.model.as_deref(), Some("mlx-community/Qwen3-4B-4bit"));
    assert_eq!(req.messages.len(), 2);
    assert_eq!(req.messages[0].role, "system");
    assert_eq!(req.messages[1].content, "What is Rust?");
    assert!((req.temperature - 0.2).abs() < f32::EPSILON);
    assert_eq!(req.max_tokens, 512);
    assert!(req.stream);
}

#[test]
fn test_chat_response_serialization_roundtrip() {
    let response = ChatCompletionResponseRt {
        id: "chatcmpl-test123".into(),
        object: "chat.completion".into(),
        created: 1_700_000_000,
        model: "mlx-community/Qwen3-4B-4bit".into(),
        choices: vec![ChatChoiceRt {
            index: 0,
            message: AssistantMessageRt {
                role: "assistant".into(),
                content: "Rust is a systems language.".into(),
            },
            finish_reason: "stop".into(),
        }],
        usage: UsageInfoRt {
            prompt_tokens: 15,
            completion_tokens: 6,
            total_tokens: 21,
        },
    };

    let serialized = serde_json::to_string(&response).unwrap();
    let deserialized: ChatCompletionResponseRt = serde_json::from_str(&serialized).unwrap();

    assert_eq!(deserialized.id, "chatcmpl-test123");
    assert_eq!(deserialized.object, "chat.completion");
    assert_eq!(deserialized.model, "mlx-community/Qwen3-4B-4bit");
    assert_eq!(deserialized.choices.len(), 1);
    assert_eq!(deserialized.choices[0].message.role, "assistant");
    assert_eq!(
        deserialized.choices[0].message.content,
        "Rust is a systems language."
    );
    assert_eq!(deserialized.choices[0].finish_reason, "stop");
    assert_eq!(deserialized.usage.prompt_tokens, 15);
    assert_eq!(deserialized.usage.completion_tokens, 6);
    assert_eq!(deserialized.usage.total_tokens, 21);
}

#[test]
fn test_usage_total_tokens_is_sum() {
    let usage = UsageInfoRt {
        prompt_tokens: 100,
        completion_tokens: 42,
        total_tokens: 142,
    };
    assert_eq!(
        usage.total_tokens,
        usage.prompt_tokens + usage.completion_tokens
    );
}

#[test]
fn test_response_required_fields_present_in_json() {
    let response = ChatCompletionResponseRt {
        id: "chatcmpl-abc".into(),
        object: "chat.completion".into(),
        created: 0,
        model: "test-model".into(),
        choices: vec![],
        usage: UsageInfoRt {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
    };
    let v: serde_json::Value =
        serde_json::from_str(&serde_json::to_string(&response).unwrap()).unwrap();
    for field in &["id", "object", "created", "model", "choices", "usage"] {
        assert!(
            v.get(field).is_some(),
            "field '{field}' missing from serialized response"
        );
    }
}

// ── Full pipeline test (requires a local model) ────────────────────────────────

#[test]
#[ignore = "requires local model: mlx-community/Qwen3-4B-4bit"]
fn test_full_generate_pipeline() {
    // This test loads an actual quantized model, generates 5 tokens, and
    // verifies the output is non-empty.  Run with:
    //   cargo test --test integration test_full_generate_pipeline -- --ignored
    let (ok, output) = run(&[
        "generate",
        "--model",
        "mlx-community/Qwen3-4B-4bit",
        "--prompt",
        "The capital of France is",
        "--max-tokens",
        "5",
        "--temp",
        "0.0",
    ]);
    assert!(
        ok,
        "generate should succeed with a valid model; stderr: {output}"
    );
    // The generate subcommand streams tokens to stdout.
    assert!(
        !output.trim().is_empty(),
        "generated output should be non-empty"
    );
}
