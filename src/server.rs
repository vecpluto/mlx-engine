//! OpenAI-compatible HTTP API server with true per-token SSE streaming.
//!
//! mlx-rs `Array` / `Model` are `!Send` (they rely on thread-local MLX stream state).
//! We cannot pass them across thread boundaries.
//!
//! Architecture:
//!   - One dedicated OS thread owns `ModelWrapper` + `Tokenizer` for its entire lifetime.
//!   - Axum handlers communicate with that thread via a `std::sync::mpsc` channel.
//!   - For streaming requests, a `tokio::sync::mpsc` channel carries `StreamEvent` tokens
//!     from the inference thread back to the async Axum handler as they are produced.
//!   - For non-streaming requests, the inference thread sends back a complete `InferResponse`.
//!   - A `Mutex<Sender<InferRequest>>` in shared state serialises concurrent requests.

use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
    Router,
};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;
use tower_http::cors::{Any, CorsLayer};
use uuid::Uuid;

use crate::inference::generate;
use crate::model::{ModelFamily, ModelWrapper};

// ── Stream event ───────────────────────────────────────────────────────────────

/// Events sent from the inference thread to the async handler during streaming.
pub enum StreamEvent {
    /// A decoded text chunk (may be multiple tokens when flushed in batches).
    Token(String),
    /// Generation complete.
    Done {
        #[allow(dead_code)]
        prompt_tokens: usize,
        #[allow(dead_code)]
        completion_tokens: usize,
    },
    /// An error occurred.
    Error(String),
}

// ── Internal channel types ─────────────────────────────────────────────────────

/// A single inference job sent from an HTTP handler to the model thread.
struct InferRequest {
    prompt: String,
    temp: f32,
    max_tokens: usize,
    /// For streaming: a sender for per-token events. `None` for non-streaming.
    stream_tx: Option<tokio::sync::mpsc::UnboundedSender<StreamEvent>>,
    /// For non-streaming: a one-shot reply channel.
    reply: Option<std::sync::mpsc::SyncSender<InferResponse>>,
}

/// Result of a single non-streaming inference job.
struct InferResponse {
    text: Result<GeneratedText>,
}

struct GeneratedText {
    content: String,
    prompt_tokens: usize,
    completion_tokens: usize,
}

/// Axum shared state: a mutex-protected sender to the inference thread, plus model metadata.
#[derive(Clone)]
pub struct AppState {
    tx: Arc<Mutex<std::sync::mpsc::SyncSender<InferRequest>>>,
    model_name: Arc<String>,
    model_family: ModelFamily,
}

// ── OpenAI-compatible request/response types ───────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    #[allow(dead_code)]
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub stream: bool,
}

fn default_temperature() -> f32 {
    0.7
}
fn default_max_tokens() -> usize {
    256
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: UsageInfo,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: AssistantMessage,
    pub finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
pub struct AssistantMessage {
    pub role: &'static str,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct UsageInfo {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
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
pub struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

// ── Chat templates ─────────────────────────────────────────────────────────────

/// Format messages using Qwen3's ChatML instruct format.
///
/// ```text
/// <|im_start|>system
/// You are a helpful assistant.<|im_end|>
/// <|im_start|>user
/// Hello<|im_end|>
/// <|im_start|>assistant
/// ```
pub fn format_qwen3_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, msg.content));
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

/// Format messages using Llama 3's instruct format.
///
/// ```text
/// <|begin_of_text|><|start_header_id|>system<|end_header_id|>
///
/// You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>
///
/// Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>
///
/// ```
pub fn format_llama3_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::from("<|begin_of_text|>");
    for msg in messages {
        prompt.push_str(&format!(
            "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
            msg.role, msg.content
        ));
    }
    prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    prompt
}

/// Format messages according to the detected model family.
pub fn format_prompt(messages: &[ChatMessage], family: ModelFamily) -> String {
    match family {
        ModelFamily::Qwen3 => format_qwen3_prompt(messages),
        ModelFamily::Llama => format_llama3_prompt(messages),
    }
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
    let prompt = format_prompt(&req.messages, state.model_family);
    let temp = req.temperature;
    let max_tokens = req.max_tokens;
    let model_name = (*state.model_name).clone();
    let request_id = format!("chatcmpl-{}", Uuid::new_v4().simple());
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let is_stream = req.stream;

    let tx = Arc::clone(&state.tx);

    if is_stream {
        // True per-token SSE streaming.
        let (stream_tx, mut stream_rx) =
            tokio::sync::mpsc::unbounded_channel::<StreamEvent>();

        // Send the inference request with streaming channel.
        {
            let sender = tx.lock().map_err(|_| anyhow::anyhow!("inference thread is gone"))?;
            sender
                .send(InferRequest {
                    prompt,
                    temp,
                    max_tokens,
                    stream_tx: Some(stream_tx),
                    reply: None,
                })
                .map_err(|_| anyhow::anyhow!("inference thread channel closed"))?;
        }

        // Build an SSE body by bridging the tokio receiver into a stream.
        use axum::body::Body;
        use axum::http::header;
        use futures_util::stream::StreamExt;

        let id = request_id.clone();
        let model_name_clone = model_name.clone();

        let sse_stream = async_stream::stream! {
            // First chunk: role header
            let role_chunk = ChatCompletionChunk {
                id: id.clone(),
                object: "chat.completion.chunk",
                created,
                model: model_name_clone.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChunkDelta { role: Some("assistant"), content: None },
                    finish_reason: None,
                }],
            };
            if let Ok(json) = serde_json::to_string(&role_chunk) {
                yield Ok::<_, std::convert::Infallible>(
                    format!("data: {json}\n\n").into_bytes()
                );
            }

            loop {
                match stream_rx.recv().await {
                    Some(StreamEvent::Token(text)) => {
                        let chunk = ChatCompletionChunk {
                            id: id.clone(),
                            object: "chat.completion.chunk",
                            created,
                            model: model_name_clone.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: ChunkDelta { role: None, content: Some(text) },
                                finish_reason: None,
                            }],
                        };
                        if let Ok(json) = serde_json::to_string(&chunk) {
                            yield Ok::<_, std::convert::Infallible>(
                                format!("data: {json}\n\n").into_bytes()
                            );
                        }
                    }
                    Some(StreamEvent::Done { .. }) | None => {
                        let done_chunk = ChatCompletionChunk {
                            id: id.clone(),
                            object: "chat.completion.chunk",
                            created,
                            model: model_name_clone.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: ChunkDelta { role: None, content: None },
                                finish_reason: Some("stop"),
                            }],
                        };
                        if let Ok(json) = serde_json::to_string(&done_chunk) {
                            yield Ok::<_, std::convert::Infallible>(
                                format!("data: {json}\n\ndata: [DONE]\n\n").into_bytes()
                            );
                        }
                        break;
                    }
                    Some(StreamEvent::Error(msg)) => {
                        let err = serde_json::json!({ "error": { "message": msg } });
                        yield Ok::<_, std::convert::Infallible>(
                            format!("data: {}\n\n", err).into_bytes()
                        );
                        break;
                    }
                }
            }
        };

        let body = Body::from_stream(sse_stream.map(|r: Result<Vec<u8>, std::convert::Infallible>| r));

        let response = axum::response::Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache")
            .header("X-Accel-Buffering", "no")
            .body(body)
            .map_err(|e| anyhow::anyhow!("failed to build SSE response: {e}"))?;

        return Ok(response);
    }

    // Non-streaming: block until complete response is ready.
    let gen = tokio::task::spawn_blocking(move || {
        let (reply_tx, reply_rx) = std::sync::mpsc::sync_channel::<InferResponse>(1);

        let infer_req = InferRequest {
            prompt,
            temp,
            max_tokens,
            stream_tx: None,
            reply: Some(reply_tx),
        };

        {
            let sender = tx.lock().map_err(|_| anyhow::anyhow!("inference thread is gone"))?;
            sender
                .send(infer_req)
                .map_err(|_| anyhow::anyhow!("inference thread channel closed"))?;
        }

        reply_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("inference thread dropped reply channel"))
    })
    .await
    .map_err(|e| anyhow::anyhow!("spawn_blocking join error: {e}"))??;

    let gt = gen.text?;

    let response = ChatCompletionResponse {
        id: request_id,
        object: "chat.completion",
        created,
        model: model_name,
        choices: vec![ChatChoice {
            index: 0,
            message: AssistantMessage { role: "assistant", content: gt.content },
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
/// Blocks until the channel sender side is dropped (server shutdown).
fn inference_loop(
    mut model: ModelWrapper,
    tokenizer: Tokenizer,
    eos_tokens: Vec<u32>,
    rx: std::sync::mpsc::Receiver<InferRequest>,
) {
    eprintln!("[serve] inference thread ready");

    for req in rx {
        let InferRequest { prompt, temp, max_tokens, stream_tx, reply } = req;

        if let Some(tx) = stream_tx {
            // Streaming mode: send tokens as they are decoded.
            let prompt_token_count = tokenizer
                .encode(prompt.as_str(), true)
                .map(|e| e.get_ids().len())
                .unwrap_or(0);

            let result = generate(
                &mut model,
                &tokenizer,
                &prompt,
                temp,
                max_tokens,
                &eos_tokens,
                |chunk| {
                    // Ignore send errors (client may have disconnected).
                    let _ = tx.send(StreamEvent::Token(chunk.to_owned()));
                },
            );

            match result {
                Ok(out) => {
                    let _ = tx.send(StreamEvent::Done {
                        prompt_tokens: prompt_token_count,
                        completion_tokens: out.completion_tokens,
                    });
                }
                Err(e) => {
                    let _ = tx.send(StreamEvent::Error(e.to_string()));
                }
            }
        } else if let Some(reply_tx) = reply {
            // Non-streaming mode: collect and send full response.
            let prompt_token_count = tokenizer
                .encode(prompt.as_str(), true)
                .map(|e| e.get_ids().len())
                .unwrap_or(0);

            let result = generate(
                &mut model,
                &tokenizer,
                &prompt,
                temp,
                max_tokens,
                &eos_tokens,
                |_| {},
            );

            let response = match result {
                Ok(out) => InferResponse {
                    text: Ok(GeneratedText {
                        content: out.text,
                        prompt_tokens: prompt_token_count,
                        completion_tokens: out.completion_tokens,
                    }),
                },
                Err(e) => InferResponse { text: Err(e) },
            };

            let _ = reply_tx.send(response);
        }
    }

    eprintln!("[serve] inference channel closed, thread exiting");
}

// ── serve subcommand entrypoint ────────────────────────────────────────────────

/// Start the HTTP server.  Must be called inside a Tokio runtime.
pub async fn run_serve(
    model: ModelWrapper,
    tokenizer: Tokenizer,
    eos_tokens: Vec<u32>,
    model_name: String,
    host: &str,
    port: u16,
    model_family: ModelFamily,
) -> Result<()> {
    let (tx, rx) = std::sync::mpsc::sync_channel::<InferRequest>(4);

    std::thread::Builder::new()
        .name("mlx-inference".into())
        .spawn(move || inference_loop(model, tokenizer, eos_tokens, rx))
        .context("failed to spawn inference thread")?;

    let state = AppState {
        tx: Arc::new(Mutex::new(tx)),
        model_name: Arc::new(model_name),
        model_family,
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

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Qwen3 prompt formatting ────────────────────────────────────────────────

    #[test]
    fn test_format_qwen3_single_user_message() {
        let messages = vec![ChatMessage { role: "user".into(), content: "Hello".into() }];
        let prompt = format_qwen3_prompt(&messages);
        assert_eq!(prompt, "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n");
    }

    #[test]
    fn test_format_qwen3_system_plus_user() {
        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: "You are a helpful assistant.".into(),
            },
            ChatMessage { role: "user".into(), content: "What is 2+2?".into() },
        ];
        let prompt = format_qwen3_prompt(&messages);
        assert!(
            prompt.starts_with("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n")
        );
        assert!(prompt.contains("<|im_start|>user\nWhat is 2+2?<|im_end|>\n"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_format_qwen3_multi_turn() {
        let messages = vec![
            ChatMessage { role: "user".into(), content: "Hi".into() },
            ChatMessage { role: "assistant".into(), content: "Hello!".into() },
            ChatMessage { role: "user".into(), content: "How are you?".into() },
        ];
        let prompt = format_qwen3_prompt(&messages);
        let user_pos = prompt.find("<|im_start|>user\nHi").unwrap();
        let assistant_pos = prompt.find("<|im_start|>assistant\nHello!").unwrap();
        let user2_pos = prompt.find("<|im_start|>user\nHow are you?").unwrap();
        assert!(user_pos < assistant_pos);
        assert!(assistant_pos < user2_pos);
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_format_qwen3_empty_messages() {
        let messages: Vec<ChatMessage> = vec![];
        let prompt = format_qwen3_prompt(&messages);
        assert_eq!(prompt, "<|im_start|>assistant\n");
    }

    #[test]
    fn test_format_qwen3_special_characters() {
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "What's 5 * 7?\n(show work)".into(),
        }];
        let prompt = format_qwen3_prompt(&messages);
        assert!(prompt.contains("What's 5 * 7?\n(show work)"));
    }

    // ── Llama3 prompt formatting ───────────────────────────────────────────────

    #[test]
    fn test_format_llama3_single_user_message() {
        let messages = vec![ChatMessage { role: "user".into(), content: "Hello".into() }];
        let prompt = format_llama3_prompt(&messages);
        assert!(prompt.starts_with("<|begin_of_text|>"));
        assert!(prompt.contains("<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>"));
        assert!(prompt.ends_with(
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        ));
    }

    #[test]
    fn test_format_llama3_system_plus_user() {
        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: "You are a helpful assistant.".into(),
            },
            ChatMessage { role: "user".into(), content: "Hi".into() },
        ];
        let prompt = format_llama3_prompt(&messages);
        assert!(prompt.starts_with("<|begin_of_text|>"));
        assert!(prompt.contains(
            "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>"
        ));
        assert!(prompt.contains("<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>"));
        assert!(prompt.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn test_format_llama3_empty_messages() {
        let messages: Vec<ChatMessage> = vec![];
        let prompt = format_llama3_prompt(&messages);
        assert_eq!(
            prompt,
            "<|begin_of_text|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn test_format_prompt_dispatch() {
        let messages = vec![ChatMessage { role: "user".into(), content: "Hi".into() }];
        let qwen3 = format_prompt(&messages, ModelFamily::Qwen3);
        let llama = format_prompt(&messages, ModelFamily::Llama);
        assert!(qwen3.contains("<|im_start|>"));
        assert!(llama.contains("<|begin_of_text|>"));
    }

    // ── Request/response serialization ────────────────────────────────────────

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

    #[test]
    fn test_response_serialization() {
        let response = ChatCompletionResponse {
            id: "chatcmpl-abc".into(),
            object: "chat.completion",
            created: 1_700_000_000,
            model: "qwen3-4b".into(),
            choices: vec![ChatChoice {
                index: 0,
                message: AssistantMessage { role: "assistant", content: "4".into() },
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

    #[test]
    fn test_usage_total() {
        let usage =
            UsageInfo { prompt_tokens: 42, completion_tokens: 13, total_tokens: 42 + 13 };
        assert_eq!(usage.total_tokens, 55);
    }

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
        let delta = ChunkDelta { role: Some("assistant"), content: Some("hello".into()) };
        let json = serde_json::to_string(&delta).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["role"], "assistant");
        assert_eq!(v["content"], "hello");
    }
}
