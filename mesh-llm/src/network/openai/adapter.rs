use crate::network::openai::schema;
use anyhow::{Context, Result};
use serde_json::Value;

pub(crate) fn parse_chat_request(value: &Value) -> Result<schema::ChatCompletionsRequest> {
    let request: schema::ChatCompletionsRequest = serde_json::from_value(value.clone())
        .context("parse typed /v1/chat/completions request")?;
    let _ = (
        request.model.as_deref(),
        request.messages.as_ref().map(Vec::len),
        request.stream,
        request.extra.len(),
    );
    if let Some(messages) = request.messages.as_ref() {
        for message in messages {
            let _ = (
                message.role.as_deref(),
                message.content.as_ref(),
                message.extra.len(),
            );
        }
    }
    Ok(request)
}

pub(crate) fn parse_responses_request(value: &Value) -> Result<schema::ResponsesRequest> {
    let request: schema::ResponsesRequest =
        serde_json::from_value(value.clone()).context("parse typed /v1/responses request")?;
    let _ = (
        request.model.as_deref(),
        request.input.as_ref(),
        request.instructions.as_deref(),
        request.messages.as_ref(),
        request.stream,
        request.extra.len(),
    );
    Ok(request)
}

pub(crate) fn parse_chat_stream_chunk(payload: &str) -> Result<schema::ChatCompletionStreamChunk> {
    let chunk: schema::ChatCompletionStreamChunk =
        serde_json::from_str(payload).context("parse typed upstream chat stream chunk")?;
    let _ = chunk.extra.len();
    if let Some(usage) = chunk.usage.as_ref() {
        let _ = usage.extra.len();
    }
    if let Some(choice) = chunk.choices.first() {
        let _ = (choice.finish_reason.as_deref(), choice.extra.len());
        if let Some(delta) = choice.delta.as_ref() {
            let _ = (delta.role.as_deref(), delta.extra.len());
        }
    }
    Ok(chunk)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_parse_chat_request_typed_with_extra_fields() {
        let body = json!({
            "model": "qwen",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
            "max_tokens": 64,
            "custom_field": "kept",
        });

        let parsed = parse_chat_request(&body).expect("chat parse should succeed");
        assert_eq!(parsed.model.as_deref(), Some("qwen"));
        assert_eq!(parsed.messages.as_ref().map(Vec::len), Some(1));
        assert_eq!(parsed.stream, Some(true));
        assert_eq!(
            parsed
                .extra
                .get("custom_field")
                .and_then(|value| value.as_str()),
            Some("kept")
        );
    }

    #[test]
    fn test_parse_responses_request_typed() {
        let body = json!({
            "model": "qwen",
            "instructions": "be concise",
            "input": "hello",
            "stream": false,
            "max_output_tokens": 128,
        });

        let parsed = parse_responses_request(&body).expect("responses parse should succeed");
        assert_eq!(parsed.model.as_deref(), Some("qwen"));
        assert_eq!(parsed.instructions.as_deref(), Some("be concise"));
        assert_eq!(parsed.stream, Some(false));
        assert!(parsed.extra.contains_key("max_output_tokens"));
    }

    #[test]
    fn test_parse_chat_stream_chunk_typed() {
        let payload = json!({
            "model": "qwen",
            "choices": [{
                "delta": {"role": "assistant", "content": "hello"},
                "finish_reason": null
            }],
            "usage": {"prompt_tokens": 12, "completion_tokens": 1, "total_tokens": 13}
        })
        .to_string();

        let parsed = parse_chat_stream_chunk(&payload).expect("stream chunk parse should succeed");
        assert_eq!(parsed.model.as_deref(), Some("qwen"));
        let delta = parsed
            .choices
            .first()
            .and_then(|choice| choice.delta.as_ref())
            .and_then(|delta| delta.content.as_deref());
        assert_eq!(delta, Some("hello"));
        assert_eq!(
            parsed.usage.as_ref().and_then(|usage| usage.total_tokens),
            Some(13)
        );
    }
}
