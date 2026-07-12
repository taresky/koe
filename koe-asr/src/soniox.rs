use crate::config::AsrConfig;
use crate::error::{AsrError, Result};
use crate::event::AsrEvent;
use crate::provider::AsrProvider;
use futures_util::{SinkExt, StreamExt};
use std::collections::VecDeque;
use tokio::time::{timeout, Duration};
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};

type WsStream = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;

const DEFAULT_URL: &str = "wss://stt-rt.soniox.com/transcribe-websocket";
const DEFAULT_MODEL: &str = "stt-rt-v5";

/// Endpoint-detection marker (auto speech-end). Never part of transcript text.
const END_TOKEN: &str = "<end>";
/// Manual-finalization marker (`{"type":"finalize"}` response). Never part of text.
const FINALIZED_TOKEN: &str = "<fin>";
const FINALIZE_MESSAGE: &str = r#"{"type":"finalize"}"#;

/// Soniox real-time streaming ASR provider.
///
/// Protocol (WebSocket):
/// 1. Connect to `wss://stt-rt.soniox.com/transcribe-websocket`
/// 2. Send a JSON config message (api_key, model, audio format, …)
/// 3. Stream raw PCM audio as binary frames
/// 4. On stop: send `{"type":"finalize"}` then an empty text frame (end-of-audio)
/// 5. Receive token responses until `finished: true`
///
/// Special tokens from endpoint detection / finalization (`<end>`, `<fin>`) are
/// control signals only and must never appear in the transcript.
///
/// Config field mapping (via shared `AsrConfig`):
/// - `api_key` / `access_key` → Soniox API key
/// - `app_key` → model name (default `stt-rt-v5`)
/// - `url` → WebSocket endpoint
/// - `language` → language_hints (e.g. `zh-CN` → `zh`, empty/auto → none)
/// - `hotwords` → context.terms
pub struct SonioxAsrProvider {
    ws: Option<WsStream>,
    input_finished: bool,
    pending_events: VecDeque<AsrEvent>,
    /// Concatenation of all finalized token texts (control tokens excluded).
    final_text: String,
    /// Last full interim text we emitted (final + non-final tokens).
    last_interim: String,
    /// Whether we have already emitted `AsrEvent::Final` for this connection.
    emitted_final: bool,
}

impl SonioxAsrProvider {
    pub fn new() -> Self {
        Self {
            ws: None,
            input_finished: false,
            pending_events: VecDeque::new(),
            final_text: String::new(),
            last_interim: String::new(),
            emitted_final: false,
        }
    }

    /// Soniox control tokens used for endpoint / finalize signaling.
    fn is_control_token(text: &str) -> bool {
        text == END_TOKEN || text == FINALIZED_TOKEN
    }

    fn resolve_api_key(config: &AsrConfig) -> Result<String> {
        if !config.api_key.is_empty() {
            return Ok(config.api_key.clone());
        }
        if !config.access_key.is_empty() {
            return Ok(config.access_key.clone());
        }
        Err(AsrError::Connection("API key is required".into()))
    }

    fn resolve_url(config: &AsrConfig) -> String {
        if config.url.is_empty() {
            DEFAULT_URL.to_string()
        } else {
            config.url.clone()
        }
    }

    fn resolve_model(config: &AsrConfig) -> String {
        if config.app_key.is_empty() {
            DEFAULT_MODEL.to_string()
        } else {
            config.app_key.clone()
        }
    }

    /// Map config language codes (e.g. `zh-CN`, `en-US`) to Soniox ISO 639-1 hints.
    fn language_hints(config: &AsrConfig) -> Vec<String> {
        let Some(lang) = config.language.as_ref() else {
            return Vec::new();
        };
        let lang = lang.trim();
        if lang.is_empty() || lang.eq_ignore_ascii_case("auto") {
            return Vec::new();
        }
        // Accept "zh-CN" / "en_US" / "ja" → primary subtag lowercased.
        let primary = lang
            .split(['-', '_'])
            .next()
            .unwrap_or(lang)
            .to_ascii_lowercase();
        if primary.is_empty() {
            Vec::new()
        } else {
            vec![primary]
        }
    }

    fn build_config_message(config: &AsrConfig, api_key: &str) -> serde_json::Value {
        let model = Self::resolve_model(config);
        let hints = Self::language_hints(config);

        let mut body = serde_json::json!({
            "api_key": api_key,
            "model": model,
            "audio_format": "pcm_s16le",
            "sample_rate": config.sample_rate_hz,
            "num_channels": 1,
            "enable_endpoint_detection": true,
        });

        if !hints.is_empty() {
            body["language_hints"] = serde_json::json!(hints);
        } else {
            // No explicit language → let the model identify languages.
            body["enable_language_identification"] = serde_json::json!(true);
        }

        if !config.hotwords.is_empty() {
            body["context"] = serde_json::json!({
                "terms": config.hotwords,
            });
        }

        body
    }

    /// Parse a Soniox response into ASR events.
    ///
    /// Token accumulation:
    /// - Control tokens (`<end>`, `<fin>`) are skipped (not transcript text)
    /// - Tokens with `is_final: true` are appended to `final_text`
    /// - Non-final tokens form the provisional suffix for interim display
    /// - `finished: true` yields a Final event (not Closed — session owns close)
    fn parse_response(&mut self, text: &str) -> Result<Vec<AsrEvent>> {
        log::debug!("[Soniox ASR] Received: {}", text);

        let json: serde_json::Value = serde_json::from_str(text)
            .map_err(|e| AsrError::Protocol(format!("parse response: {e}")))?;

        // Error response
        if json.get("error_code").is_some() || json.get("error_type").is_some() {
            let msg = json
                .get("error_message")
                .and_then(|v| v.as_str())
                .or_else(|| json.get("error_type").and_then(|v| v.as_str()))
                .unwrap_or("Unknown Soniox error");
            log::error!("[Soniox ASR] Error: {}", msg);
            return Ok(vec![AsrEvent::Error(msg.to_string())]);
        }

        let mut events = Vec::new();
        let mut non_final = String::new();
        let mut added_final = false;

        if let Some(tokens) = json.get("tokens").and_then(|t| t.as_array()) {
            for token in tokens {
                let token_text = token
                    .get("text")
                    .and_then(|t| t.as_str())
                    .unwrap_or("");
                if token_text.is_empty() {
                    continue;
                }
                // Endpoint / finalize markers are signals, not speech text.
                if Self::is_control_token(token_text) {
                    log::debug!("[Soniox ASR] Skipping control token: {}", token_text);
                    continue;
                }
                // Skip pure translation tokens if present
                if token
                    .get("translation_status")
                    .and_then(|v| v.as_str())
                    == Some("translation")
                {
                    continue;
                }

                let is_final = token
                    .get("is_final")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                if is_final {
                    self.final_text.push_str(token_text);
                    added_final = true;
                } else {
                    non_final.push_str(token_text);
                }
            }
        }

        let interim = format!("{}{}", self.final_text, non_final);
        if !interim.is_empty() && interim != self.last_interim {
            self.last_interim = interim.clone();
            events.push(AsrEvent::Interim(interim));
        }

        if added_final && !self.final_text.is_empty() {
            // Definite captures progressive finalized segments for the overlay /
            // transcript aggregator.
            events.push(AsrEvent::Definite(self.final_text.clone()));
        }

        let finished = json
            .get("finished")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if finished {
            self.push_final_event(&mut events);
        }

        Ok(events)
    }

    /// Emit a single Final event from accumulated text (idempotent).
    ///
    /// Always emits once so `wait_for_final` can unblock, even when the
    /// transcript is empty (silent take).
    fn push_final_event(&mut self, events: &mut Vec<AsrEvent>) {
        if self.emitted_final {
            return;
        }
        self.emitted_final = true;
        log::info!("[Soniox ASR] Final: {}", self.final_text);
        events.push(AsrEvent::Final(self.final_text.clone()));
    }

    fn queue_final_if_needed(&mut self) {
        if self.emitted_final {
            return;
        }
        self.emitted_final = true;
        log::info!("[Soniox ASR] Final (on close): {}", self.final_text);
        self.pending_events
            .push_back(AsrEvent::Final(self.final_text.clone()));
    }
}

impl Default for SonioxAsrProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl AsrProvider for SonioxAsrProvider {
    async fn connect(&mut self, config: &AsrConfig) -> Result<()> {
        let api_key = Self::resolve_api_key(config)?;
        let ws_url = Self::resolve_url(config);
        let model = Self::resolve_model(config);

        log::info!(
            "[Soniox ASR] Connecting: url={}, model={}, sample_rate={}",
            ws_url,
            model,
            config.sample_rate_hz
        );

        let mut request = ws_url
            .into_client_request()
            .map_err(|e| AsrError::Connection(format!("invalid URL: {e}")))?;

        // Prefer custom headers when provided; otherwise send Bearer auth as a
        // secondary channel (docs accept api_key in the first JSON message).
        if config.custom_headers.is_empty() {
            request.headers_mut().insert(
                "Authorization",
                format!("Bearer {api_key}")
                    .parse()
                    .map_err(|_| AsrError::Connection("invalid api_key".into()))?,
            );
        } else {
            for (key, value) in &config.custom_headers {
                let header_name: tokio_tungstenite::tungstenite::http::HeaderName = key
                    .parse()
                    .map_err(|_| AsrError::Connection(format!("invalid header name: {key}")))?;
                request.headers_mut().insert(
                    header_name,
                    value.parse().map_err(|_| {
                        AsrError::Connection(format!("invalid header value for {key}"))
                    })?,
                );
            }
        }

        let (ws_stream, response) =
            timeout(Duration::from_millis(config.connect_timeout_ms), async {
                connect_async(request)
                    .await
                    .map_err(|e| AsrError::Connection(e.to_string()))
            })
            .await
            .map_err(|_| AsrError::Connection("connection timed out".into()))??;

        log::info!("[Soniox ASR] WebSocket connected: {}", response.status());
        self.ws = Some(ws_stream);

        let config_msg = Self::build_config_message(config, &api_key);
        let config_text = serde_json::to_string(&config_msg)
            .map_err(|e| AsrError::Protocol(format!("serialize config: {e}")))?;

        if let Some(ref mut ws) = self.ws {
            ws.send(Message::Text(config_text.into()))
                .await
                .map_err(|e| AsrError::Protocol(format!("send config: {e}")))?;
        }

        self.final_text.clear();
        self.last_interim.clear();
        self.input_finished = false;
        self.emitted_final = false;
        self.pending_events.push_back(AsrEvent::Connected);

        log::info!("[Soniox ASR] Configured and ready");
        Ok(())
    }

    async fn send_audio(&mut self, frame: &[u8]) -> Result<()> {
        if frame.is_empty() {
            return Ok(());
        }
        if let Some(ref mut ws) = self.ws {
            ws.send(Message::Binary(frame.to_vec().into()))
                .await
                .map_err(|e| AsrError::Protocol(format!("send audio: {e}")))?;
            Ok(())
        } else {
            Err(AsrError::Connection("not connected".into()))
        }
    }

    async fn finish_input(&mut self) -> Result<()> {
        if self.input_finished {
            return Ok(());
        }
        self.input_finished = true;

        // Push-to-talk stop: finalize first so trailing non-final tokens are
        // promoted quickly, then end the audio stream.
        if let Some(ref mut ws) = self.ws {
            ws.send(Message::Text(FINALIZE_MESSAGE.to_string().into()))
                .await
                .map_err(|e| AsrError::Protocol(format!("send finalize: {e}")))?;
            log::info!("[Soniox ASR] Finalize sent");

            // Empty text frame signals end-of-audio to Soniox.
            ws.send(Message::Text(String::new().into()))
                .await
                .map_err(|e| AsrError::Protocol(format!("send end-of-audio: {e}")))?;
            log::info!("[Soniox ASR] End-of-audio sent");
        }
        Ok(())
    }

    async fn next_event(&mut self) -> Result<AsrEvent> {
        if let Some(event) = self.pending_events.pop_front() {
            return Ok(event);
        }

        loop {
            let msg = self
                .ws
                .as_mut()
                .ok_or_else(|| AsrError::Connection("not connected".into()))?
                .next()
                .await;

            match msg {
                Some(Ok(Message::Text(text))) => {
                    let events = self.parse_response(&text)?;
                    self.pending_events.extend(events);
                    if let Some(event) = self.pending_events.pop_front() {
                        return Ok(event);
                    }
                    log::debug!("[Soniox ASR] Skipping text frame with no parseable events");
                }
                Some(Ok(Message::Close(frame))) => {
                    let reason = frame
                        .as_ref()
                        .map(|f| format!("code={}, reason={:?}", f.code, f.reason));
                    // Promote accumulated text if the server closed without
                    // sending finished:true (e.g. after finalize).
                    self.queue_final_if_needed();
                    self.pending_events.push_back(AsrEvent::Closed(reason));
                    return Ok(self.pending_events.pop_front().unwrap());
                }
                Some(Ok(Message::Binary(data))) => {
                    log::debug!("[Soniox ASR] Skipping binary frame ({} bytes)", data.len());
                }
                Some(Ok(frame)) => {
                    log::debug!("[Soniox ASR] Skipping frame: {:?}", frame);
                }
                Some(Err(e)) => return Err(AsrError::Protocol(e.to_string())),
                None => {
                    self.queue_final_if_needed();
                    self.pending_events
                        .push_back(AsrEvent::Closed(Some("WebSocket stream ended".into())));
                    return Ok(self.pending_events.pop_front().unwrap());
                }
            }
        }
    }

    async fn close(&mut self) -> Result<()> {
        if let Some(mut ws) = self.ws.take() {
            let _ = ws.close(None).await;
        }
        self.pending_events.clear();
        self.final_text.clear();
        self.last_interim.clear();
        self.input_finished = false;
        self.emitted_final = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_creation() {
        let p = SonioxAsrProvider::new();
        assert!(p.ws.is_none());
        assert!(!p.input_finished);
        assert!(p.final_text.is_empty());
    }

    #[test]
    fn language_hints_maps_locale() {
        let config = AsrConfig {
            language: Some("zh-CN".into()),
            ..Default::default()
        };
        assert_eq!(SonioxAsrProvider::language_hints(&config), vec!["zh"]);

        let config = AsrConfig {
            language: Some("en_US".into()),
            ..Default::default()
        };
        assert_eq!(SonioxAsrProvider::language_hints(&config), vec!["en"]);

        let config = AsrConfig {
            language: Some("ja".into()),
            ..Default::default()
        };
        assert_eq!(SonioxAsrProvider::language_hints(&config), vec!["ja"]);
    }

    #[test]
    fn language_hints_auto_is_empty() {
        for lang in [None, Some("".into()), Some("auto".into()), Some("AUTO".into())] {
            let config = AsrConfig {
                language: lang,
                ..Default::default()
            };
            assert!(SonioxAsrProvider::language_hints(&config).is_empty());
        }
    }

    #[test]
    fn build_config_message_defaults() {
        let config = AsrConfig {
            api_key: "test-key".into(),
            url: String::new(),
            app_key: String::new(),
            language: None,
            sample_rate_hz: 16000,
            hotwords: Vec::new(),
            ..Default::default()
        };
        let body = SonioxAsrProvider::build_config_message(&config, "test-key");
        assert_eq!(body["api_key"], "test-key");
        assert_eq!(body["model"], DEFAULT_MODEL);
        assert_eq!(body["audio_format"], "pcm_s16le");
        assert_eq!(body["sample_rate"], 16000);
        assert_eq!(body["num_channels"], 1);
        assert_eq!(body["enable_endpoint_detection"], true);
        assert_eq!(body["enable_language_identification"], true);
        assert!(body.get("language_hints").is_none());
    }

    #[test]
    fn build_config_message_with_language_and_hotwords() {
        let config = AsrConfig {
            api_key: "k".into(),
            app_key: "stt-rt-v4".into(),
            language: Some("zh-CN".into()),
            hotwords: vec!["Koe".into(), "Soniox".into()],
            sample_rate_hz: 16000,
            ..Default::default()
        };
        let body = SonioxAsrProvider::build_config_message(&config, "k");
        assert_eq!(body["model"], "stt-rt-v4");
        assert_eq!(body["language_hints"], serde_json::json!(["zh"]));
        assert!(body.get("enable_language_identification").is_none());
        assert_eq!(
            body["context"]["terms"],
            serde_json::json!(["Koe", "Soniox"])
        );
    }

    #[test]
    fn parse_response_interim_and_final_tokens() {
        let mut p = SonioxAsrProvider::new();
        let msg = r#"{
            "tokens": [
                {"text": "Hello", "is_final": true, "confidence": 0.9},
                {"text": " world", "is_final": false, "confidence": 0.8}
            ],
            "final_audio_proc_ms": 100,
            "total_audio_proc_ms": 200
        }"#;
        let events = p.parse_response(msg).unwrap();
        assert!(matches!(
            events.first(),
            Some(AsrEvent::Interim(t)) if t == "Hello world"
        ));
        assert!(matches!(
            events.iter().find(|e| matches!(e, AsrEvent::Definite(_))),
            Some(AsrEvent::Definite(t)) if t == "Hello"
        ));
        assert_eq!(p.final_text, "Hello");
    }

    #[test]
    fn parse_response_finished() {
        let mut p = SonioxAsrProvider::new();
        p.final_text = "Hello".into();
        let msg = r#"{
            "tokens": [],
            "final_audio_proc_ms": 100,
            "total_audio_proc_ms": 100,
            "finished": true
        }"#;
        let events = p.parse_response(msg).unwrap();
        assert!(matches!(
            events.iter().find(|e| matches!(e, AsrEvent::Final(_))),
            Some(AsrEvent::Final(t)) if t == "Hello"
        ));
        // finished must not emit Closed — unexpected Closed during the
        // streaming loop is treated as a session error and discards text.
        assert!(
            !events.iter().any(|e| matches!(e, AsrEvent::Closed(_))),
            "finished should not emit Closed"
        );
        assert!(p.emitted_final);
    }

    #[test]
    fn parse_response_skips_end_and_fin_control_tokens() {
        let mut p = SonioxAsrProvider::new();
        // Real Soniox endpoint-detection payload: speech tokens + <end>.
        let msg = r#"{
            "tokens": [
                {"text": "识别之后，好。", "is_final": true},
                {"text": "<end>", "is_final": true},
                {"text": "像要自己。", "is_final": true},
                {"text": "<end>", "is_final": true}
            ]
        }"#;
        let events = p.parse_response(msg).unwrap();
        assert_eq!(p.final_text, "识别之后，好。像要自己。");
        assert!(
            !p.final_text.contains("<end>") && !p.final_text.contains("<fin>"),
            "control tokens must not leak into transcript"
        );
        assert!(matches!(
            events.iter().find(|e| matches!(e, AsrEvent::Definite(_))),
            Some(AsrEvent::Definite(t)) if t == "识别之后，好。像要自己。"
        ));

        // Manual finalize response may include a lone <fin> token.
        let fin_msg = r#"{
            "tokens": [
                {"text": "<fin>", "is_final": true}
            ]
        }"#;
        let _ = p.parse_response(fin_msg).unwrap();
        assert_eq!(p.final_text, "识别之后，好。像要自己。");
    }

    #[test]
    fn parse_response_error() {
        let mut p = SonioxAsrProvider::new();
        let msg = r#"{
            "tokens": [],
            "error_code": 401,
            "error_type": "unauthenticated",
            "error_message": "Incorrect API key provided."
        }"#;
        let events = p.parse_response(msg).unwrap();
        assert!(matches!(
            events.first(),
            Some(AsrEvent::Error(m)) if m.contains("Incorrect API key")
        ));
    }

    #[test]
    fn parse_response_skips_translation_tokens() {
        let mut p = SonioxAsrProvider::new();
        let msg = r#"{
            "tokens": [
                {"text": "Hello", "is_final": true},
                {"text": "Hola", "is_final": true, "translation_status": "translation"}
            ]
        }"#;
        let _ = p.parse_response(msg).unwrap();
        assert_eq!(p.final_text, "Hello");
    }

    #[tokio::test]
    async fn connect_requires_api_key() {
        let config = AsrConfig::default();
        let mut provider = SonioxAsrProvider::new();
        let result = provider.connect(&config).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn send_audio_before_connect_fails() {
        let mut provider = SonioxAsrProvider::new();
        let result = provider.send_audio(&[0u8; 100]).await;
        assert!(result.is_err());
    }

    #[test]
    fn resolve_url_and_model_defaults() {
        let config = AsrConfig {
            url: String::new(),
            app_key: String::new(),
            ..Default::default()
        };
        assert_eq!(SonioxAsrProvider::resolve_url(&config), DEFAULT_URL);
        assert_eq!(SonioxAsrProvider::resolve_model(&config), DEFAULT_MODEL);

        let config = AsrConfig {
            url: "wss://custom.example/ws".into(),
            app_key: "custom-model".into(),
            ..Default::default()
        };
        assert_eq!(
            SonioxAsrProvider::resolve_url(&config),
            "wss://custom.example/ws"
        );
        assert_eq!(SonioxAsrProvider::resolve_model(&config), "custom-model");
    }
}
