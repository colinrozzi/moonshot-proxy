// Simplified API module that uses the new conversion logic
use crate::bindings::colinrozzi::genai_types::types::CompletionRequest;
use crate::types::conversion::{MessageConverter, OpenAIMessage};
use crate::types::state::ContentFormat;
use serde::{Deserialize, Serialize};

/// OpenAI completion request structure
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAICompletionRequest {
    pub model: String,
    pub messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

impl From<CompletionRequest> for OpenAICompletionRequest {
    fn from(request: CompletionRequest) -> Self {
        let mut messages: Vec<OpenAIMessage> = request
            .messages
            .into_iter()
            .map(MessageConverter::to_openai_message)
            .collect();

        // Add system message if provided
        if let Some(system_content) = request.system {
            let system_message = OpenAIMessage {
                role: "system".to_string(),
                content: Some(crate::types::conversion::OpenAIContent::from_text(system_content)),
                tool_calls: None,
                tool_call_id: None,
                name: None,
                audio: None,
                refusal: None,
            };
            messages.insert(0, system_message);
        }

        Self {
            model: request.model,
            messages,
            max_tokens: Some(request.max_tokens),
            temperature: request.temperature.map(|t| t as f64),
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            stream: Some(false), // We don't support streaming yet
        }
    }
}

impl OpenAICompletionRequest {
    /// Serialize for specific provider format
    pub fn serialize_for_provider(&self, format: &ContentFormat) -> serde_json::Value {
        let mut request = serde_json::Map::new();
        
        request.insert("model".to_string(), serde_json::Value::String(self.model.clone()));
        
        // Serialize messages using the format-aware method
        let messages: Vec<serde_json::Value> = self.messages
            .iter()
            .map(|msg| msg.serialize_for_format(format))
            .collect();
        
        request.insert("messages".to_string(), serde_json::Value::Array(messages));
        
        if let Some(max_tokens) = self.max_tokens {
            request.insert("max_tokens".to_string(), serde_json::Value::Number(max_tokens.into()));
        }
        
        if let Some(temperature) = self.temperature {
            if let Some(temp_num) = serde_json::Number::from_f64(temperature) {
                request.insert("temperature".to_string(), serde_json::Value::Number(temp_num));
            }
        }
        
        if let Some(top_p) = self.top_p {
            if let Some(top_p_num) = serde_json::Number::from_f64(top_p) {
                request.insert("top_p".to_string(), serde_json::Value::Number(top_p_num));
            }
        }
        
        if let Some(frequency_penalty) = self.frequency_penalty {
            if let Some(freq_num) = serde_json::Number::from_f64(frequency_penalty) {
                request.insert("frequency_penalty".to_string(), serde_json::Value::Number(freq_num));
            }
        }
        
        if let Some(presence_penalty) = self.presence_penalty {
            if let Some(pres_num) = serde_json::Number::from_f64(presence_penalty) {
                request.insert("presence_penalty".to_string(), serde_json::Value::Number(pres_num));
            }
        }
        
        if let Some(stop) = &self.stop {
            request.insert("stop".to_string(), serde_json::to_value(stop).unwrap_or(serde_json::Value::Null));
        }
        
        if let Some(stream) = self.stream {
            request.insert("stream".to_string(), serde_json::Value::Bool(stream));
        }
        
        serde_json::Value::Object(request)
    }
}

/// Usage statistics from the API
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Input audio structure for audio-capable models
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIInputAudio {
    pub data: String,   // base64 encoded audio
    pub format: String, // "wav", "mp3", etc.
}

// Implement conversion traits for OpenAIUsage
use crate::bindings::colinrozzi::genai_types::types::Usage;

impl From<Usage> for OpenAIUsage {
    fn from(usage: Usage) -> Self {
        Self {
            prompt_tokens: usage.input_tokens,
            completion_tokens: usage.output_tokens,
            total_tokens: usage.input_tokens + usage.output_tokens,
        }
    }
}

impl From<OpenAIUsage> for Usage {
    fn from(usage: OpenAIUsage) -> Self {
        Self {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
        }
    }
}
