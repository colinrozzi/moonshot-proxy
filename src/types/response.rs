use crate::types::api::*;
use crate::types::conversion::*;
use crate::bindings::colinrozzi::genai_types::types::{
    CompletionResponse, ModelInfo, ModelPricing, StopReason,
};
use serde::{Deserialize, Serialize};

/// A single choice in the completion response
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIChoice {
    pub index: u32,
    pub message: OpenAIMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: String, // "stop" | "length" | "tool_calls" | "content_filter" | "function_call"
}

/// Response from a completion request
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAICompletionResponse {
    pub id: String,
    pub object: String, // "chat.completion"
    pub created: i64,
    pub model: String,
    pub choices: Vec<OpenAIChoice>,
    pub usage: OpenAIUsage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

// Implement conversion from OpenAI types to genai-types
impl From<OpenAICompletionResponse> for CompletionResponse {
    fn from(response: OpenAICompletionResponse) -> Self {
        // Take the first choice (OpenAI can return multiple choices)
        let choice = response.choices.into_iter().next().unwrap_or(OpenAIChoice {
            index: 0,
            message: OpenAIMessage {
                role: "assistant".to_string(),
                content: Some(OpenAIContent::from_text("No response".to_string())),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                audio: None,
                refusal: None,
            },
            logprobs: None,
            finish_reason: "error".to_string(),
        });

        // Convert the OpenAI message back to genai Message format
        let genai_message = MessageConverter::from_openai_message(choice.message);

        // Map finish reason to stop reason
        let stop_reason = match choice.finish_reason.as_str() {
            "stop" => StopReason::EndTurn,
            "length" => StopReason::MaxTokens,
            "tool_calls" => StopReason::ToolUse,
            "content_filter" => StopReason::EndTurn, // Map to closest equivalent
            _ => StopReason::EndTurn,
        };

        CompletionResponse {
            id: response.id,
            model: response.model,
            role: genai_message.role,
            content: genai_message.content,
            stop_reason,
            usage: response.usage.into(),
        }
    }
}

/// Error types for OpenAI API interactions
#[derive(Debug, Clone)]
pub enum OpenAIError {
    /// Network or HTTP error
    HttpError(String),
    /// JSON serialization/deserialization error
    SerializationError(String),
    /// OpenAI API returned an error
    ApiError { status: u16, message: String },
    /// Authentication failed
    AuthenticationError(String),
    /// Rate limit exceeded
    RateLimitExceeded { retry_after: Option<u64> },
    /// Invalid response format
    InvalidResponse(String),
    /// Unsupported model requested
    UnsupportedModel { requested: String, suggestions: Vec<String> },
}

impl std::fmt::Display for OpenAIError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpenAIError::HttpError(msg) => write!(f, "HTTP error: {}", msg),
            OpenAIError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            OpenAIError::ApiError { status, message } => {
                write!(f, "API error {}: {}", status, message)
            }
            OpenAIError::AuthenticationError(msg) => write!(f, "Authentication error: {}", msg),
            OpenAIError::RateLimitExceeded { retry_after } => {
                if let Some(seconds) = retry_after {
                    write!(f, "Rate limit exceeded, retry after {} seconds", seconds)
                } else {
                    write!(f, "Rate limit exceeded")
                }
            }
            OpenAIError::InvalidResponse(msg) => write!(f, "Invalid response: {}", msg),
            OpenAIError::UnsupportedModel { requested, suggestions } => {
                write!(f, "Unsupported model '{}'. ", requested)?;
                if !suggestions.is_empty() {
                    write!(f, "Did you mean one of: {}?", suggestions.join(", "))?;
                } else {
                    write!(f, "Please check the available models list.")?;
                }
                Ok(())
            }
        }
    }
}

impl From<serde_json::Error> for OpenAIError {
    fn from(err: serde_json::Error) -> Self {
        OpenAIError::SerializationError(err.to_string())
    }
}

/// Model information for OpenAI models
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIModelInfo {
    pub id: String,
    pub object: String,
    pub created: Option<i64>,
    pub owned_by: String,
    pub context_length: u32,
    pub pricing: Option<ModelPricing>,
}

impl OpenAIModelInfo {
    /// Check if a model is supported
    pub fn is_model_supported(model_id: &str) -> bool {
        Self::get_available_models()
            .iter()
            .any(|model| model.id == model_id)
    }

    /// Get supported model suggestions for error messages
    pub fn get_model_suggestions(requested_model: &str) -> Vec<String> {
        let available_models = Self::get_available_models();
        let mut suggestions = Vec::new();
        
        // First, look for exact case-insensitive matches
        for model in &available_models {
            if model.id.to_lowercase() == requested_model.to_lowercase() {
                suggestions.push(model.id.clone());
            }
        }
        
        // If no exact matches, look for partial matches
        if suggestions.is_empty() {
            for model in &available_models {
                if model.id.to_lowercase().contains(&requested_model.to_lowercase()) 
                   || requested_model.to_lowercase().contains(&model.id.to_lowercase()) {
                    suggestions.push(model.id.clone());
                }
            }
        }
        
        // If still no matches, return popular models based on provider hints
        if suggestions.is_empty() {
            if requested_model.to_lowercase().contains("moonshot") 
               || requested_model.to_lowercase().contains("kimi") {
                suggestions.extend([
                    "moonshot-v1-8k".to_string(),
                    "moonshot-v1-32k".to_string(),
                    "kimi-k2-0711-preview".to_string(),
                ]);
            } else if requested_model.to_lowercase().contains("gpt") {
                suggestions.extend([
                    "gpt-4".to_string(),
                    "gpt-4-turbo".to_string(),
                    "gpt-3.5-turbo".to_string(),
                ]);
            } else {
                // Default popular suggestions
                suggestions.extend([
                    "moonshot-v1-8k".to_string(),
                    "kimi-k2-0711-preview".to_string(),
                    "gpt-4".to_string(),
                ]);
            }
        }
        
        suggestions.into_iter().take(3).collect()
    }

    /// Get a list of available OpenAI and Moonshot models
    pub fn get_available_models() -> Vec<Self> {
        vec![
            // Moonshot models
            Self {
                id: "moonshot-v1-8k".to_string(),
                object: "model".to_string(),
                created: None,
                owned_by: "moonshot".to_string(),
                context_length: 8192,
                pricing: None,
            },
            Self {
                id: "moonshot-v1-32k".to_string(),
                object: "model".to_string(),
                created: None,
                owned_by: "moonshot".to_string(),
                context_length: 32768,
                pricing: None,
            },
            Self {
                id: "moonshot-v1-128k".to_string(),
                object: "model".to_string(),
                created: None,
                owned_by: "moonshot".to_string(),
                context_length: 131072,
                pricing: None,
            },
            Self {
                id: "moonshot-v1-8k-vision-preview".to_string(),
                object: "model".to_string(),
                created: None,
                owned_by: "moonshot".to_string(),
                context_length: 8192,
                pricing: None,
            },
            Self {
                id: "kimi-k2-0711-preview".to_string(),
                object: "model".to_string(),
                created: None,
                owned_by: "moonshot".to_string(),
                context_length: 128000,
                pricing: None,
            },
            Self {
                id: "kimi-k2-0707-preview".to_string(),
                object: "model".to_string(),
                created: None,
                owned_by: "moonshot".to_string(),
                context_length: 128000,
                pricing: None,
            },
            // OpenAI models (for compatibility)
            Self {
                id: "gpt-4".to_string(),
                object: "model".to_string(),
                created: None,
                owned_by: "openai".to_string(),
                context_length: 8192,
                pricing: None,
            },
            Self {
                id: "gpt-4-turbo".to_string(),
                object: "model".to_string(),
                created: None,
                owned_by: "openai".to_string(),
                context_length: 128000,
                pricing: None,
            },
            Self {
                id: "gpt-3.5-turbo".to_string(),
                object: "model".to_string(),
                created: None,
                owned_by: "openai".to_string(),
                context_length: 4096,
                pricing: None,
            },
        ]
    }
}

impl From<OpenAIModelInfo> for ModelInfo {
    fn from(model: OpenAIModelInfo) -> Self {
        Self {
            id: model.id.clone(),
            display_name: format!("{} ({})", model.id, model.owned_by),
            max_tokens: model.context_length,
            provider: model.owned_by,
            pricing: model.pricing,
        }
    }
}
