use crate::bindings::theater::simple::runtime::log;
use crate::types::api::*;
use genai_types::{CompletionResponse, ModelInfo, ModelPricing, ToolChoice};
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
impl From<OpenAICompletionResponse> for genai_types::CompletionResponse {
    fn from(response: OpenAICompletionResponse) -> Self {
        // Take the first choice (OpenAI can return multiple choices)
        let choice = response.choices.into_iter().next().unwrap_or(OpenAIChoice {
            index: 0,
            message: OpenAIMessage {
                role: "assistant".to_string(),
                content: Some(OpenAIMessageContentFormat::String("No response".to_string())),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                audio: None,
                refusal: None,
            },
            logprobs: None,
            finish_reason: "error".to_string(),
        });

        // Convert content, handling both string and array formats
        let mut content = match choice.message.content {
            Some(content_format) => {
                content_format.to_content_vec()
                    .into_iter()
                    .map(|c| match c {
                        OpenAIMessageContent::Text { text } => genai_types::MessageContent::Text { text },
                        OpenAIMessageContent::ToolUse { id, name, input } => {
                            genai_types::MessageContent::ToolUse { id, name, input }
                        }
                        OpenAIMessageContent::ToolResult { tool_use_id, content, is_error } => {
                            genai_types::MessageContent::ToolResult { tool_use_id, content, is_error }
                        }
                        _ => genai_types::MessageContent::Text { text: "Unsupported content type".to_string() },
                    })
                    .collect()
            }
            None => vec![],
        };

        // Process tool calls from the tool_calls field
        if let Some(tool_calls) = choice.message.tool_calls {
            for tool_call in tool_calls {
                if tool_call.tool_type == "function" {
                    // Parse the arguments JSON string
                    let input = match serde_json::from_str(&tool_call.function.arguments) {
                        Ok(args) => args,
                        Err(e) => {
                            log(&format!("Failed to parse tool call arguments: {}", e));
                            serde_json::json!({})
                        }
                    };

                    content.push(genai_types::MessageContent::ToolUse {
                        id: tool_call.id,
                        name: tool_call.function.name,
                        input,
                    });
                }
            }
        }

        // If we have no content at all, add a default empty response
        if content.is_empty() {
            content.push(genai_types::MessageContent::Text { 
                text: "Empty response".to_string() 
            });
        }

        let stop_reason = match choice.finish_reason.as_str() {
            "stop" => genai_types::messages::StopReason::EndTurn,
            "length" => genai_types::messages::StopReason::MaxTokens,
            "tool_calls" => genai_types::messages::StopReason::ToolUse,
            "content_filter" => genai_types::messages::StopReason::StopSequence,
            other => genai_types::messages::StopReason::Other(other.to_string()),
        };

        Self {
            id: response.id,
            model: response.model,
            role: match choice.message.role.as_str() {
                "user" => genai_types::messages::Role::User,
                "assistant" => genai_types::messages::Role::Assistant,
                "system" => genai_types::messages::Role::System,
                _ => genai_types::messages::Role::Assistant, // default fallback
            },
            content,
            stop_reason,
            stop_sequence: None,
            message_type: "message".to_string(),
            usage: response.usage.into(),
        }
    }
}

impl From<CompletionResponse> for OpenAICompletionResponse {
    fn from(response: CompletionResponse) -> Self {
        Self {
            id: response.id,
            object: "chat.completion".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64,
            model: response.model,
            choices: vec![OpenAIChoice {
                index: 0,
                message: OpenAIMessage {
                    role: match response.role {
                        genai_types::messages::Role::User => "user".to_string(),
                        genai_types::messages::Role::Assistant => "assistant".to_string(),
                        genai_types::messages::Role::System => "system".to_string(),
                    },
                    content: Some(OpenAIMessageContentFormat::Array(
                        response
                            .content
                            .into_iter()
                            .map(OpenAIMessageContent::from)
                            .collect(),
                    )),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    audio: None,
                    refusal: None,
                },
                logprobs: None,
                finish_reason: match response.stop_reason {
                    genai_types::messages::StopReason::EndTurn => "stop".to_string(),
                    genai_types::messages::StopReason::MaxTokens => "length".to_string(),
                    genai_types::messages::StopReason::ToolUse => "tool_calls".to_string(),
                    genai_types::messages::StopReason::StopSequence => "stop".to_string(),
                    genai_types::messages::StopReason::Other(_) => "stop".to_string(),
                },
            }],
            usage: response.usage.into(),
            service_tier: None,
            system_fingerprint: None,
        }
    }
}

/// Request format for the openai-proxy actor
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum OpenAIRequest {
    ListModels,
    GenerateCompletion { request: OpenAICompletionRequest },
}

/// Response status
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ResponseStatus {
    /// Operation succeeded
    #[serde(rename = "Success")]
    Success,

    /// Operation failed
    #[serde(rename = "Error")]
    Error,
}

/// Response format from the openai-proxy actor
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum OpenAIResponse {
    /// List of available models
    ListModels { models: Vec<OpenAIModelInfo> },

    /// Generated completion
    Completion {
        completion: OpenAICompletionResponse,
    },

    /// Error response
    Error { error: String },
}

/// Information about a model
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIModelInfo {
    /// Model ID
    pub id: String,

    /// Display name
    pub display_name: String,

    /// Maximum context window size
    pub max_tokens: u32,

    /// Provider name
    pub provider: String,

    /// Optional pricing information
    pub pricing: Option<OpenAIModelPricing>,
}

impl From<ModelInfo> for OpenAIModelInfo {
    fn from(model_info: ModelInfo) -> Self {
        Self {
            id: model_info.id,
            display_name: model_info.display_name,
            max_tokens: model_info.max_tokens,
            provider: model_info.provider,
            pricing: model_info.pricing.map(|p| p.into()),
        }
    }
}

impl From<OpenAIModelInfo> for ModelInfo {
    fn from(model_info: OpenAIModelInfo) -> Self {
        Self {
            id: model_info.id,
            display_name: model_info.display_name,
            max_tokens: model_info.max_tokens,
            provider: model_info.provider,
            pricing: model_info.pricing.map(|p| p.into()),
        }
    }
}

/// Pricing information for a model
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIModelPricing {
    /// Cost per million input tokens
    pub input_cost_per_million_tokens: f64,

    /// Cost per million output tokens
    pub output_cost_per_million_tokens: f64,
}

impl From<ModelPricing> for OpenAIModelPricing {
    fn from(pricing: ModelPricing) -> Self {
        Self {
            input_cost_per_million_tokens: pricing.input_cost_per_million_tokens,
            output_cost_per_million_tokens: pricing.output_cost_per_million_tokens,
        }
    }
}

impl From<OpenAIModelPricing> for ModelPricing {
    fn from(pricing: OpenAIModelPricing) -> Self {
        Self {
            input_cost_per_million_tokens: pricing.input_cost_per_million_tokens,
            output_cost_per_million_tokens: pricing.output_cost_per_million_tokens,
        }
    }
}

impl From<ToolChoice> for OpenAIToolChoice {
    fn from(choice: ToolChoice) -> Self {
        match choice {
            ToolChoice::Auto => OpenAIToolChoice::String("auto".to_string()),
            ToolChoice::Tool { name } => OpenAIToolChoice::Object {
                choice_type: "function".to_string(),
                function: OpenAIChoiceFunction { name },
            },
            ToolChoice::Any => OpenAIToolChoice::String("required".to_string()),
            ToolChoice::None => OpenAIToolChoice::String("none".to_string()),
        }
    }
}

impl OpenAIModelInfo {
    /// Get maximum tokens for a given model ID
    pub fn get_max_tokens(model_id: &str) -> u32 {
        match model_id {
            // GPT-4o models
            "gpt-4o" | "gpt-4o-2024-11-20" | "gpt-4o-2024-08-06" | "gpt-4o-2024-05-13" => 128000,
            "gpt-4o-mini" | "gpt-4o-mini-2024-07-18" => 128000,
            
            // GPT-4 Turbo models
            "gpt-4-turbo" | "gpt-4-turbo-2024-04-09" | "gpt-4-turbo-preview" => 128000,
            "gpt-4-0125-preview" | "gpt-4-1106-preview" => 128000,
            
            // Standard GPT-4 models
            "gpt-4" | "gpt-4-0613" | "gpt-4-0314" => 8192,
            "gpt-4-32k" | "gpt-4-32k-0613" | "gpt-4-32k-0314" => 32768,
            
            // o-series reasoning models
            "o3" | "o3-2025-01-31" => 200000,
            "o3-mini" | "o3-mini-2025-01-31" => 200000,
            "o1" | "o1-2024-12-17" => 200000,
            "o1-mini" | "o1-mini-2024-09-12" => 128000,
            "o1-preview" | "o1-preview-2024-09-12" => 128000,
            
            // GPT-3.5 models
            "gpt-3.5-turbo" | "gpt-3.5-turbo-0125" | "gpt-3.5-turbo-1106" => 16385,
            "gpt-3.5-turbo-16k" | "gpt-3.5-turbo-16k-0613" => 16385,
            
            // Audio models
            "gpt-4o-audio-preview" | "gpt-4o-audio-preview-2024-10-01" => 128000,
            
            // Default case
            _ => 128000, // Conservative default for newer models
        }
    }

    /// Get pricing information for a given model ID
    pub fn get_pricing(model_id: &str) -> OpenAIModelPricing {
        match model_id {
            // GPT-4o models
            "gpt-4o" | "gpt-4o-2024-11-20" => OpenAIModelPricing {
                input_cost_per_million_tokens: 2.50,
                output_cost_per_million_tokens: 10.00,
            },
            "gpt-4o-2024-08-06" => OpenAIModelPricing {
                input_cost_per_million_tokens: 2.50,
                output_cost_per_million_tokens: 10.00,
            },
            "gpt-4o-2024-05-13" => OpenAIModelPricing {
                input_cost_per_million_tokens: 5.00,
                output_cost_per_million_tokens: 15.00,
            },
            "gpt-4o-mini" | "gpt-4o-mini-2024-07-18" => OpenAIModelPricing {
                input_cost_per_million_tokens: 0.15,
                output_cost_per_million_tokens: 0.60,
            },
            
            // GPT-4 Turbo models
            "gpt-4-turbo" | "gpt-4-turbo-2024-04-09" => OpenAIModelPricing {
                input_cost_per_million_tokens: 10.00,
                output_cost_per_million_tokens: 30.00,
            },
            "gpt-4-0125-preview" | "gpt-4-1106-preview" | "gpt-4-turbo-preview" => OpenAIModelPricing {
                input_cost_per_million_tokens: 10.00,
                output_cost_per_million_tokens: 30.00,
            },
            
            // Standard GPT-4 models
            "gpt-4" | "gpt-4-0613" | "gpt-4-0314" => OpenAIModelPricing {
                input_cost_per_million_tokens: 30.00,
                output_cost_per_million_tokens: 60.00,
            },
            "gpt-4-32k" | "gpt-4-32k-0613" | "gpt-4-32k-0314" => OpenAIModelPricing {
                input_cost_per_million_tokens: 60.00,
                output_cost_per_million_tokens: 120.00,
            },
            
            // o-series reasoning models
            "o3" | "o3-2025-01-31" => OpenAIModelPricing {
                input_cost_per_million_tokens: 60.00,
                output_cost_per_million_tokens: 240.00,
            },
            "o3-mini" | "o3-mini-2025-01-31" => OpenAIModelPricing {
                input_cost_per_million_tokens: 15.00,
                output_cost_per_million_tokens: 60.00,
            },
            "o1" | "o1-2024-12-17" => OpenAIModelPricing {
                input_cost_per_million_tokens: 15.00,
                output_cost_per_million_tokens: 60.00,
            },
            "o1-mini" | "o1-mini-2024-09-12" => OpenAIModelPricing {
                input_cost_per_million_tokens: 3.00,
                output_cost_per_million_tokens: 12.00,
            },
            "o1-preview" | "o1-preview-2024-09-12" => OpenAIModelPricing {
                input_cost_per_million_tokens: 15.00,
                output_cost_per_million_tokens: 60.00,
            },
            
            // GPT-3.5 models
            "gpt-3.5-turbo" | "gpt-3.5-turbo-0125" | "gpt-3.5-turbo-1106" => OpenAIModelPricing {
                input_cost_per_million_tokens: 0.50,
                output_cost_per_million_tokens: 1.50,
            },
            "gpt-3.5-turbo-16k" | "gpt-3.5-turbo-16k-0613" => OpenAIModelPricing {
                input_cost_per_million_tokens: 3.00,
                output_cost_per_million_tokens: 4.00,
            },
            
            // Audio models
            "gpt-4o-audio-preview" | "gpt-4o-audio-preview-2024-10-01" => OpenAIModelPricing {
                input_cost_per_million_tokens: 2.50,
                output_cost_per_million_tokens: 10.00,
            },
            
            // Default for unknown models
            _ => OpenAIModelPricing {
                input_cost_per_million_tokens: 10.00,
                output_cost_per_million_tokens: 30.00,
            },
        }
    }

    /// Get all available OpenAI models
    pub fn get_available_models() -> Vec<OpenAIModelInfo> {
        vec![
            // GPT-4o models
            OpenAIModelInfo {
                id: "gpt-4o".to_string(),
                display_name: "GPT-4o".to_string(),
                max_tokens: Self::get_max_tokens("gpt-4o"),
                provider: "OpenAI".to_string(),
                pricing: Some(Self::get_pricing("gpt-4o")),
            },
            OpenAIModelInfo {
                id: "gpt-4o-mini".to_string(),
                display_name: "GPT-4o Mini".to_string(),
                max_tokens: Self::get_max_tokens("gpt-4o-mini"),
                provider: "OpenAI".to_string(),
                pricing: Some(Self::get_pricing("gpt-4o-mini")),
            },
            
            // GPT-4 Turbo models
            OpenAIModelInfo {
                id: "gpt-4-turbo".to_string(),
                display_name: "GPT-4 Turbo".to_string(),
                max_tokens: Self::get_max_tokens("gpt-4-turbo"),
                provider: "OpenAI".to_string(),
                pricing: Some(Self::get_pricing("gpt-4-turbo")),
            },
            
            // Standard GPT-4 models
            OpenAIModelInfo {
                id: "gpt-4".to_string(),
                display_name: "GPT-4".to_string(),
                max_tokens: Self::get_max_tokens("gpt-4"),
                provider: "OpenAI".to_string(),
                pricing: Some(Self::get_pricing("gpt-4")),
            },
            
            // o-series reasoning models
            OpenAIModelInfo {
                id: "o3".to_string(),
                display_name: "o3".to_string(),
                max_tokens: Self::get_max_tokens("o3"),
                provider: "OpenAI".to_string(),
                pricing: Some(Self::get_pricing("o3")),
            },
            OpenAIModelInfo {
                id: "o3-mini".to_string(),
                display_name: "o3 Mini".to_string(),
                max_tokens: Self::get_max_tokens("o3-mini"),
                provider: "OpenAI".to_string(),
                pricing: Some(Self::get_pricing("o3-mini")),
            },
            OpenAIModelInfo {
                id: "o1".to_string(),
                display_name: "o1".to_string(),
                max_tokens: Self::get_max_tokens("o1"),
                provider: "OpenAI".to_string(),
                pricing: Some(Self::get_pricing("o1")),
            },
            OpenAIModelInfo {
                id: "o1-mini".to_string(),
                display_name: "o1 Mini".to_string(),
                max_tokens: Self::get_max_tokens("o1-mini"),
                provider: "OpenAI".to_string(),
                pricing: Some(Self::get_pricing("o1-mini")),
            },
            
            // GPT-3.5 models
            OpenAIModelInfo {
                id: "gpt-3.5-turbo".to_string(),
                display_name: "GPT-3.5 Turbo".to_string(),
                max_tokens: Self::get_max_tokens("gpt-3.5-turbo"),
                provider: "OpenAI".to_string(),
                pricing: Some(Self::get_pricing("gpt-3.5-turbo")),
            },
        ]
    }
}

use std::error::Error;
use std::fmt;

/// Error type for OpenAI API operations
#[derive(Debug)]
pub enum OpenAIError {
    /// HTTP request failed
    HttpError(String),

    /// Failed to serialize/deserialize JSON
    JsonError(String),

    /// API returned an error
    ApiError { status: u16, message: String },

    /// Unexpected response format
    InvalidResponse(String),

    /// Rate limit exceeded
    RateLimitExceeded { retry_after: Option<u64> },

    /// Authentication error
    AuthenticationError(String),
}

impl fmt::Display for OpenAIError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpenAIError::HttpError(msg) => write!(f, "HTTP error: {}", msg),
            OpenAIError::JsonError(msg) => write!(f, "JSON error: {}", msg),
            OpenAIError::ApiError { status, message } => {
                write!(f, "API error ({}): {}", status, message)
            }
            OpenAIError::InvalidResponse(msg) => write!(f, "Invalid response: {}", msg),
            OpenAIError::RateLimitExceeded { retry_after } => {
                if let Some(seconds) = retry_after {
                    write!(f, "Rate limit exceeded. Retry after {} seconds", seconds)
                } else {
                    write!(f, "Rate limit exceeded")
                }
            }
            OpenAIError::AuthenticationError(msg) => write!(f, "Authentication error: {}", msg),
        }
    }
}

impl Error for OpenAIError {}

impl From<serde_json::Error> for OpenAIError {
    fn from(error: serde_json::Error) -> Self {
        OpenAIError::JsonError(error.to_string())
    }
}
