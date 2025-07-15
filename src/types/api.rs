use genai_types::{
    CompletionRequest, Message, MessageContent, Usage,
};
use mcp_protocol::tool::ToolContent;
use serde::{Deserialize, Deserializer, Serialize};

// Helper enum to handle both string and array content formats
#[derive(Debug, Clone)]
pub enum OpenAIMessageContentFormat {
    String(String),
    Array(Vec<OpenAIMessageContent>),
}

impl<'de> Deserialize<'de> for OpenAIMessageContentFormat {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde_json::Value;
        let value = Value::deserialize(deserializer)?;
        
        match value {
            Value::String(s) => Ok(OpenAIMessageContentFormat::String(s)),
            Value::Array(_) => {
                let content_array: Vec<OpenAIMessageContent> = 
                    serde_json::from_value(value).map_err(serde::de::Error::custom)?;
                Ok(OpenAIMessageContentFormat::Array(content_array))
            }
            _ => Err(serde::de::Error::custom("content must be string or array")),
        }
    }
}

impl Serialize for OpenAIMessageContentFormat {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            OpenAIMessageContentFormat::String(s) => s.serialize(serializer),
            OpenAIMessageContentFormat::Array(arr) => arr.serialize(serializer),
        }
    }
}

impl OpenAIMessageContentFormat {
    pub fn to_content_vec(self) -> Vec<OpenAIMessageContent> {
        match self {
            OpenAIMessageContentFormat::String(s) => {
                vec![OpenAIMessageContent::Text { text: s }]
            }
            OpenAIMessageContentFormat::Array(arr) => arr,
        }
    }
    
    // Provider-aware serialization
    pub fn serialize_for_provider(&self, format: &crate::types::state::ContentFormat) -> serde_json::Value {
        match (self, format) {
            (OpenAIMessageContentFormat::String(s), _) => {
                serde_json::Value::String(s.clone())
            }
            (OpenAIMessageContentFormat::Array(arr), crate::types::state::ContentFormat::String) => {
                // Convert array to string for legacy providers that don't support structured content
                let mut text_parts = Vec::new();
                
                for content in arr {
                    match content {
                        OpenAIMessageContent::Text { text } => {
                            text_parts.push(text.clone());
                        }
                        OpenAIMessageContent::ToolResult { tool_use_id: _, content, is_error } => {
                            // Convert tool results to plain text for legacy providers
                            let result_text = content.iter()
                                .filter_map(|c| match c {
                                    mcp_protocol::tool::ToolContent::Text { text } => Some(text.as_str()),
                                    _ => None,
                                })
                                .collect::<Vec<_>>()
                                .join("\n");
                            
                            if *is_error == Some(true) {
                                text_parts.push(format!("Error: {}", result_text));
                            } else {
                                text_parts.push(result_text);
                            }
                        }
                        _ => {
                            // For other content types (like ToolUse), skip in String format
                            // as they should be handled separately
                        }
                    }
                }
                
                let final_text = text_parts.join("\n").trim().to_string();
                if final_text.is_empty() {
                    serde_json::Value::String("[No content]".to_string())
                } else {
                    serde_json::Value::String(final_text)
                }
            }
            (OpenAIMessageContentFormat::Array(arr), crate::types::state::ContentFormat::Array) => {
                serde_json::to_value(arr).unwrap_or(serde_json::Value::Null)
            }
        }
    }
}

/// Different types of content that can be in a message for OpenAI
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type")]
pub enum OpenAIMessageContent {
    #[serde(rename = "text")]
    Text { text: String },

    #[serde(rename = "image_url")]
    ImageUrl { 
        image_url: OpenAIImageUrl,
    },

    #[serde(rename = "audio")]
    Audio {
        #[serde(skip_serializing_if = "Option::is_none")]
        input_audio: Option<OpenAIInputAudio>,
    },

    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },

    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: Vec<ToolContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>, // "low" | "high" | "auto"
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIInputAudio {
    pub data: String, // base64 encoded audio
    pub format: String, // "wav", "mp3", etc.
}

impl From<MessageContent> for OpenAIMessageContent {
    fn from(content: MessageContent) -> Self {
        match content {
            MessageContent::Text { text } => OpenAIMessageContent::Text { text },
            MessageContent::ToolUse { id, name, input } => {
                OpenAIMessageContent::ToolUse { id, name, input }
            }
            MessageContent::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => OpenAIMessageContent::ToolResult {
                tool_use_id,
                content,
                is_error,
            },
        }
    }
}

/// A single message in a conversation with OpenAI
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIMessage {
    /// Role of the message sender (system, user, assistant, developer, tool)
    pub role: String,

    /// Content of the message - can be string or array depending on API version
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<OpenAIMessageContentFormat>,

    /// Name of the tool that was called (for tool messages)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Tool calls made by the assistant
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIToolCall>>,

    /// Tool call ID (for tool messages)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,

    /// Audio content for audio-capable models
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<OpenAIAudio>,

    /// Refusal reason if the model refused to answer
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: String, // "function"
    pub function: OpenAIFunctionCall,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIFunctionCall {
    pub name: String,
    pub arguments: String, // JSON string
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIAudio {
    pub id: String,
    pub expires_at: i64,
    pub data: String, // base64 encoded
    pub transcript: String,
}

impl From<Message> for OpenAIMessage {
    fn from(message: Message) -> Self {
        // Check if this message contains tool results - if so, it should be role "tool"
        let has_tool_results = message.content.iter().any(|content| {
            matches!(content, genai_types::MessageContent::ToolResult { .. })
        });

        // Check if this message contains tool calls - if so, extract them
        let mut tool_calls = Vec::new();
        let mut regular_content = Vec::new();
        let mut tool_call_id = None;

        for content in message.content {
            match content {
                genai_types::MessageContent::ToolUse { id, name, input } => {
                    tool_calls.push(OpenAIToolCall {
                        id: id.clone(),
                        tool_type: "function".to_string(),
                        function: OpenAIFunctionCall {
                            name,
                            arguments: serde_json::to_string(&input).unwrap_or("{}".to_string()),
                        },
                    });
                }
                genai_types::MessageContent::ToolResult { tool_use_id, content, is_error: _ } => {
                    // For tool results, we need to set the tool_call_id and role to "tool"
                    tool_call_id = Some(tool_use_id);
                    // Convert tool result content to string format
                    for tool_content in content {
                        match tool_content {
                            mcp_protocol::tool::ToolContent::Text { text } => {
                                regular_content.push(OpenAIMessageContent::Text { text });
                            }
                            // Handle other tool content types as needed
                            _ => {
                                regular_content.push(OpenAIMessageContent::Text { 
                                    text: format!("{:?}", tool_content) 
                                });
                            }
                        }
                    }
                }
                other => {
                    regular_content.push(OpenAIMessageContent::from(other));
                }
            }
        }

        // Determine the role
        let role = if has_tool_results {
            "tool".to_string()  // Tool results must use "tool" role
        } else {
            match message.role {
                genai_types::messages::Role::User => "user".to_string(),
                genai_types::messages::Role::Assistant => "assistant".to_string(),
                genai_types::messages::Role::System => "system".to_string(),
            }
        };

        Self {
            role,
            content: if regular_content.is_empty() {
                None
            } else {
                Some(OpenAIMessageContentFormat::Array(regular_content))
            },
            name: None, // Could be extracted from message metadata if needed
            tool_calls: if tool_calls.is_empty() { None } else { Some(tool_calls) },
            tool_call_id,
            audio: None,
            refusal: None,
        }
    }
}

impl OpenAIMessage {
    // Custom serialization method that respects provider format
    pub fn serialize_for_provider(&self, format: &crate::types::state::ContentFormat) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        
        map.insert("role".to_string(), serde_json::Value::String(self.role.clone()));
        
        if let Some(content) = &self.content {
            map.insert("content".to_string(), content.serialize_for_provider(format));
        }
        
        if let Some(name) = &self.name {
            map.insert("name".to_string(), serde_json::Value::String(name.clone()));
        }
        
        if let Some(tool_calls) = &self.tool_calls {
            map.insert("tool_calls".to_string(), serde_json::to_value(tool_calls).unwrap_or(serde_json::Value::Null));
        }
        
        if let Some(tool_call_id) = &self.tool_call_id {
            map.insert("tool_call_id".to_string(), serde_json::Value::String(tool_call_id.clone()));
        }
        
        if let Some(audio) = &self.audio {
            map.insert("audio".to_string(), serde_json::to_value(audio).unwrap_or(serde_json::Value::Null));
        }
        
        if let Some(refusal) = &self.refusal {
            map.insert("refusal".to_string(), serde_json::Value::String(refusal.clone()));
        }
        
        serde_json::Value::Object(map)
    }
}

/// Request to generate a completion from OpenAI
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAICompletionRequest {
    /// The OpenAI model to use (e.g., "gpt-4o", "o3")
    pub model: String,

    /// List of messages in the conversation
    pub messages: Vec<OpenAIMessage>,

    /// Maximum number of tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,

    /// Temperature parameter (0.0 to 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Top-p nucleus sampling parameter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Number of completions to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,

    /// Whether to stream the response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// Stop sequences
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,

    /// Presence penalty (-2.0 to 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,

    /// Frequency penalty (-2.0 to 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,

    /// Tools to make available to the model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenAITool>>,

    /// Tool choice configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<OpenAIToolChoice>,

    /// Whether to enable parallel tool calls
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,

    /// User identifier for abuse detection
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    /// Response format specification
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<OpenAIResponseFormat>,

    /// Random seed for deterministic responses
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i32>,

    /// Whether to return log probabilities
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,

    /// Number of top log probabilities to return
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,

    /// Output modalities (e.g., ["text", "audio"])
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<String>>,

    /// Audio parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<OpenAIAudioParams>,

    /// Reasoning effort for o-series models
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>, // "low" | "medium" | "high"

    /// Service tier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>, // "auto" | "default" | "flex" | "priority"
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAITool {
    #[serde(rename = "type")]
    pub tool_type: String, // "function"
    pub function: OpenAIFunction,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIFunction {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: serde_json::Value, // JSON schema
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum OpenAIToolChoice {
    String(String), // "none" | "auto" | "required"
    Object {
        #[serde(rename = "type")]
        choice_type: String,
        function: OpenAIChoiceFunction,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIChoiceFunction {
    pub name: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String, // "text" | "json_object" | "json_schema"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<serde_json::Value>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIAudioParams {
    pub voice: String, // "alloy" | "echo" | "fable" | "onyx" | "nova" | "shimmer"
    pub format: String, // "wav" | "mp3" | "flac" | "opus"
}

impl From<CompletionRequest> for OpenAICompletionRequest {
    fn from(request: CompletionRequest) -> Self {
        Self {
            model: request.model,
            messages: request
                .messages
                .into_iter()
                .map(OpenAIMessage::from)
                .collect(),
            max_completion_tokens: Some(request.max_tokens),
            temperature: request.temperature,
            top_p: None,
            n: None,
            stream: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            tools: request.tools.map(|tools| {
                tools.into_iter().map(|tool| OpenAITool {
                    tool_type: "function".to_string(),
                    function: OpenAIFunction {
                        name: tool.name,
                        description: tool.description,
                        parameters: tool.input_schema,
                        strict: None,
                    },
                }).collect()
            }),
            tool_choice: request.tool_choice.map(OpenAIToolChoice::from),
            parallel_tool_calls: request.disable_parallel_tool_use.map(|disable| !disable),
            user: None,
            response_format: None,
            seed: None,
            logprobs: None,
            top_logprobs: None,
            modalities: None,
            audio: None,
            reasoning_effort: None,
            service_tier: None,
        }
    }
}

impl OpenAICompletionRequest {
    // Custom serialization method that respects provider format
    pub fn serialize_for_provider(&self, format: &crate::types::state::ContentFormat) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        
        map.insert("model".to_string(), serde_json::Value::String(self.model.clone()));
        
        let messages: Vec<serde_json::Value> = self.messages
            .iter()
            .map(|msg| msg.serialize_for_provider(format))
            .collect();
        map.insert("messages".to_string(), serde_json::Value::Array(messages));
        
        if let Some(max_tokens) = self.max_completion_tokens {
            map.insert("max_completion_tokens".to_string(), serde_json::Value::Number(max_tokens.into()));
        }
        
        if let Some(temperature) = self.temperature {
            if let Some(num) = serde_json::Number::from_f64(temperature as f64) {
                map.insert("temperature".to_string(), serde_json::Value::Number(num));
            }
        }
        
        if let Some(top_p) = self.top_p {
            if let Some(num) = serde_json::Number::from_f64(top_p as f64) {
                map.insert("top_p".to_string(), serde_json::Value::Number(num));
            }
        }
        
        if let Some(n) = self.n {
            map.insert("n".to_string(), serde_json::Value::Number(n.into()));
        }
        
        if let Some(stream) = self.stream {
            map.insert("stream".to_string(), serde_json::Value::Bool(stream));
        }
        
        if let Some(stop) = &self.stop {
            map.insert("stop".to_string(), serde_json::to_value(stop).unwrap_or(serde_json::Value::Null));
        }
        
        if let Some(presence_penalty) = self.presence_penalty {
            if let Some(num) = serde_json::Number::from_f64(presence_penalty as f64) {
                map.insert("presence_penalty".to_string(), serde_json::Value::Number(num));
            }
        }
        
        if let Some(frequency_penalty) = self.frequency_penalty {
            if let Some(num) = serde_json::Number::from_f64(frequency_penalty as f64) {
                map.insert("frequency_penalty".to_string(), serde_json::Value::Number(num));
            }
        }
        
        if let Some(tools) = &self.tools {
            map.insert("tools".to_string(), serde_json::to_value(tools).unwrap_or(serde_json::Value::Null));
        }
        
        if let Some(tool_choice) = &self.tool_choice {
            map.insert("tool_choice".to_string(), serde_json::to_value(tool_choice).unwrap_or(serde_json::Value::Null));
        }
        
        serde_json::Value::Object(map)
    }
}

/// Information about token usage from OpenAI
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<OpenAIPromptTokensDetails>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<OpenAICompletionTokensDetails>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIPromptTokensDetails {
    pub cached_tokens: u32,
    pub audio_tokens: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAICompletionTokensDetails {
    pub reasoning_tokens: u32,
    pub audio_tokens: u32,
    pub accepted_prediction_tokens: u32,
    pub rejected_prediction_tokens: u32,
}

impl From<Usage> for OpenAIUsage {
    fn from(usage: Usage) -> Self {
        Self {
            prompt_tokens: usage.input_tokens,
            completion_tokens: usage.output_tokens,
            total_tokens: usage.input_tokens + usage.output_tokens,
            prompt_tokens_details: None,
            completion_tokens_details: None,
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
