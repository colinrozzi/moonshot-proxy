// Improved conversion logic for moonshot-proxy
// This addresses the hacky conversion issues by:
// 1. Separating concerns between content format and role determination
// 2. Creating cleaner, more robust deserialization
// 3. Reducing code duplication
// 4. Making the logic more testable

use crate::bindings::colinrozzi::genai_types::types::{
    Message, MessageContent, MessageRole, ToolResult, ToolUse,
};
use crate::bindings::colinrozzi::mcp_protocol::types::ContentItem;
use crate::bindings::theater::simple::runtime::log;
use crate::types::state::ContentFormat;
use serde::{Deserialize, Serialize};

// === CLEANER CONTENT FORMAT HANDLING ===

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIContent {
    items: Vec<OpenAIContentItem>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type")]
pub enum OpenAIContentItem {
    #[serde(rename = "text")]
    Text { text: String },

    #[serde(rename = "image_url")]
    ImageUrl { image_url: OpenAIImageUrl },

    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },

    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: Vec<ContentItem>,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl OpenAIContent {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    pub fn from_text(text: String) -> Self {
        Self {
            items: vec![OpenAIContentItem::Text { text }],
        }
    }

    pub fn add_item(&mut self, item: OpenAIContentItem) {
        self.items.push(item);
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Serialize for different provider formats
    pub fn serialize_for_format(&self, format: &ContentFormat) -> serde_json::Value {
        match format {
            ContentFormat::String => self.serialize_as_string(),
            ContentFormat::Array => self.serialize_as_array(),
        }
    }

    fn serialize_as_string(&self) -> serde_json::Value {
        let text_parts: Vec<String> = self
            .items
            .iter()
            .filter_map(|item| match item {
                OpenAIContentItem::Text { text } => Some(text.clone()),
                OpenAIContentItem::ToolResult {
                    content, is_error, ..
                } => {
                    let result_text = extract_text_from_content_items(content);
                    if *is_error == Some(true) {
                        Some(format!("Error: {}", result_text))
                    } else {
                        Some(result_text)
                    }
                }
                // Skip tool_use and other complex types in string format
                _ => None,
            })
            .collect();

        let combined_text = text_parts.join("\n").trim().to_string();
        serde_json::Value::String(if combined_text.is_empty() {
            "[No content]".to_string()
        } else {
            combined_text
        })
    }

    fn serialize_as_array(&self) -> serde_json::Value {
        serde_json::to_value(&self.items).unwrap_or(serde_json::Value::Null)
    }
}

// === ROBUST TOOL RESULT HANDLING ===

pub struct ToolResultParser;

impl ToolResultParser {
    /// Parse tool result content with proper error handling and fallbacks
    pub fn parse_content(raw_content: &[u8]) -> Vec<ContentItem> {
        // Strategy 1: Direct deserialization
        if let Ok(items) = serde_json::from_slice::<Vec<ContentItem>>(raw_content) {
            log("Successfully parsed tool result as ContentItem array");
            return items;
        }

        // Strategy 2: Parse as UTF-8 string first, then JSON
        if let Ok(content_str) = String::from_utf8(raw_content.to_vec()) {
            if let Ok(items) = serde_json::from_str::<Vec<ContentItem>>(&content_str) {
                log("Successfully parsed tool result from UTF-8 string");
                return items;
            }

            // Strategy 3: Try parsing as single ContentItem
            if let Ok(item) = serde_json::from_str::<ContentItem>(&content_str) {
                log("Successfully parsed tool result as single ContentItem");
                return vec![item];
            }

            // Strategy 4: Treat as plain text
            log("Treating tool result as plain text");
            return vec![Self::create_text_content_item(content_str)];
        }

        // Strategy 5: Final fallback - binary content as text
        let content_str = String::from_utf8_lossy(raw_content);
        log(&format!(
            "Using lossy UTF-8 conversion for tool result: {}",
            content_str
        ));
        vec![Self::create_text_content_item(content_str.to_string())]
    }

    fn create_text_content_item(text: String) -> ContentItem {
        ContentItem::Text(Some(
            crate::bindings::colinrozzi::mcp_protocol::types::TextContent {
                type_: "text".to_string(),
                text,
                annotations: None,
                meta: None,
            },
        ))
    }
}

// === SIMPLIFIED MESSAGE CONVERSION ===

pub struct MessageConverter;

impl MessageConverter {
    /// Convert from genai Message to OpenAI message format
    pub fn to_openai_message(message: Message) -> OpenAIMessage {
        let mut content = OpenAIContent::new();
        let mut tool_calls = Vec::new();
        let mut tool_call_id = None;
        let mut message_role = Self::map_role(&message.role);

        // Process each content item
        for content_item in message.content {
            match content_item {
                MessageContent::Text(text) => {
                    content.add_item(OpenAIContentItem::Text { text });
                }

                MessageContent::ToolUse(ToolUse { id, name, input }) => {
                    let input_value: serde_json::Value = serde_json::from_slice(&input)
                        .unwrap_or_else(|_| {
                            log("Failed to parse tool use input, using empty object");
                            serde_json::json!({})
                        });

                    tool_calls.push(OpenAIToolCall {
                        id: id.clone(),
                        tool_type: "function".to_string(),
                        function: OpenAIFunctionCall {
                            name,
                            arguments: serde_json::to_string(&input_value)
                                .unwrap_or_else(|_| "{}".to_string()),
                        },
                    });
                }

                MessageContent::ToolResult(ToolResult {
                    tool_use_id,
                    content: raw_content,
                    is_error,
                }) => {
                    // Override role for tool results
                    message_role = "tool".to_string();
                    tool_call_id = Some(tool_use_id.clone());

                    let parsed_content = ToolResultParser::parse_content(&raw_content);
                    content.add_item(OpenAIContentItem::ToolResult {
                        tool_use_id,
                        content: parsed_content,
                        is_error: Some(is_error),
                    });
                }
            }
        }

        OpenAIMessage {
            role: message_role,
            content: if content.is_empty() {
                None
            } else {
                Some(content)
            },
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            tool_call_id,
            name: None,
            audio: None,
            refusal: None,
        }
    }

    /// Convert from OpenAI message back to genai Message
    pub fn from_openai_message(openai_msg: OpenAIMessage) -> Message {
        let mut content = Vec::new();

        // Process content items
        if let Some(openai_content) = openai_msg.content {
            for item in openai_content.items {
                match item {
                    OpenAIContentItem::Text { text } => {
                        content.push(MessageContent::Text(text));
                    }

                    OpenAIContentItem::ToolResult {
                        tool_use_id,
                        content: items,
                        is_error,
                    } => {
                        let serialized_content = serde_json::to_vec(&items).unwrap_or_else(|_| {
                            log("Failed to serialize tool result content");
                            Vec::new()
                        });

                        content.push(MessageContent::ToolResult(ToolResult {
                            tool_use_id,
                            content: serialized_content,
                            is_error: is_error.unwrap_or(false),
                        }));
                    }

                    OpenAIContentItem::ToolUse { id, name, input } => {
                        let serialized_input = serde_json::to_vec(&input).unwrap_or_else(|_| {
                            log("Failed to serialize tool use input");
                            serde_json::to_vec(&serde_json::json!({})).unwrap_or_default()
                        });

                        content.push(MessageContent::ToolUse(ToolUse {
                            id,
                            name,
                            input: serialized_input,
                        }));
                    }

                    _ => {
                        log("Unsupported OpenAI content item type");
                        content.push(MessageContent::Text("[Unsupported content]".to_string()));
                    }
                }
            }
        }

        // Process tool calls
        if let Some(tool_calls) = openai_msg.tool_calls {
            for tool_call in tool_calls {
                if tool_call.tool_type == "function" {
                    let input: serde_json::Value =
                        serde_json::from_str(&tool_call.function.arguments).unwrap_or_else(|_| {
                            log("Failed to parse tool call arguments");
                            serde_json::json!({})
                        });

                    let serialized_input = serde_json::to_vec(&input).unwrap_or_default();

                    content.push(MessageContent::ToolUse(ToolUse {
                        id: tool_call.id,
                        name: tool_call.function.name,
                        input: serialized_input,
                    }));
                }
            }
        }

        Message {
            role: Self::map_role_back(&openai_msg.role),
            content,
        }
    }

    fn map_role(role: &MessageRole) -> String {
        match role {
            MessageRole::User => "user".to_string(),
            MessageRole::Assistant => "assistant".to_string(),
            MessageRole::System => "system".to_string(),
        }
    }

    fn map_role_back(role: &str) -> MessageRole {
        match role {
            "user" | "tool" => MessageRole::User, // Tool results come from user context
            "assistant" => MessageRole::Assistant,
            "system" => MessageRole::System,
            _ => MessageRole::User, // Default fallback
        }
    }
}

// === UPDATED MESSAGE STRUCTURE ===

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<OpenAIContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<OpenAIAudio>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
}

impl OpenAIMessage {
    /// Serialize for specific provider format
    pub fn serialize_for_format(&self, format: &ContentFormat) -> serde_json::Value {
        let mut map = serde_json::Map::new();

        map.insert(
            "role".to_string(),
            serde_json::Value::String(self.role.clone()),
        );

        if let Some(content) = &self.content {
            map.insert("content".to_string(), content.serialize_for_format(format));
        }

        if let Some(tool_calls) = &self.tool_calls {
            map.insert(
                "tool_calls".to_string(),
                serde_json::to_value(tool_calls).unwrap_or(serde_json::Value::Null),
            );
        }

        if let Some(tool_call_id) = &self.tool_call_id {
            map.insert(
                "tool_call_id".to_string(),
                serde_json::Value::String(tool_call_id.clone()),
            );
        }

        if let Some(name) = &self.name {
            map.insert("name".to_string(), serde_json::Value::String(name.clone()));
        }

        if let Some(audio) = &self.audio {
            map.insert(
                "audio".to_string(),
                serde_json::to_value(audio).unwrap_or(serde_json::Value::Null),
            );
        }

        if let Some(refusal) = &self.refusal {
            map.insert(
                "refusal".to_string(),
                serde_json::Value::String(refusal.clone()),
            );
        }

        serde_json::Value::Object(map)
    }
}

// Re-export the structs we need from the original api.rs
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: OpenAIFunctionCall,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIAudio {
    pub id: String,
    pub expires_at: i64,
    pub data: String,
    pub transcript: String,
}

// === HELPER FUNCTIONS ===

fn extract_text_from_content_items(items: &[ContentItem]) -> String {
    items
        .iter()
        .filter_map(|item| match item {
            ContentItem::Text(Some(text_content)) => Some(text_content.text.clone()),
            ContentItem::Text(None) => Some("[Empty text]".to_string()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_result_parsing() {
        let json_content = r#"[{"type": "text", "text": "Hello, world!"}]"#;
        let result = ToolResultParser::parse_content(json_content.as_bytes());
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_content_format_serialization() {
        let mut content = OpenAIContent::new();
        content.add_item(OpenAIContentItem::Text {
            text: "Test message".to_string(),
        });

        let string_format = content.serialize_for_format(&ContentFormat::String);
        assert_eq!(
            string_format,
            serde_json::Value::String("Test message".to_string())
        );

        let array_format = content.serialize_for_format(&ContentFormat::Array);
        assert!(array_format.is_array());
    }

    #[test]
    fn test_message_conversion_roundtrip() {
        let original = Message {
            role: MessageRole::User,
            content: vec![MessageContent::Text("Hello".to_string())],
        };

        let openai_msg = MessageConverter::to_openai_message(original.clone());
        let converted_back = MessageConverter::from_openai_message(openai_msg);

        // Check that essential data is preserved
        assert_eq!(converted_back.role, original.role);
        assert_eq!(converted_back.content.len(), original.content.len());
    }
}
