#[cfg(test)]
mod tests {
    use crate::types::*;
    use genai_types::{Message, MessageContent};
    use crate::types::api::OpenAIMessage;
    use mcp_protocol::tool::ToolContent;

    #[test]
    fn test_openai_model_info() {
        let models = OpenAIModelInfo::get_available_models();
        assert!(!models.is_empty());
        
        // Check that GPT-4o is in the list
        assert!(models.iter().any(|m| m.id == "gpt-4o"));
        
        // Check that o3 is in the list
        assert!(models.iter().any(|m| m.id == "o3"));
    }

    #[test]
    fn test_token_limits() {
        assert_eq!(OpenAIModelInfo::get_max_tokens("gpt-4o"), 128000);
        assert_eq!(OpenAIModelInfo::get_max_tokens("o3"), 200000);
        assert_eq!(OpenAIModelInfo::get_max_tokens("gpt-3.5-turbo"), 16385);
        assert_eq!(OpenAIModelInfo::get_max_tokens("unknown-model"), 128000);
    }

    #[test]
    fn test_pricing() {
        let pricing = OpenAIModelInfo::get_pricing("gpt-4o");
        assert_eq!(pricing.input_cost_per_million_tokens, 2.50);
        assert_eq!(pricing.output_cost_per_million_tokens, 10.00);
        
        let mini_pricing = OpenAIModelInfo::get_pricing("gpt-4o-mini");
        assert_eq!(mini_pricing.input_cost_per_million_tokens, 0.15);
        assert_eq!(mini_pricing.output_cost_per_million_tokens, 0.60);
    }

    #[test]
    fn test_config_defaults() {
        let config = Config::default();
        assert_eq!(config.default_model, "gpt-4o");
        assert_eq!(config.timeout_ms, 30000);
        assert_eq!(config.max_cache_size, Some(100));
    }

    #[test]
    fn test_tool_result_message_uses_tool_role() {
        // Create a message with tool result content
        let message = Message {
            role: genai_types::messages::Role::User, // Original role doesn't matter
            content: vec![
                MessageContent::ToolResult {
                    tool_use_id: "call_123".to_string(),
                    content: vec![
                        ToolContent::Text {
                            text: "Search results: Moonshot AI is a company...".to_string()
                        }
                    ],
                    is_error: Some(false),
                }
            ],
        };

        // Convert to OpenAI format
        let openai_message: OpenAIMessage = message.into();

        // Verify it uses "tool" role
        assert_eq!(openai_message.role, "tool");
        assert_eq!(openai_message.tool_call_id, Some("call_123".to_string()));
    }

    #[test]
    fn test_tool_use_message_creates_tool_calls() {
        // Create a message with tool use content
        let message = Message {
            role: genai_types::messages::Role::Assistant,
            content: vec![
                MessageContent::Text {
                    text: "I'll search for that information.".to_string()
                },
                MessageContent::ToolUse {
                    id: "call_456".to_string(),
                    name: "search".to_string(),
                    input: serde_json::json!({
                        "query": "Context Caching Moonshot AI"
                    }),
                }
            ],
        };

        // Convert to OpenAI format
        let openai_message: OpenAIMessage = message.into();

        // Verify it uses "assistant" role (tool calls are made by assistant)
        assert_eq!(openai_message.role, "assistant");
        
        // Verify tool_calls is populated
        assert!(openai_message.tool_calls.is_some());
        let tool_calls = openai_message.tool_calls.unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call_456");
        assert_eq!(tool_calls[0].function.name, "search");
    }

    #[test]
    fn test_regular_message_unchanged() {
        // Create a regular message without tools
        let message = Message {
            role: genai_types::messages::Role::User,
            content: vec![
                MessageContent::Text {
                    text: "Hello, can you help me search for something?".to_string()
                }
            ],
        };

        // Convert to OpenAI format
        let openai_message: OpenAIMessage = message.into();

        // Verify it uses original role
        assert_eq!(openai_message.role, "user");
        assert!(openai_message.tool_calls.is_none());
        assert!(openai_message.tool_call_id.is_none());
    }
}
