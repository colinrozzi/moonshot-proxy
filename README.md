# Moonshot Proxy Actor

A WebAssembly component actor that serves as a proxy for the Moonshot AI API, making it easy to interact with Moonshot models within the Theater system through message passing.

## Features

- **API Key Management**: Securely stores and manages Moonshot API keys via `MOONSHOT_API_KEY` environment variable
- **Message Interface**: Simple request-response messaging system  
- **Model Information**: Includes details about available Moonshot models and context limits
- **Error Handling**: Robust error reporting and handling
- **Retry Logic**: Exponential backoff retry for transient failures
- **Content Format**: Configured for Moonshot's string-based content format

## Supported Moonshot Models

- `moonshot-v1-8k` - 8K context window (default)
- `moonshot-v1-32k` - 32K context window  
- `moonshot-v1-128k` - 128K context window
- `moonshot-v1-8k-vision-preview` - Vision model with 8K context

## Configuration

The actor is pre-configured for Moonshot AI with optimal settings:

```json
{
  "store_id": null,
  "config": {
    "default_model": "moonshot-v1-8k",
    "base_url": "https://api.moonshot.ai/v1",
    "api_key_env": "MOONSHOT_API_KEY",
    "content_format": "String",
    "max_cache_size": 100,
    "timeout_ms": 30000,
    "retry_config": {
      "max_retries": 4,
      "initial_delay_ms": 1000,
      "max_delay_ms": 30000,
      "backoff_multiplier": 2.0,
      "max_total_timeout_ms": 120000
    }
  }
}
```

## Environment Setup

Set your Moonshot API key as an environment variable:

```bash
export MOONSHOT_API_KEY="your_moonshot_api_key_here"
```

## Building

Build the actor using cargo-component:

```bash
cargo component build --release
```

The generated WebAssembly component will be at:
`./target/wasm32-wasip1/release/moonshot_proxy.wasm`

## Starting

Start the actor using the Theater system with the pre-configured moonshot settings:

```rust
let actor_id = start_actor(
    "/path/to/moonshot-proxy/manifest.toml",
    None, // Uses ./init-moonshot.json by default
    ("moonshot-proxy-instance",)
);
```

## Message Interface

The Moonshot proxy uses the standard genai-types interface for consistency with other AI provider proxies.

### Request Format

```rust
ProxyRequest::GenerateCompletion {
    request: CompletionRequest {
        model: "moonshot-v1-8k".to_string(),
        messages: vec![
            Message {
                role: "user".to_string(),
                content: vec![MessageContent::Text { 
                    text: "Hello, Moonshot!".to_string() 
                }],
            },
        ],
        max_tokens: 1024,
        temperature: Some(0.7),
        system: Some("You are a helpful AI assistant.".to_string()),
        tools: None,
        tool_choice: None,
        disable_parallel_tool_use: None,
    },
}
```

### Response Format

```rust
ProxyResponse::Completion {
    completion: CompletionResponse {
        id: "moonshot-123".to_string(),
        model: "moonshot-v1-8k".to_string(),
        role: "assistant".to_string(),
        content: vec![MessageContent::Text { 
            text: "Hello! How can I help you today?".to_string() 
        }],
        stop_reason: StopReason::EndTurn,
        stop_sequence: None,
        message_type: "message".to_string(),
        usage: Usage {
            input_tokens: 15,
            output_tokens: 10,
        },
    },
}
```

## Key Differences from OpenAI

This proxy is specifically configured for Moonshot AI's API requirements:

1. **Content Format**: Uses `"String"` format instead of OpenAI's array format
2. **Base URL**: Points to `https://api.moonshot.ai/v1`
3. **API Key**: Expects `MOONSHOT_API_KEY` environment variable
4. **Models**: Optimized for Moonshot's model naming conventions

## Error Handling

The proxy provides detailed error information:

- **Authentication errors** for invalid Moonshot API keys
- **Rate limit errors** with retry-after information  
- **Model-specific errors** for unsupported features
- **Network errors** with automatic retry logic

## File Structure

- `init-moonshot.json` - Default initialization configuration for Moonshot
- `manifest.toml` - Actor manifest with correct component path
- `src/` - Source code (copied from openai-proxy but adapted)

## Migration from OpenAI Proxy

This project was created by copying and adapting the OpenAI proxy. Key changes made:

- Package name changed from `openai-proxy` to `moonshot-proxy`
- Default configuration set to Moonshot API endpoint
- Content format set to "String" for Moonshot compatibility
- Environment variable changed to `MOONSHOT_API_KEY`
- Build target updated to `wasm32-wasip1`

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
