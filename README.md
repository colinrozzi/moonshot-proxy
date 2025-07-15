# OpenAI Proxy Actor

A WebAssembly component actor that serves as a proxy for the OpenAI API, making it easy to interact with GPT models within the Theater system through message passing.

## Features

- **API Key Management**: Securely stores and manages OpenAI API keys
- **Message Interface**: Simple request-response messaging system  
- **Model Information**: Includes details about available OpenAI models, context limits, and pricing
- **Error Handling**: Robust error reporting and handling
- **Retry Logic**: Exponential backoff retry for transient failures
- **Multi-modal Support**: Support for text, images, and audio (model dependent)

## Usage

The actor implements a simple request-response message interface that supports:

- **Chat Completion**: Generate responses from OpenAI models (GPT-4o, GPT-4, GPT-3.5, o-series)
- **Model Listing**: List available OpenAI models with their capabilities and pricing
- **Tool Calling**: Support for function calling and tool use
- **Advanced Features**: JSON mode, structured outputs, reasoning effort control (o-series)

## Configuration

The actor accepts these configuration parameters during initialization:

### Base URL Configuration

The proxy supports any OpenAI-compatible API by configuring the `base_url` parameter:

- **OpenAI** (default): `https://api.openai.com/v1`
- **Moonshot AI**: `https://api.moonshot.ai/v1`
- **Other providers**: Any OpenAI-compatible endpoint

### API Key Configuration

The proxy can use different environment variables for API keys by configuring the `api_key_env` parameter:

- **OpenAI** (default): `OPENAI_API_KEY`
- **Moonshot AI**: `MOONSHOT_API_KEY`
- **Other providers**: Any environment variable name

This allows you to keep different API keys for different providers in separate environment variables.

### Content Format Configuration

The proxy supports different content formats for API compatibility:

- **Array** (default): Modern OpenAI format with structured content arrays
- **String**: Legacy format for providers like Moonshot AI that expect simple strings

Configure using the `content_format` parameter: `"Array"` or `"String"`.

```json
{
  "store_id": "optional-store-id",
  "config": {
    "default_model": "gpt-4o",
    "base_url": "https://api.openai.com/v1",
    "api_key_env": "OPENAI_API_KEY",
    "content_format": "Array",
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

## Building

Build the actor using cargo-component:

```bash
cargo component build --release --target wasm32-unknown-unknown
```

Then update the `component_path` in `manifest.toml` to point to the built WASM file.

## Starting

Start the actor using the Theater system:

```rust
let actor_id = start_actor(
    "/path/to/openai-proxy/manifest.toml",
    Some(init_data),
    ("openai-proxy-instance",)
);
```

### Example Configuration Files

The repository includes example configuration files:

- `init.json` - Default configuration (OpenAI)
- `init-openai.json` - Explicit OpenAI configuration
- `init-moonshot.json` - Moonshot AI configuration  
- `init-example-custom.json` - Example custom provider configuration

You can modify the `init_state` path in `manifest.toml` to point to your desired configuration file.

## Message Interface

The OpenAI proxy uses the standard genai-types interface for consistency with other AI provider proxies.

### Request Format

```rust
ProxyRequest::GenerateCompletion {
    request: CompletionRequest {
        model: "gpt-4o".to_string(),
        messages: vec![
            Message {
                role: "user".to_string(),
                content: vec![MessageContent::Text { 
                    text: "Hello, GPT!".to_string() 
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
        id: "chatcmpl-123".to_string(),
        model: "gpt-4o".to_string(),
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

## Supported Models

The proxy supports models from any OpenAI-compatible provider:

### OpenAI Models

When using the default OpenAI endpoint, all current OpenAI models are supported:

### GPT-4o Models
- `gpt-4o` - Latest GPT-4o model
- `gpt-4o-mini` - Efficient version of GPT-4o

### GPT-4 Models
- `gpt-4-turbo` - GPT-4 Turbo with vision
- `gpt-4` - Original GPT-4 model

### Reasoning Models (o-series)
- `o3` - Advanced reasoning model
- `o3-mini` - Efficient reasoning model
- `o1` - Previous generation reasoning
- `o1-mini` - Efficient o1 model

### GPT-3.5 Models
- `gpt-3.5-turbo` - Fast and affordable model

### Moonshot AI Models

When using Moonshot AI's endpoint (`https://api.moonshot.ai/v1`):
- `moonshot-v1-8k` - 8K context window
- `moonshot-v1-32k` - 32K context window
- `moonshot-v1-128k` - 128K context window
- `moonshot-v1-8k-vision-preview` - Vision model with 8K context

### Other Providers

The proxy works with any OpenAI-compatible API. Check your provider's documentation for supported model names.

## Environment Variables

The proxy looks for API keys in environment variables. The default is `OPENAI_API_KEY`, but you can configure it to use any environment variable name:

- `OPENAI_API_KEY` - Default for OpenAI
- `MOONSHOT_API_KEY` - Common for Moonshot AI
- `ANTHROPIC_API_KEY` - Example for Anthropic (if they had OpenAI compatibility)
- `YOUR_PROVIDER_API_KEY` - Any custom environment variable name

The environment variable name is configured using the `api_key_env` field in your configuration.

## Usage Examples

### Using with OpenAI (Default)

```json
{
  "config": {
    "default_model": "gpt-4o"
  }
}
```

### Using with Moonshot AI

```json
{
  "config": {
    "default_model": "moonshot-v1-8k",
    "base_url": "https://api.moonshot.ai/v1",
    "api_key_env": "MOONSHOT_API_KEY",
    "content_format": "String"
  }
}
```

### Using with Custom OpenAI-Compatible Provider

```json
{
  "config": {
    "default_model": "your-model-name",
    "base_url": "https://your-provider.com/v1",
    "api_key_env": "YOUR_PROVIDER_API_KEY",
    "content_format": "String"
  }
}
```

## Advanced Features

### Tool Calling
The proxy supports OpenAI's function calling capabilities:

```rust
let tools = vec![Tool {
    name: "get_weather".to_string(),
    description: Some("Get current weather".to_string()),
    parameters: json!({
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city name"
            }
        },
        "required": ["location"]
    }),
}];
```

### Structured Outputs
For models that support it, you can request structured JSON responses:

```rust
// This would be configured in the OpenAI-specific request parameters
response_format: Some(OpenAIResponseFormat {
    format_type: "json_schema".to_string(),
    json_schema: Some(your_schema),
})
```

### Reasoning Control (o-series models)
Control reasoning effort for o-series models:

```rust
reasoning_effort: Some("high".to_string()) // "low", "medium", "high"
```

## Error Handling

The proxy provides detailed error information:

- **Authentication errors** for invalid API keys
- **Rate limit errors** with retry-after information  
- **Model-specific errors** for unsupported features
- **Network errors** with automatic retry logic

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
