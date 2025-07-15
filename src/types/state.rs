use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ContentFormat {
    /// Use string format for content (legacy providers like Moonshot)
    String,
    /// Use array format for content (modern OpenAI)
    Array,
}

impl Default for ContentFormat {
    fn default() -> Self {
        ContentFormat::Array // Default to modern format
    }
}

/// Retry configuration for API requests
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_retries: u32,
    
    /// Initial delay before first retry in milliseconds
    pub initial_delay_ms: u32,
    
    /// Maximum delay between retries in milliseconds
    pub max_delay_ms: u32,
    
    /// Multiplier for exponential backoff (e.g., 2.0 doubles the delay each time)
    pub backoff_multiplier: f64,
    
    /// Maximum total time to spend on retries in milliseconds
    pub max_total_timeout_ms: u32,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay_ms: 1000,    // Start with 1 second
            max_delay_ms: 30000,       // Cap at 30 seconds
            backoff_multiplier: 2.0,   // Double the delay each time
            max_total_timeout_ms: 60000, // 1 minute total
        }
    }
}

/// Configuration options for the OpenAI API proxy
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Config {
    /// The default OpenAI model to use
    pub default_model: String,
    
    /// Base URL for the API (defaults to OpenAI's API)
    pub base_url: Option<String>,
    
    /// Environment variable name for the API key (defaults to "OPENAI_API_KEY")
    pub api_key_env: Option<String>,
    
    /// Content format to use for requests
    #[serde(default)]
    pub content_format: ContentFormat,
    
    /// Maximum number of items to keep in the optional cache
    pub max_cache_size: Option<usize>,
    
    /// Request timeout in milliseconds
    pub timeout_ms: u32,
    
    /// Retry configuration for failed requests
    pub retry_config: RetryConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            default_model: "gpt-4o".to_string(),
            base_url: None,  // Will default to OpenAI's API
            api_key_env: None,  // Will default to "OPENAI_API_KEY"
            content_format: ContentFormat::default(),
            max_cache_size: Some(100),
            timeout_ms: 30000,  // 30 seconds
            retry_config: RetryConfig::default(),
        }
    }
}

/// Main state for the openai-proxy actor
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct State {
    /// Actor ID
    pub id: String,
    
    /// OpenAI API key
    pub api_key: String,
    
    /// Actor configuration
    pub config: Config,
    
    /// Store ID (if using runtime store)
    pub store_id: Option<String>,
}

impl State {
    pub fn new(
        id: String,
        api_key: String,
        store_id: Option<String>,
        config: Option<Config>,
    ) -> Self {
        Self {
            id,
            api_key,
            config: config.unwrap_or_default(),
            store_id,
        }
    }
}
