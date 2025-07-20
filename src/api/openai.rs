use crate::bindings::theater::simple::http_client::{send_http, HttpRequest};
use crate::bindings::theater::simple::runtime::log;
use crate::bindings::theater::simple::timing;
use crate::types::{
    api::OpenAICompletionRequest,
    response::{OpenAICompletionResponse, OpenAIError, OpenAIModelInfo},
    state::{ContentFormat, RetryConfig},
};

/// Client for interacting with the OpenAI-compatible API (including Moonshot)
pub struct OpenAIClient {
    /// API key
    api_key: String,
    /// Base URL for the API
    base_url: String,
}

impl OpenAIClient {
    /// Create a new OpenAI client
    pub fn new(api_key: String) -> Self {
        Self::new_with_base_url(api_key, "https://api.openai.com/v1".to_string())
    }
    
    /// Create a new OpenAI client with custom base URL
    pub fn new_with_base_url(api_key: String, base_url: String) -> Self {
        Self {
            api_key,
            base_url,
        }
    }

    /// Check if a status code indicates a retryable error
    fn is_retryable_error(status: u16) -> bool {
        match status {
            429 => true, // Rate limit exceeded
            502 => true, // Bad gateway
            503 => true, // Service unavailable
            504 => true, // Gateway timeout
            _ => false,
        }
    }

    /// Execute an HTTP request with exponential backoff retry logic
    fn execute_with_retry(
        &self,
        request: &HttpRequest,
        retry_config: &RetryConfig,
    ) -> Result<crate::bindings::theater::simple::http_client::HttpResponse, OpenAIError> {
        let start_time = timing::now();
        let mut current_delay = retry_config.initial_delay_ms;
        let mut attempt = 0;

        loop {
            attempt += 1;
            
            log(&format!("HTTP request attempt {}/{}", attempt, retry_config.max_retries + 1));

            // Send the request
            let response = match send_http(request) {
                Ok(resp) => resp,
                Err(e) => {
                    log(&format!("HTTP request failed: {}", e));
                    if attempt > retry_config.max_retries {
                        return Err(OpenAIError::HttpError(e));
                    }
                    
                    // Check if we've exceeded the total timeout
                    let elapsed = timing::now() - start_time;
                    if elapsed >= retry_config.max_total_timeout_ms as u64 {
                        log("Total retry timeout exceeded");
                        return Err(OpenAIError::HttpError(e));
                    }
                    
                    // Wait before retrying
                    log(&format!("Retrying after {} ms due to HTTP error", current_delay));
                    let _ = timing::sleep(current_delay as u64);
                    current_delay = std::cmp::min(
                        (current_delay as f64 * retry_config.backoff_multiplier) as u32,
                        retry_config.max_delay_ms
                    );
                    continue;
                }
            };

            // Check if we got a successful response
            if response.status == 200 {
                log(&format!("Request successful on attempt {}", attempt));
                return Ok(response);
            }

            // Check if this is a retryable error
            if !Self::is_retryable_error(response.status) {
                log(&format!("Non-retryable error: {}", response.status));
                return Ok(response); // Return the error response to be handled by caller
            }

            // Check if we've exhausted our retries
            if attempt > retry_config.max_retries {
                log(&format!("Max retries ({}) exceeded", retry_config.max_retries));
                return Ok(response);
            }

            // Check if we've exceeded the total timeout
            let elapsed = timing::now() - start_time;
            if elapsed >= retry_config.max_total_timeout_ms as u64 {
                log("Total retry timeout exceeded");
                return Ok(response);
            }

            // Log the retry attempt
            let message = String::from_utf8_lossy(&response.body.unwrap_or_default()).to_string();
            log(&format!(
                "Retryable error {} on attempt {}: {}",
                response.status, attempt, message
            ));
            log(&format!("Retrying after {} ms", current_delay));

            // Wait before retrying
            let _ = timing::sleep(current_delay as u64);
            
            // Update delay for next attempt (exponential backoff)
            current_delay = std::cmp::min(
                (current_delay as f64 * retry_config.backoff_multiplier) as u32,
                retry_config.max_delay_ms
            );
        }
    }

    /// List available models from the OpenAI API
    pub fn list_models(&self) -> Result<Vec<OpenAIModelInfo>, OpenAIError> {
        log("Using static OpenAI model list");
        
        // OpenAI doesn't provide detailed model information via API, so we return our curated list
        Ok(OpenAIModelInfo::get_available_models())
    }

    /// Generate a completion using the OpenAI API with retry logic
    pub fn generate_completion(
        &self,
        request: &OpenAICompletionRequest,
        retry_config: &RetryConfig,
        content_format: &ContentFormat,
    ) -> Result<OpenAICompletionResponse, OpenAIError> {
        log(&format!("Generating completion with model: {}", request.model));

        // Serialize the request using provider-aware format
        let request_json = request.serialize_for_provider(content_format);
        let request_body = serde_json::to_vec(&request_json)?;

        log(&format!(
            "OpenAI API request: {}",
            String::from_utf8_lossy(&request_body)
        ));

        let http_request = HttpRequest {
            method: "POST".to_string(),
            uri: format!("{}/chat/completions", self.base_url),
            headers: vec![
                ("authorization".to_string(), format!("Bearer {}", self.api_key)),
                ("content-type".to_string(), "application/json".to_string()),
                ("user-agent".to_string(), "moonshot-proxy/0.1.0".to_string()),
            ],
            body: Some(request_body),
        };

        let response = self.execute_with_retry(&http_request, retry_config)?;

        // Check status code
        if response.status != 200 {
            let message = String::from_utf8_lossy(&response.body.unwrap_or_default()).to_string();
            
            // Handle specific error cases
            return match response.status {
                401 => Err(OpenAIError::AuthenticationError(message)),
                429 => {
                    // Try to extract retry-after header
                    let retry_after = None; // TODO: Extract from headers if needed
                    Err(OpenAIError::RateLimitExceeded { retry_after })
                }
                _ => Err(OpenAIError::ApiError {
                    status: response.status,
                    message,
                }),
            };
        }

        // Parse the response
        let body = response
            .body
            .ok_or_else(|| OpenAIError::InvalidResponse("No response body".to_string()))?;

        log(&format!(
            "OpenAI API response: {}",
            String::from_utf8_lossy(&body)
        ));

        let completion: OpenAICompletionResponse = serde_json::from_slice(&body)?;

        log("Completion generated successfully");

        Ok(completion)
    }
}
