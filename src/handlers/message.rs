use crate::api::OpenAIClient;
use crate::bindings::theater::simple::runtime::log;
use crate::types::state::State;
use genai_types::{ProxyRequest, ProxyResponse};

pub fn handle_request(
    data: Vec<u8>,
    state_bytes: Vec<u8>,
) -> Result<(Option<Vec<u8>>, (Option<Vec<u8>>,)), String> {
    log("Handling request in openai-proxy actor");

    // Parse the state
    let state: State = match serde_json::from_slice(&state_bytes) {
        Ok(s) => s,
        Err(e) => {
            log(&format!("Error parsing state: {}", e));
            return Err(format!("Failed to parse state: {}", e));
        }
    };

    // Debug log the incoming request
    log(&format!(
        "Received request data: {}",
        String::from_utf8_lossy(&data)
    ));

    // Parse the request using the shared ProxyRequest type
    let request: ProxyRequest = match serde_json::from_slice(&data) {
        Ok(req) => req,
        Err(e) => {
            log(&format!("Error parsing request: {}", e));

            // Try to respond with a properly formatted error
            let error_response = ProxyResponse::Error {
                error: format!("Invalid request format: {}", e),
            };

            match serde_json::to_vec(&error_response) {
                Ok(bytes) => return Ok((Some(state_bytes), (Some(bytes),))),
                Err(_) => return Err(format!("Invalid request format: {}", e)),
            }
        }
    };

    // Create OpenAI client with configurable base URL
    let client = match &state.config.base_url {
        Some(base_url) => OpenAIClient::new_with_base_url(state.api_key.clone(), base_url.clone()),
        None => OpenAIClient::new(state.api_key.clone()),
    };

    // Process based on operation type
    let response = match request {
        ProxyRequest::GenerateCompletion { request } => {
            log(&format!(
                "Generating completion with model: {}",
                request.model
            ));

            match client.generate_completion(&request.into(), &state.config.retry_config, &state.config.content_format) {
                Ok(completion) => ProxyResponse::Completion {
                    completion: completion.into(),
                },
                Err(e) => {
                    log(&format!("Error generating completion: {}", e));
                    ProxyResponse::Error {
                        error: format!("Failed to generate completion: {}", e),
                    }
                }
            }
        }

        ProxyRequest::ListModels => {
            log("Listing available models");

            match client.list_models() {
                Ok(models) => ProxyResponse::ListModels {
                    models: models.into_iter().map(|m| m.into()).collect(),
                },
                Err(e) => {
                    log(&format!("Error listing models: {}", e));
                    ProxyResponse::Error {
                        error: format!("Failed to list models: {}", e),
                    }
                }
            }
        }
    };

    // Serialize the response
    let response_bytes = match serde_json::to_vec(&response) {
        Ok(bytes) => bytes,
        Err(e) => {
            log(&format!("Error serializing response: {}", e));
            return Err(format!("Failed to serialize response: {}", e));
        }
    };

    // Return the updated state and response
    Ok((Some(state_bytes), (Some(response_bytes),)))
}
