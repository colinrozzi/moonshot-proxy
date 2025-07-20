use crate::api::OpenAIClient;
use crate::types::OpenAICompletionRequest;
use crate::types::response::OpenAIModelInfo;
use crate::bindings::colinrozzi::genai_types::types::{ProxyRequest, ProxyResponse};
use crate::bindings::theater::simple::runtime::log;
use crate::types::state::State;
//use genai_types::{ProxyRequest, ProxyResponse};

pub fn handle_request(
    data: Vec<u8>,
    state_bytes: Vec<u8>,
) -> Result<(Option<Vec<u8>>, (Option<Vec<u8>>,)), String> {
    log("Handling request in moonshot-proxy actor");

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
            let error_response = ProxyResponse::Error(format!("Invalid request format: {}", e));

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
        ProxyRequest::GenerateCompletion(request) => {
            log(&format!(
                "Generating completion with model: {}",
                request.model
            ));

            // Validate that the model is supported
            if !OpenAIModelInfo::is_model_supported(&request.model) {
                let suggestions = OpenAIModelInfo::get_model_suggestions(&request.model);
                let error_msg = if !suggestions.is_empty() {
                    format!(
                        "Unsupported model '{}'. Did you mean one of: {}? Available models can be listed using the ListModels request.",
                        request.model,
                        suggestions.join(", ")
                    )
                } else {
                    format!(
                        "Unsupported model '{}'. Please check the available models using the ListModels request.",
                        request.model
                    )
                };
                
                log(&format!("Model validation failed: {}", error_msg));
                return Ok((Some(state_bytes), (Some(serde_json::to_vec(&ProxyResponse::Error(error_msg)).unwrap()),)));
            }

            let openai_request: OpenAICompletionRequest = request.into();
            match client.generate_completion(
                &openai_request,
                &state.config.retry_config,
                &state.config.content_format,
            ) {
                Ok(completion) => ProxyResponse::Completion(completion.into()),
                Err(e) => {
                    log(&format!("Error generating completion: {}", e));
                    ProxyResponse::Error(format!("Failed to generate completion: {}", e))
                }
            }
        }

        ProxyRequest::ListModels => {
            log("Listing available models");

            match client.list_models() {
                Ok(models) => {
                    ProxyResponse::ListModels(models.into_iter().map(|m| m.into()).collect())
                }
                Err(e) => {
                    log(&format!("Error listing models: {}", e));
                    ProxyResponse::Error(format!("Failed to list models: {}", e))
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
