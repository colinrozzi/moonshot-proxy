#[cfg(test)]
mod tests {
    use crate::types::*;

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
}
