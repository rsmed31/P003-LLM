import google.generativeai as genai
import os
import json
import requests
from dotenv import load_dotenv
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from prompt_builder import assemble_rag_prompt, assemble_rag_prompt_gemini, load_system_instructions

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'keys.env')
load_dotenv(env_path)

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'inference.json')
with open(config_path, 'r') as f:
    CONFIG = json.load(f)


def get_config_value(config, key):
    """Get configuration value from environment or config directly."""
    env_key = config.get(f"{key}_env")
    if env_key:
        return os.getenv(env_key, config.get(key, ""))
    return config.get(key, "")


def configureGemini(apiKey, model_name, system_instructions=None):
    genai.configure(api_key=apiKey)
    if system_instructions:
        model = genai.GenerativeModel(model_name=model_name, system_instruction=system_instructions)
    else:
        model = genai.GenerativeModel(model_name=model_name)
    return model


def callGemini(model, prompt):
    """Call Gemini and return clean JSON."""
    response = model.generate_content(prompt)
    result = {
        "model": model.model_name,
        "response": response.text.strip() if response and hasattr(response, "text") else ""
    }
    return json.dumps(result, indent=2)


def callLlama(api_link, prompt, api_key=None, model=None, stream=False):
    """Call Llama API (Ollama, etc.) and return clean JSON."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream
    }
    response = requests.post(api_link, json=payload, headers=headers)
    response.raise_for_status()

    data = response.json()

    # Normalize keys in case model name differs or text is nested
    model_name = data.get("model", model or "unknown_model")
    text_output = data.get("response") or data.get("text") or data.get("output") or ""

    # Clean output
    clean = {
        "model": model_name,
        "response": text_output.strip()
    }
    return json.dumps(clean, indent=2)


def generate(
    model_name: str,
    prompt: str,
    context: str = None,
    factual_data: str = "",
    filtered_context: str = "",
    use_code_filter: bool = True,
    use_triage: bool = True,
    orchestrator = None  # Injected from app.py
) -> str:
    """
    Main inference function with integrated retrieval triage.
    
    Args:
        model_name: "gemini" or local model name
        prompt: User's query
        use_code_filter: Apply code-aware filtering
        use_triage: Use retrieval orchestrator triage logic
        orchestrator: RetrievalOrchestrator instance (injected)
    
    Returns:
        JSON string with {model, response} or direct definitive answer
    """
    # Import here to avoid circular dependency
    if orchestrator is None:
        from app import RETRIEVAL_ORCHESTRATOR
        orchestrator = RETRIEVAL_ORCHESTRATOR
    
    config = CONFIG.get(model_name, CONFIG["llama"]) if model_name != "gemini" else CONFIG["gemini"]
    
    api_key = get_config_value(config, "api_key")
    api_link = get_config_value(config, "api_link")
    
    # Execute retrieval triage if enabled
    if use_triage and config.get("supports_rag", False):
        triage_result = orchestrator.retrieve_with_triage(prompt)
        
        if triage_result["route"] == "definitive":
            return json.dumps({
                "model": "retrieval_triage",
                "response": triage_result["factual_data"],
                "confidence": triage_result["confidence"],
                "route": "definitive"
            }, indent=2)
        
        filtered_context = triage_result["filtered_context"]
        factual_data = triage_result["factual_data"]
        use_rag = True
    else:
        use_rag = config.get("supports_rag", False) and (factual_data or filtered_context)
    
    # Generate response via LLM
    if model_name == "gemini":
        system_source = os.path.join(os.path.dirname(__file__), "..", "prompts", "prompts.json")
        system_instructions = load_system_instructions(system_source)
        model = configureGemini(api_key, config["model"], system_instructions=system_instructions)
        
        assembled_prompt = (
            assemble_rag_prompt_gemini(system_source, filtered_context, prompt, use_code_filter)
            if use_rag
            else prompt
        )
        return callGemini(model, assembled_prompt)
    else:
        # Local model
        stream_flag = config.get("stream", False)
        system_source = os.path.join(os.path.dirname(__file__), "..", "prompts", "prompts.json")
        
        prompt_to_send = (
            assemble_rag_prompt(system_source, filtered_context, prompt, use_code_filter)
            if use_rag
            else prompt
        )
        return callLlama(api_link, prompt_to_send, api_key, model=config.get("model"), stream=stream_flag)
