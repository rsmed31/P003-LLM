import google.generativeai as genai
import os
import json
import requests
from dotenv import load_dotenv

from prompt_builder import assemble_rag_prompt, assemble_rag_prompt_gemini, load_system_instructions, build_context_block

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
    response = model.generate_content(prompt)
    return response.text

def callLlama(api_link, prompt, api_key=None, model=None, stream=False):
    headers = {}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream
    }
    response = requests.post(api_link, json=payload, headers=headers)
    response.raise_for_status()
    return response.json().get('text', response.text)

def generate(
    model_name: str,
    prompt: str,
    context: str = None,
    factual_data: str = "",
    filtered_context: str = ""
) -> str:
    """
    Main inference function that routes to appropriate API based on model.
    Automatically handles RAG if model supports it and context data is provided.
    
    Args:
        model_name: "gemini" or any other name for local model
        prompt: The input prompt string (user query)
        context: Optional pre-formatted context string for local models
        factual_data: PostgreSQL factual retrieval results (for RAG)
        filtered_context: FAISS code-aware filtered chunks (for RAG)
    
    Returns:
        Generated text response
    """
    config = CONFIG.get(model_name, CONFIG['llama']) if model_name != "gemini" else CONFIG['gemini']
    
    # Get actual values from environment
    api_key = get_config_value(config, 'api_key')
    api_link = get_config_value(config, 'api_link')
    
    # Check if RAG should be applied
    use_rag = config.get('supports_rag', False) and (factual_data or filtered_context)
    
    if model_name == "gemini":
        # Load system instructions from JSON
        system_source = os.path.join(
            os.path.dirname(__file__),
            '..',
            'prompts',
            'prompts.json'
        )
        system_instructions = load_system_instructions(system_source)

        if use_rag:
            # Use prompt_builder to assemble the RAG prompt for Gemini
            assembled_prompt = assemble_rag_prompt_gemini(
                system_source,
                factual_data,
                filtered_context,
                prompt
            )
            model = configureGemini(api_key, config['model'], system_instructions=system_instructions)
            return callGemini(model, assembled_prompt)
        else:
            # Non-RAG Gemini still benefits from system prompt
            model = configureGemini(api_key, config['model'], system_instructions=system_instructions)
            return callGemini(model, prompt)
    else:
        # Local model (Ollama)
        stream_flag = config.get('stream', False)
        if use_rag:
            # Assemble the full RAG prompt with chat tags for local template
            system_source = os.path.join(
                os.path.dirname(__file__),
                '..',
                'prompts',
                'prompts.json'
            )
            prompt_to_send = assemble_rag_prompt(system_source, factual_data, filtered_context, prompt)
        else:
            prompt_to_send = prompt
        return callLlama(api_link, prompt_to_send, api_key, model=config.get('model'), stream=stream_flag)

# Example usage
if __name__ == "__main__":
    model_name = "llama"  # Change to "llama" to use local API

    # Test without RAG
    print("=== Test 1: Simple query without RAG ===")
    try:
        response = generate(
            model_name=model_name,
            prompt="What is OSPF?"
        )
        print(response)
    except Exception as e:
        print(f"Error: {e}")

    print("\n=== Test 2: Query with RAG ===")
    # Test with RAG (now providing sufficient interface/IP data)
    try:
        response = generate(
            model_name=model_name,
            prompt="Generate OSPF configuration for R1, R2, R3 using area 0.\nInterfaces:\nR1: GigabitEthernet0/0 = 10.0.0.1/24\nR2: GigabitEthernet0/1 = 10.0.0.2/24\nR3: GigabitEthernet0/2 = 10.0.0.3/24",
            factual_data="Router-IDs must be unique per router. Use OSPF process ID 1.",
            filtered_context="Example:\nrouter ospf 1\n router-id 1.1.1.1\n network 10.0.0.0 0.0.0.255 area 0"
        )
        print(response)
    except Exception as e:
        print(f"Error: {e}")