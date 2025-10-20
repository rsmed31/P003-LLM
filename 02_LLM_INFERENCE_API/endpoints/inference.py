import google.generativeai as genai
import os
import json
import requests
from dotenv import load_dotenv

# Handle both relative and absolute imports
try:
    from .prompt_builder import assemble_rag_prompt, assemble_rag_prompt_gemini
except ImportError:
    from prompt_builder import assemble_rag_prompt, assemble_rag_prompt_gemini

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

def callLlama(api_link, prompt, api_key=None, context=None):
    headers = {}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    
    payload = {
        "prompt": prompt
    }
    
    # Add context if provided (for RAG)
    if context:
        payload["context"] = context
    
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
        if use_rag:
            # Load system instructions and build a clean prompt (no chat tags)
            system_file = os.path.join(
                os.path.dirname(__file__),
                '..',
                'prompts',
                'prompts_run.txt'
            )
            with open(system_file, 'r', encoding='utf-8') as f:
                system_instructions = f.read().strip()

            combined_context = []
            if factual_data and factual_data.strip():
                combined_context.append("## PostgreSQL Facts:\n" + factual_data.strip())
            if filtered_context and filtered_context.strip():
                combined_context.append("## FAISS Retrieved Context:\n" + filtered_context.strip())
            context_body = "\n\n".join(combined_context) if combined_context else "No contextual data available."

            user_content = f"""# CONTEXTUAL KNOWLEDGE BASE:
{context_body}

# USER COMMAND:
{prompt}
"""
            model = configureGemini(api_key, config['model'], system_instructions=system_instructions)
            return callGemini(model, user_content)
        else:
            model = configureGemini(api_key, config['model'])
            return callGemini(model, prompt)
    else:
        # Local model
        if use_rag:
            # Combine context for local models
            combined_context = []
            if factual_data and factual_data.strip():
                combined_context.append("## PostgreSQL Facts:\n" + factual_data.strip())
            if filtered_context and filtered_context.strip():
                combined_context.append("## FAISS Retrieved Context:\n" + filtered_context.strip())
            
            context = "\n\n".join(combined_context) if combined_context else None
        
        return callLlama(api_link, prompt, api_key, context)

# Example usage
if __name__ == "__main__":
    model_name = "gemini"  # Change to "llama" to use local API

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