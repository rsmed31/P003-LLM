import google.generativeai as genai
import os
import json
import requests
from dotenv import load_dotenv
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables first
load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'models', 'keys.env'))

from prompt_builder import assemble_rag_prompt, assemble_rag_prompt_gemini, load_system_instructions

# --- SETUP FROM keys.env (no hardcoded defaults) ---
CONFIG = {
    'llama': {
        'model': 'zephyr_configurator',
        'api_link': os.getenv('LLAMA_API_LINK'),
        'api_key': os.getenv('LLAMA_API_KEY'),
        'supports_rag': True
    },
    'gemini': {
        'model': 'gemini-2.5-flash',
        'api_key': os.getenv('GEMINI_API_KEY'),
        'supports_rag': True
    }
}

# Configuration for external retrieval service (from env only)
RETRIEVAL_SERVICE_URL = os.getenv("RETRIEVAL_SERVICE_URL")

def get_config_value(config, key):
    return config.get(key, "")

# --- LLM CALL FUNCTIONS ---

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


# --- MAIN GENERATION LOGIC ---

def generate(
    query: str,
    model_name: str = "gemini"
) -> str:
    """
    Main inference function - simplified interface.
    
    Args:
        query: User's query/configuration goal
        model_name: "gemini" or "llama" (default: "gemini")
    
    Returns:
        JSON string with {model, response}
    """
    # Resolve model config
    config = CONFIG.get(model_name, CONFIG["llama"])

    # Resolve API credentials
    if model_name == "gemini":
        api_key = config.get('api_key')
        api_link = None
    else:
        api_key = config.get('api_key')
        api_link = config.get('api_link')

    # Path to system instructions
    system_source = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'prompts.json')

    # Fetch context from external retrieval service (endpoint only)
    filtered_context = ""
    if config.get("supports_rag", False) and RETRIEVAL_SERVICE_URL:
        try:
            response = requests.get(
                RETRIEVAL_SERVICE_URL,
                params={"query": query, "number": 5},
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                # External service returns { "chunks": [...] }
                chunks = data.get("chunks", [])
                filtered_context = "\n---\n".join(chunks) if chunks else ""
        except Exception as e:
            print(f"Warning: Failed to retrieve context from external service: {e}")
            filtered_context = ""

    # Determine whether to use RAG
    use_rag = config.get('supports_rag', False) and bool(filtered_context)

    # Generate response via LLM
    if model_name == "gemini":
        system_instructions = load_system_instructions(system_source)
        model = configureGemini(api_key, config['model'], system_instructions=system_instructions)
        
        if use_rag:
            assembled_prompt = assemble_rag_prompt_gemini(
                system_source,
                filtered_context,
                query
            )
        else:
            assembled_prompt = query
            
        return callGemini(model, assembled_prompt)
    else:
        # Local model (Ollama/Llama)
        if use_rag:
            prompt_to_send = assemble_rag_prompt(
                system_source,
                filtered_context,
                query
            )
        else:
            prompt_to_send = query
        
        return callLlama(api_link, prompt_to_send, api_key=api_key, model=config.get('model'))


# --- EXAMPLE USAGE ---

if __name__ == "__main__":
    # Example 1: Simple query without RAG
    print("--- Test 1: Simple query ---")
    try:
        response = generate(
            query="What is BGP?",
            model_name="gemini"
        )
        print(response)
    except Exception as e:
        print(f"Error: {e}")

    print("\n--- Test 2: Configuration query with RAG ---")
    try:
        response = generate(
            query="Configure OSPF area 0 between R1 and R2 with router-ids 1.1.1.1 and 2.2.2.2 respectively.",
            model_name="gemini"
        )
        print(response)
    except Exception as e:
        print(f"Error: {e}")