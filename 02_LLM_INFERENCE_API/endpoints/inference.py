import google.generativeai as genai
import os
import json
import requests
from dotenv import load_dotenv
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# Add current directory so absolute import 'prompt_builder' works when run directly
sys.path.append(os.path.dirname(__file__))

# Load environment variables first
load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'models', 'keys.env'))

# Use package-relative import when available, fallback to absolute for direct runs
try:
    from .prompt_builder import assemble_rag_prompt, assemble_rag_prompt_gemini, load_system_instructions
except Exception:
    from prompt_builder import assemble_rag_prompt, assemble_rag_prompt_gemini, load_system_instructions

# --- LOAD CONFIG FROM inference.json ---
def load_config():
    """Load model configuration from inference.json"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'inference.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Resolve environment variables for API keys and links
    config = {}
    for model_name, model_config in data.items():
        config[model_name] = {
            'model': model_config.get('model'),
            'supports_rag': model_config.get('supports_rag', False)
        }
        
        # Resolve API key from environment
        if 'api_key_env' in model_config:
            config[model_name]['api_key'] = os.getenv(model_config['api_key_env'])
        
        # Resolve API link from environment (for local models)
        if 'api_link_env' in model_config:
            config[model_name]['api_link'] = os.getenv(model_config['api_link_env'])
        
        # Additional settings for local models
        if 'stream' in model_config:
            config[model_name]['stream'] = model_config['stream']
        if 'context_param' in model_config:
            config[model_name]['context_param'] = model_config['context_param']
    
    return config

CONFIG = load_config()

# Configuration for external retrieval service (from env only)
# This is an external service endpoint - not part of this codebase
# Example: http://192.168.103.100:8000/api/retrieve
RETRIEVAL_SERVICE_URL = os.getenv("RETRIEVAL_SERVICE_URL")

# --- LOAD RETRIEVAL CONFIG ---
def load_retrieval_config():
    """Load retrieval configuration with protocol-specific chunk counts"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'retrieval_config.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback defaults
        return {
            "default_chunks": 75,
            "protocol_chunks": {}
        }

RETRIEVAL_CONFIG = load_retrieval_config()

def detect_protocol_from_query(query: str) -> str:
    """
    Detect protocol mentioned in query to determine chunk count.
    Returns protocol name in lowercase or None.
    """
    query_lower = query.lower()
    # Check against known protocols
    for protocol in RETRIEVAL_CONFIG.get("protocol_chunks", {}).keys():
        if protocol in query_lower:
            return protocol
    return None

def get_chunk_count_for_query(query: str) -> int:
    """
    Determine how many chunks to retrieve based on query content.
    Uses protocol-specific configuration or default.
    """
    protocol = detect_protocol_from_query(query)
    if protocol:
        return RETRIEVAL_CONFIG["protocol_chunks"].get(protocol, RETRIEVAL_CONFIG["default_chunks"])
    return RETRIEVAL_CONFIG["default_chunks"]

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
    # result = {
    #     "model": model.model_name,
    #     "response": response.text.strip() if response and hasattr(response, "text") else ""
    # }
    # return json.dumps(result, indent=2)
    return response.text.strip() if response and hasattr(response, "text") else ""


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
    # clean = {
    #     "model": model_name,
    #     "response": text_output.strip()
    # }
    # return json.dumps(clean, indent=2)
    return text_output

def clean_model_output(text: str) -> str:
    """
    Aggressively remove common wrappers the model adds.
    Handles cases where model output might be fragmented or have trailing text.
    """
    if not text:
        return text

    t = text.strip()

    # First, try to extract content between first '[' and last ']'
    first_bracket = t.find('[')
    last_bracket = t.rfind(']')
    
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        # Extract the JSON array portion
        json_portion = t[first_bracket:last_bracket + 1]
        
        # Remove any markdown code fences that might be inside
        if '```' in json_portion:
            # Remove all ``` markers
            json_portion = json_portion.replace('```json', '').replace('```', '')
        
        t = json_portion.strip()
    
    # Remove leading 'json' keyword if present
    if t.lower().startswith("json"):
        t = t[4:].strip()
    
    # Final validation: ensure it starts with [ and ends with ]
    if not t.startswith('['):
        idx = t.find('[')
        if idx != -1:
            t = t[idx:]
        else:
            # No opening bracket found - this is an error
            return t
    
    if not t.endswith(']'):
        idx = t.rfind(']')
        if idx != -1:
            t = t[:idx + 1]
        else:
            # No closing bracket found - this is an error
            return t

    return t

def parse_and_validate_array(json_text: str):
    try:
        obj = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model output is not valid JSON: {e}\nOutput was:\n{json_text}")

    if not isinstance(obj, list):
        raise ValueError("Model output is valid JSON but not a JSON array.")

    # Basic per-item checks
    for i, entry in enumerate(obj):
        if not isinstance(entry, dict):
            raise ValueError(f"Array element {i} is not an object.")
        if 'device_name' not in entry or 'protocol' not in entry or 'configuration_mode_commands' not in entry:
            raise ValueError(f"Array element {i} missing required keys.")
        # you can add schema enforcement here...
    return obj

# --- MAIN GENERATION LOGIC ---

# Initialize local orchestrator for correlation analysis only
# External service provides raw chunks, local orchestrator analyzes them
try:
    from retrieval import RetrievalOrchestrator
    _LOCAL_ORCHESTRATOR = RetrievalOrchestrator()
except Exception:
    _LOCAL_ORCHESTRATOR = None

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

    # Determine dynamic chunk count based on query
    chunk_count = get_chunk_count_for_query(query)

    # Fetch context from external retrieval service (endpoint only)
    filtered_context = ""
    if config.get("supports_rag", False) and RETRIEVAL_SERVICE_URL:
        try:
            response = requests.get(
                RETRIEVAL_SERVICE_URL,
                params={"query": query, "number": chunk_count},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                # External service returns { "chunks": [...] }
                chunks = data.get("chunks", [])
                
                # Perform local correlation analysis on received chunks
                if chunks and _LOCAL_ORCHESTRATOR:
                    # Temporarily add chunks to local orchestrator for analysis
                    _LOCAL_ORCHESTRATOR.add_chunks(chunks)
                    
                    # Get correlation-analyzed results
                    correlation_result = _LOCAL_ORCHESTRATOR.retrieve_with_correlation(query, len(chunks))
                    
                    # Rebuild context with proper ordering (code + theory sections)
                    code_chunks = [c["chunk"] for c in correlation_result["chunks"] if c["type"] == "code"]
                    theory_chunks = [c["chunk"] for c in correlation_result["chunks"] if c["type"] == "theory"]
                    
                    # Build separated context
                    context_parts = []
                    if code_chunks:
                        context_parts.append("## CODE-AWARE CONTEXT:\n" + "\n---\n".join(code_chunks))
                    if theory_chunks:
                        context_parts.append("## THEORETICAL CONTEXT:\n" + "\n---\n".join(theory_chunks))
                    
                    filtered_context = "\n\n".join(context_parts) if context_parts else ""
                    
                    # Log correlation metrics
                    print(f"Correlation Score: {correlation_result['correlation_score']}")
                    print(f"Overall Confidence: {correlation_result['overall_confidence']}")
                else:
                    # Fallback: simple join if orchestrator unavailable
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
        
        if True:
            assembled_prompt = assemble_rag_prompt_gemini(
                system_source,
                filtered_context,
                query
            )
        else:
            assembled_prompt = query
        # print(assembled_prompt)
        unCleanedResponse = callGemini(model, assembled_prompt)
        # print(f"The model uncleaned response: {unCleanedResponse}")
        cleanedResponse = clean_model_output(unCleanedResponse)
        parsedObject = parse_and_validate_array(cleanedResponse)
        clean = {
            "model": model_name,
            "response": parsedObject
        }
        return json.dumps(clean, indent=2)
    else:
        # Local model (Ollama/Llama)
        if True:
            prompt_to_send = assemble_rag_prompt(
                system_source,
                filtered_context,
                query
            )
        else:
            prompt_to_send = query
        print(prompt_to_send)
        unCleanedResponse = callLlama(api_link, prompt_to_send, api_key=api_key, model=config.get('model'))
        # print(f"The model uncleaned response: {unCleanedResponse}")
        cleanedResponse = clean_model_output(unCleanedResponse)
        parsedObject = parse_and_validate_array(cleanedResponse)
        # Clean output
        clean = {
            "model": model_name,
            "response": parsedObject
        }
        return json.dumps(clean, indent=2)


# --- EXAMPLE USAGE ---

if __name__ == "__main__":
    print("\n--- Test 2: Configuration query with RAG ---")
    try:
        response = generate(
            query="Configure OSPF area 0 between R1 and R2 with router-ids 1.1.1.1 and 2.2.2.2 respectively.",
            model_name="llama"
        )
        print(response)
    except Exception as e:
        print(f"Error: {e}")