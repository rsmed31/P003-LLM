import google.generativeai as genai
import os
import json
import requests
import datetime  # Add this import
import re  # <-- added
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
            "default_chunks": 50,
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

def get_chunk_count_for_query(query: str, model_config: dict = None) -> int:
    """
    Determine how many chunks to retrieve based on query content and model limits.
    Uses protocol-specific configuration or default, capped by model max_chunks.
    """
    protocol = detect_protocol_from_query(query)
    if protocol:
        base_count = RETRIEVAL_CONFIG["protocol_chunks"].get(protocol, RETRIEVAL_CONFIG["default_chunks"])
    else:
        base_count = RETRIEVAL_CONFIG["default_chunks"]
    
    # Apply model-specific max_chunks limit
    if model_config and 'max_chunks' in model_config:
        return min(base_count, model_config['max_chunks'])
    
    return base_count

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


def callLlama(api_link, prompt, api_key=None, model=None):
    """Call Llama (Ollama) using /api/chat for fully non-streaming JSON output."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    # Switch endpoint if old /api/generate is configured
    if api_link and "/api/generate" in api_link:
        api_link = api_link.replace("/api/generate", "/api/chat")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False  # enforce non-streaming
    }
    response = requests.post(api_link, json=payload, headers=headers, timeout=120)
    response.raise_for_status()
    data = response.json()
    return (data.get("message", {}) or {}).get("content", "").strip()

def clean_model_output(text: str) -> str:
    """
    Extract ONLY the first complete top-level JSON array from model output.
    Discards any trailing instructional text the model may append.
    Returns '[]' if no valid array detected.
    """
    if not text:
        return "[]"
    # Remove code fences
    t = re.sub(r"```(?:json)?", "", text).strip()

    # Find first '['
    start = t.find('[')
    if start == -1:
        return "[]"

    depth = 0
    in_string = False
    escape = False
    end = -1

    for i, ch in enumerate(t[start:], start):
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:
                end = i
                break

    if end == -1 or depth != 0:
        return "[]"

    json_portion = t[start:end + 1].strip()
    return json_portion

def parse_and_validate_array(json_text: str):
    try:
        obj = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model output is not valid JSON: {e}\nOutput was:\n{json_text}")

    if not isinstance(obj, list):
        raise ValueError("Model output is valid JSON but not a JSON array.")

    # All fields are mandatory
    required_keys = ["device_name", "configuration_mode_commands", "protocol", "intent"]
    
    for i, entry in enumerate(obj):
        if not isinstance(entry, dict):
            logger.error(f"[VALIDATION] Element {i} is not a dict, type={type(entry)}")
            raise ValueError(f"Array element {i} is not an object.")
        
        # Check all required keys are present
        missing_keys = [k for k in required_keys if k not in entry]
        if missing_keys:
            logger.error(f"[VALIDATION] Element {i} FAILED validation")
            logger.error(f"[VALIDATION] Required keys: {required_keys}")
            logger.error(f"[VALIDATION] Element actual keys: {list(entry.keys())}")
            logger.error(f"[VALIDATION] Missing keys: {missing_keys}")
            logger.error(f"[VALIDATION] Element content (first 800 chars): {json.dumps(entry, indent=2)[:800]}")
            logger.error(f"[VALIDATION] Full cleaned response (first 1500 chars): {json_text[:1500]}")
            logger.error(f"[VALIDATION] Full array length: {len(obj)}")
            raise ValueError(f"Array element {i} missing required keys: {', '.join(missing_keys)}")
    
    return obj

# --- MAIN GENERATION LOGIC ---

# Initialize local orchestrator for correlation analysis only
# External service provides raw chunks, local orchestrator analyzes them
try:
    from retrieval import RetrievalOrchestrator
    _LOCAL_ORCHESTRATOR = RetrievalOrchestrator()
except Exception:
    _LOCAL_ORCHESTRATOR = None

# Import retrieval strategy and code filter
try:
    import sys
    import os
    rag_logic_path = os.path.join(os.path.dirname(__file__), '..', 'rag_logic')
    if rag_logic_path not in sys.path:
        sys.path.append(rag_logic_path)
    from retrieval_strategy import RetrievalStrategy
    from code_aware_filter import filter_and_refine_context, extract_cli_heavy_chunks, refine_code_chunks
    _RETRIEVAL_STRATEGY = RetrievalStrategy()
except Exception as e:
    print(f"[WARN] Could not load retrieval strategy: {e}")
    _RETRIEVAL_STRATEGY = None
    filter_and_refine_context = None
    extract_cli_heavy_chunks = None
    refine_code_chunks = None

# NOTE (#file:prompts.json): If prompts.json load fails we silently continue without system instructions.
def safe_load_prompts(path: str) -> dict:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {"SYSTEM_PROMPT": "", "CRITICAL_RULE_INSTRUCTION": "", "TASK_INSTRUCTION": ""}

import logging
logger = logging.getLogger("team2_inference")

# --- CONTEXT STORAGE ---
def save_context_to_file(query: str, chunks: list, built_context: str, model_name: str):
    """
    Save query context to models/context.json for debugging and analysis.
    
    Args:
        query: User's original query
        chunks: List of retrieved chunk texts
        built_context: Final assembled context (with CODE/THEORY sections)
        model_name: Model that will use this context
    """
    context_file = os.path.join(os.path.dirname(__file__), '..', 'models', 'context.json')
    
    try:
        # Load existing history
        if os.path.exists(context_file):
            with open(context_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {"history": []}
        
        # Create entry
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "model": model_name,
            "query": query,
            "chunks_retrieved": len(chunks),
            "chunks": [
                {
                    "index": i + 1,
                    "text": chunk[:300] + "..." if len(chunk) > 300 else chunk
                }
                for i, chunk in enumerate(chunks)
            ],
            "built_context": built_context,
            "context_length": len(built_context),
            "retrieval_plan": getattr(save_context_to_file, '_last_plan', {})
        }
        
        # Clear the plan for next call
        if hasattr(save_context_to_file, '_last_plan'):
            delattr(save_context_to_file, '_last_plan')
        
        # Append to history (keep last 50 entries)
        data["history"].append(entry)
        if len(data["history"]) > 50:
            data["history"] = data["history"][-50:]
        
        # Write back
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"[INFO] Context saved to context.json (entry #{len(data['history'])})")
    
    except Exception as e:
        logger.warning(f"Failed to save context to file: {e}")

def generate(
    query: str,
    model_name: str = "gemini",
    loopback: bool = False   # NEW: loopback mode disables retrieval/RAG
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

    system_source = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'prompts.json')
    _prompts_raw = safe_load_prompts(system_source)

    # Determine dynamic chunk count unless loopback
    chunk_count = 0 if loopback else get_chunk_count_for_query(query, config)
    print(f"[INFO] loopback={loopback} chunk_count={chunk_count} model='{model_name}'")

    filtered_context = ""
    retrieved_chunks = []  # NEW: Store chunks for logging
    
    if not loopback and config.get("supports_rag", False) and RETRIEVAL_SERVICE_URL:
        try:
            response = requests.get(
                f"{RETRIEVAL_SERVICE_URL}/chunks/query",
                params={"query": query, "limit": chunk_count},
                timeout=1000
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("found", False):
                    results = data.get("results", [])
                    chunks = [item["text"] for item in results if "text" in item][:chunk_count]
                    retrieved_chunks = chunks  # Store for logging
                    
                    print(f"[INFO] Retrieved {len(chunks)} chunks from service (limit: {chunk_count})")
                    
                    # NEW: Strategy-based processing with structured context
                    if chunks and _LOCAL_ORCHESTRATOR and _RETRIEVAL_STRATEGY:
                        # Add chunks to orchestrator
                        _LOCAL_ORCHESTRATOR.add_chunks(chunks)
                        
                        # Get retrieval plan for this query
                        plan = _RETRIEVAL_STRATEGY.get_plan_for_query(query, len(chunks))
                        print(f"[INFO] Retrieval Plan: {plan.total_chunks} chunks (code: {plan.target_code_chunks}, theory: {plan.target_theory_chunks})")
                        
                        # Retrieve with correlation
                        correlation_result = _LOCAL_ORCHESTRATOR.retrieve_with_correlation(query, plan.total_chunks)
                        
                        # Separate code and theory chunks
                        code_chunks_raw = [c["chunk"] for c in correlation_result["chunks"] if c["type"] == "code"]
                        theory_chunks_raw = [c["chunk"] for c in correlation_result["chunks"] if c["type"] == "theory"]
                        
                        # Apply code-aware filtering to code chunks
                        if filter_and_refine_context and code_chunks_raw:
                            code_refined = filter_and_refine_context(code_chunks_raw, aggressive=False)
                            if code_refined:
                                code_chunks = [code_refined]
                            else:
                                code_chunks = code_chunks_raw
                        else:
                            code_chunks = code_chunks_raw
                        
                        theory_chunks = theory_chunks_raw
                        
                        # Build structured context
                        context_parts = []
                        
                        if code_chunks:
                            code_header = "## CODE CONTEXT"
                            code_body = "\n\n---\n\n".join(code_chunks)
                            context_parts.append(f"{code_header}\n\n{code_body}")
                        
                        if theory_chunks:
                            theory_header = "## THEORY CONTEXT"
                            theory_body = "\n\n---\n\n".join(theory_chunks)
                            context_parts.append(f"{theory_header}\n\n{theory_body}")
                        
                        filtered_context = "\n\n".join(context_parts) if context_parts else ""
                        
                        # Log metrics
                        print(f"[INFO] ═══ RAG METRICS ═══")
                        print(f"[INFO] Correlation Score: {correlation_result.get('correlation_score', 'N/A')}")
                        print(f"[INFO] Context Quality: {correlation_result.get('context_quality_score', 'N/A')}")
                        print(f"[INFO] Overall Confidence: {correlation_result.get('overall_confidence', 'N/A')}")
                        print(f"[INFO] Code chunks: {len(code_chunks)}, Theory chunks: {len(theory_chunks)}")
                        print(f"[INFO] Target code/theory: {plan.target_code_chunks}/{plan.target_theory_chunks}")
                        print(f"[INFO] ═══════════════════")
                        
                        # Store plan metadata for context.json
                        save_context_to_file._last_plan = {
                            "total_chunks": plan.total_chunks,
                            "code_chunks": len(code_chunks),
                            "theory_chunks": len(theory_chunks),
                            "target_code": plan.target_code_chunks,
                            "target_theory": plan.target_theory_chunks,
                            "code_ratio": plan.code_ratio,
                            "theory_ratio": plan.theory_ratio,
                            "correlation_score": correlation_result.get('correlation_score'),
                            "quality_score": correlation_result.get('context_quality_score'),
                            "confidence": correlation_result.get('overall_confidence')
                        }
                    elif chunks:
                        # Fallback: simple concatenation
                        filtered_context = "\n\n---\n\n".join(chunks)
                    else:
                        filtered_context = ""
                else:
                    print(f"[WARN] No matching chunks found for query")
        except Exception as e:
            print(f"[WARN] Error processing retrieval: {e}")
            import traceback
            traceback.print_exc()

    # Force no context if loopback
    if loopback:
        filtered_context = ""

    # Save context to file (only if RAG was used and context exists)
    if not loopback and filtered_context:
        save_context_to_file(query, retrieved_chunks, filtered_context, model_name)

    # Generate response via LLM
    if model_name == "gemini":
        system_instructions = load_system_instructions(system_source)
        model = configureGemini(api_key, config['model'], system_instructions=system_instructions)
        assembled_prompt = assemble_rag_prompt_gemini(system_source, filtered_context, query)
        unCleanedResponse = callGemini(model, assembled_prompt)
        cleanedResponse = clean_model_output(unCleanedResponse)
        logger.debug(f"[GENERATE] Raw model response (first 500 chars): {unCleanedResponse[:500]}")
        logger.debug(f"[GENERATE] Cleaned response (first 500 chars): {cleanedResponse[:500]}")
        
        logger.info(f"[GENERATE] About to validate response. loopback={loopback} model={model_name}")
        logger.debug(f"[GENERATE] Raw API response length: {len(unCleanedResponse) if isinstance(unCleanedResponse, str) else 'N/A'}")
        logger.debug(f"[GENERATE] Cleaned response preview (first 1000 chars): {cleanedResponse[:1000]}")
        
        try:
            parsedObject = parse_and_validate_array(cleanedResponse)
        except ValueError as ve:
            logger.error(f"[GENERATE] Validation failed: {ve}")
            logger.error(f"[GENERATE] Query was: '{query}'")
            logger.error(f"[GENERATE] Model: {model_name}")
            logger.error(f"[GENERATE] Loopback: {loopback}")
            raise
        
        return json.dumps({"model": model_name, "response": parsedObject}, indent=2)

    # Llama path
    prompt_to_send = assemble_rag_prompt(system_source, filtered_context, query)
    unCleanedResponse = callLlama(api_link, prompt_to_send, api_key=api_key, model=config.get('model'))
    cleanedResponse = clean_model_output(unCleanedResponse)
    parsedObject = parse_and_validate_array(cleanedResponse)
    return json.dumps({"model": model_name, "response": parsedObject}, indent=2)


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