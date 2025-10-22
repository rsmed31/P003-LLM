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

# --- SETUP FROM keys.env ---
CONFIG = {
    'llama': {
        'model': 'zephyr_configurator',
        'api_link': os.getenv('LLAMA_API_LINK', 'http://localhost:11434/api/generate'),
        'api_key': os.getenv('LLAMA_API_KEY', ''),
        'supports_rag': True
    },
    'gemini': {
        'model': 'gemini-2.5-flash',
        'api_key': os.getenv('GEMINI_API_KEY', ''),
        'supports_rag': True
    }
}

# Configuration for external retrieval service (not localhost - external service)
RETRIEVAL_SERVICE_URL = os.getenv("RETRIEVAL_SERVICE_URL", "http://external-service:5000/api/retrieve")

def get_config_value(config, key):
    return config.get(key, "")

# Default orchestrator (optional, for local testing only)
try:
    from retrieval import RetrievalOrchestrator
    _DEFAULT_ORCHESTRATOR = RetrievalOrchestrator()
except Exception:
    _DEFAULT_ORCHESTRATOR = None

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

    # This function is mock as the local LLM is not accessible here
    print(f"--- Mock Llama Call: Hitting {api_link} with model {model} ---")
    # Mocking the structured JSON output we expect
    mock_response = """
[
  {
    "device_name": "R1",
    "protocol": "BGP",
    "configuration_mode_commands": [
      "configure terminal",
      "router bgp 65000",
      "router-id 1.1.1.1",
      "neighbor 192.168.12.2 remote-as 65000"
    ],
    "verification_command": "show ip bgp summary"
  },
  {
    "device_name": "R2",
    "protocol": "BGP",
    "configuration_mode_commands": [
      "configure terminal",
      "router bgp 65000",
      "router-id 2.2.2.2",
      "neighbor 192.168.12.1 remote-as 65000"
    ],
    "verification_command": "show ip bgp summary"
  }
]
"""
    return mock_response


# --- MAIN GENERATION LOGIC ---

def generate(
    model_name: str,
    prompt: str,
    context: str = None,
    factual_data: str = "",
    filtered_context: str = "",
    use_code_filter: bool = True,
    use_triage: bool = False,
    orchestrator = None
) -> str:
    """
    Main inference function with external retrieval service integration.
    
    Args:
        model_name: "gemini" or "llama"
        prompt: User's query
        filtered_context: Pre-fetched context (if already retrieved externally)
        use_code_filter: Apply code-aware filtering
        use_triage: DEPRECATED - use external retrieval service instead
        orchestrator: DEPRECATED - use external retrieval service
    
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

    # Fetch context from external retrieval service if not provided
    if not filtered_context and config.get("supports_rag", False):
        try:
            response = requests.get(
                RETRIEVAL_SERVICE_URL,
                params={"query": prompt, "number": 5},
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
        assembled_prompt = assemble_rag_prompt_gemini(
            system_source,
            filtered_context if use_rag else "",
            prompt,
            use_code_filter
        ) if use_rag else prompt
        return callGemini(model, assembled_prompt)
    else:
        # Local model (Ollama)
        if use_rag:
            # prompt_to_send will now be the fully assembled prompt including all RAG context
            prompt_to_send = assemble_rag_prompt(system_source, factual_data, filtered_context, prompt)
        else:
            prompt_to_send = prompt
        
        return callLlama(api_link, prompt_to_send, api_key, model=config.get('model'))

        # Local model
        stream_flag = config.get("stream", False)
        system_source = os.path.join(os.path.dirname(__file__), "..", "prompts", "prompts.json")
        
        prompt_to_send = (
            assemble_rag_prompt(system_source, filtered_context, prompt, use_code_filter)
            if use_rag
            else prompt
        )
        return callLlama(api_link, prompt_to_send, api_key, model=config.get("model"), stream=stream_flag)

# --- EXAMPLE USAGE ---

# if __name__ == "__main__":
#     model_name = "gemini"  # Set to use the local LLama/Zephyr model
    
#     # --- Define Placeholder Variables for FULL RAG Test (BGP Configuration) ---

#     # 1. Configuration Goal (User Query)
#     user_query = "Configure BGP peering between R1 and R2 using AS 65000. Use direct interface IPs for peering. Set router IDs manually."

#     # 2. Target Devices and Details (Factual/PostgreSQL Data)
#     target_devices_details = """
# * R1: Device Type: Router. Interfaces: {G0/0/0: 192.168.12.1/30, L0: 1.1.1.1/32}. BGP AS: 65000.
# * R2: Device Type: Router. Interfaces: {G0/0/0: 192.168.12.2/30, L0: 2.2.2.2/32}. BGP AS: 65000.
# """

#     # 3. Protocol Facts (Content/FAISS Data) + 4. VPP Error Report (combined for demonstration)
#     protocol_facts_and_errors = """
# **Protocol Facts:** The BGP command to start the process is 'router bgp <AS_number>'. The neighbor command is 'neighbor <remote_IP> remote-as <remote_AS>'. The router-id command is 'bgp router-id <id>' or 'router-id <id>' depending on the model/IOS version. Layer 2 commands (like 'switchport') can ONLY be applied to Switch devices.
# **VPP Error Report (OPTIONAL):** No errors reported in previous attempt. This is the first run.
# """
    
#     print("--- Test 1: Simple query without RAG (for baseline) ---")
#     try:
#         response = generate(
#             model_name=model_name,
#             prompt="What is BGP?",
#             factual_data="",
#             filtered_context=""
#         )
#         print(response)
#     except Exception as e:
#         print(f"Error: {e}")

#     print("\n--- Test 2: Query with FULL RAG context (BGP Test) ---")
#     try:
#         response = generate(
#             model_name=model_name,
#             prompt=user_query,
#             factual_data=target_devices_details,
#             filtered_context=protocol_facts_and_errors
#         )
#         print(response)
#     except Exception as e:
#         print(f"Error: {e}")

def generate_final_prompt(
    system_json_path: str,
    factual_data: str,
    context_data: str,
    user_goal: str,
    model_type: str = "gemini"
):
    """
    Generate and print the final network configuration prompt using the modular system.

    Parameters
    ----------
    system_json_path : str
        Path to the 'prompts.json' file containing all modular prompt parts.
    factual_data : str
        Factual data block (e.g., target device info from database).
    context_data : str
        Contextual or protocol-related data (e.g., retrieved from FAISS).
    user_goal : str
        The user's configuration goal (e.g., 'Enable OSPF on R1 and R2').
    model_type : str, optional
        Which model format to use for assembling the prompt:
        - "gemini" → plain text (for Google Gemini API)
        - "local" → uses <|system|> and <|user|> chat format (for Llama/Zephyr)
    """
    if model_type.lower() == "gemini":
        final_prompt = assemble_rag_prompt_gemini(
            system_file_path=system_json_path,
            factual_data_str=factual_data,
            filtered_context_str=context_data,
            user_query_str=user_goal
        )
    else:
        final_prompt = assemble_rag_prompt(
            system_file_path=system_json_path,
            factual_data_str=factual_data,
            filtered_context_str=context_data,
            user_query_str=user_goal
        )

    print(final_prompt)
    return final_prompt


# --- Example usage ---
if __name__ == "__main__":
    # Example inputs
    system_json_path = system_source = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'prompts.json')  # Path to your JSON file
    factual_data = """
    R1: Cisco ISR 4321, interfaces Gi0/0 and Gi0/1, connected to R2
    R2: Cisco ISR 4331, interfaces Gi0/0 and Gi0/1, connected to R1
    """
    context_data = """
    OSPF configuration requires router-id, network statements, and area IDs.
    Default verification command: 'show ip ospf neighbor'
    """
    user_goal = "Configure OSPF area 0 between R1 and R2 with router-ids 1.1.1.1 and 2.2.2.2 respectively."

    # Generate and print final prompt for Gemini
    generate_final_prompt(system_json_path, factual_data, context_data, user_goal, model_type="gemini")