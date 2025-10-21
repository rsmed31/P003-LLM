import google.generativeai as genai
import os
import json
import requests
from dotenv import load_dotenv

# Assuming prompt_builder.py is in the same directory (or accessible via path)
from prompt_builder import assemble_rag_prompt, assemble_rag_prompt_gemini, load_system_instructions, build_context_block

# --- SETUP (PLACEHOLDER CONFIGURATION) ---
# NOTE: In a real environment, you would set up your keys.env and inference.json files.
# Mocking a basic CONFIG structure for the script to run:
CONFIG = {
    'llama': {
        'model': 'zephyr_configurator',
        'api_link': 'http://localhost:11434/api/generate',
        'supports_rag': True
    },
    'gemini': {
        'model': 'gemini-2.5-flash',
        'api_key_env': 'GEMINI_API_KEY',
        'supports_rag': True
    }
}
# Mocking a function to get config values
def get_config_value(config, key):
    return config.get(key, "")

# --- LLM CALL FUNCTIONS (as provided by user) ---

def configureGemini(apiKey, model_name, system_instructions=None):
    # This function is mock, as genai.configure() relies on external setup in this environment
    print(f"--- Mock Gemini Config: Using model {model_name} with system instructions. ---")
    class MockModel:
        def generate_content(self, prompt):
            # This is where the actual API call would happen
            return f"Mocked Gemini Response for prompt length {len(prompt)}"
    return MockModel()

def callGemini(model, prompt):
    return model.generate_content(prompt)

def callLlama(api_link, prompt, api_key=None, model=None, stream=False):
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
    filtered_context: str = ""
) -> str:
    """
    Main inference function that routes to appropriate API based on model.
    """
    config = CONFIG.get(model_name, CONFIG['llama']) if model_name != "gemini" else CONFIG['gemini']
    
    # Get actual values from environment (mocked)
    api_key = get_config_value(config, 'api_key')
    api_link = get_config_value(config, 'api_link')
    
    # RAG is used if the model supports it AND there is contextual data provided.
    use_rag = config.get('supports_rag', False) and (factual_data or filtered_context)
    
    # Define the path to the prompts.json file
    system_source = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'prompts.json')
    
    
    if model_name == "gemini":
        system_instructions = load_system_instructions(system_source)

        if use_rag:
            assembled_prompt = assemble_rag_prompt_gemini(
                system_source, factual_data, filtered_context, prompt
            )
            model = configureGemini(api_key, config['model'], system_instructions=system_instructions)
            print(f'The assembled prompt: {assembled_prompt}')
            return callGemini(model, assembled_prompt)
        else:
            model = configureGemini(api_key, config['model'], system_instructions=system_instructions)
            return callGemini(model, prompt)
    else:
        # Local model (Ollama)
        if use_rag:
            # prompt_to_send will now be the fully assembled prompt including all RAG context
            prompt_to_send = assemble_rag_prompt(system_source, factual_data, filtered_context, prompt)
        else:
            prompt_to_send = prompt
        
        return callLlama(api_link, prompt_to_send, api_key, model=config.get('model'))


# --- EXAMPLE USAGE ---

if __name__ == "__main__":
    model_name = "gemini"  # Set to use the local LLama/Zephyr model
    
    # --- Define Placeholder Variables for FULL RAG Test (BGP Configuration) ---

    # 1. Configuration Goal (User Query)
    user_query = "Configure BGP peering between R1 and R2 using AS 65000. Use direct interface IPs for peering. Set router IDs manually."

    # 2. Target Devices and Details (Factual/PostgreSQL Data)
    target_devices_details = """
* R1: Device Type: Router. Interfaces: {G0/0/0: 192.168.12.1/30, L0: 1.1.1.1/32}. BGP AS: 65000.
* R2: Device Type: Router. Interfaces: {G0/0/0: 192.168.12.2/30, L0: 2.2.2.2/32}. BGP AS: 65000.
"""

    # 3. Protocol Facts (Content/FAISS Data) + 4. VPP Error Report (combined for demonstration)
    protocol_facts_and_errors = """
**Protocol Facts:** The BGP command to start the process is 'router bgp <AS_number>'. The neighbor command is 'neighbor <remote_IP> remote-as <remote_AS>'. The router-id command is 'bgp router-id <id>' or 'router-id <id>' depending on the model/IOS version. Layer 2 commands (like 'switchport') can ONLY be applied to Switch devices.
**VPP Error Report (OPTIONAL):** No errors reported in previous attempt. This is the first run.
"""
    
    print("--- Test 1: Simple query without RAG (for baseline) ---")
    try:
        response = generate(
            model_name=model_name,
            prompt="What is BGP?",
            factual_data="",
            filtered_context=""
        )
        print(response)
    except Exception as e:
        print(f"Error: {e}")

    print("\n--- Test 2: Query with FULL RAG context (BGP Test) ---")
    try:
        response = generate(
            model_name=model_name,
            prompt=user_query,
            factual_data=target_devices_details,
            filtered_context=protocol_facts_and_errors
        )
        print(response)
    except Exception as e:
        print(f"Error: {e}")
