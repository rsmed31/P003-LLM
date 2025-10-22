import os
import json
# NOTE: The rag_logic import is omitted here as the file is not provided.

# --- START MODULARITY SETUP ---
# Dictionary to cache all prompt parts loaded from the JSON file
PROMPT_PARTS = {}

def load_prompt_parts(system_source_path: str) -> dict:
    """
    Load all required prompt parts from a JSON file.
    Caches the results in PROMPT_PARTS for efficiency and future calls.
    """
    global PROMPT_PARTS
    if PROMPT_PARTS:
        return PROMPT_PARTS

    if not os.path.exists(system_source_path):
        raise FileNotFoundError(f"System instruction source not found: {system_source_path}")
        
    _, ext = os.path.splitext(system_source_path)
    if ext.lower() != ".json":
        raise ValueError("Prompt system source must be a JSON file.")

    with open(system_source_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Keys defined in the final modular prompts.json structure
    required_keys = [
        "SYSTEM_PROMPT", "CRITICAL_RULE_INSTRUCTION", "TASK_INSTRUCTION",
        "SITUATION_HEADER", "CONFIG_GOAL_HEADER", "CONTEXT_BASE_HEADER",
        "SCHEMA_HEADER", "JSON_SCHEMA_BODY"
    ]
    
    # Load and strip all required parts
    for key in required_keys:
        if key not in data:
             raise KeyError(f"Missing required prompt part in JSON file: {key}")
        PROMPT_PARTS[key] = data.get(key, "").strip()

    return PROMPT_PARTS


def load_system_instructions(system_source_path: str) -> str:
    """
    Loads only the SYSTEM_PROMPT for API calls, using the modular loader internally.
    """
    parts = load_prompt_parts(system_source_path)
    return parts['SYSTEM_PROMPT']

# --- END MODULARITY SETUP ---

def build_context_block(filtered_context_str: str) -> str | None:
    """
    Build the context block from retrieval service chunks only.
    No PostgreSQL integration - all context comes from external retrieval endpoint.
    """
    if not filtered_context_str or not filtered_context_str.strip():
        return None
    
    # Single header: Protocol Facts and Errors (from retrieval service)
    return "## PROTOCOL FACTS AND ERRORS (Retrieved Context):\n" + filtered_context_str.strip()


# --- ASSEMBLY FUNCTIONS ---

def assemble_rag_prompt(
    system_file_path: str,
    filtered_context_str: str,
    user_query_str: str
) -> str:
    """
    Assembles the final RAG prompt for local models (e.g., Llama/Zephyr).
    """
    
    parts = load_prompt_parts(system_file_path)
    combined_context = build_context_block(filtered_context_str) or "No contextual data available."
    
    # Assemble the modular template using retrieved parts and the core user data
    modular_template = f"""
{parts['CRITICAL_RULE_INSTRUCTION']}

{parts['TASK_INSTRUCTION']}

{parts['SITUATION_HEADER']}
{parts['CONFIG_GOAL_HEADER']}
{user_query_str}

{parts['CONTEXT_BASE_HEADER']}
{combined_context}

{parts['SCHEMA_HEADER']}
{parts['JSON_SCHEMA_BODY']}
"""

    # Assemble final prompt following chat template structure (e.g., Llama/Zephyr)
    final_prompt = f"""<|system|>
{parts['SYSTEM_PROMPT']}
</s>
<|user|>
{modular_template.strip()}
</s>
<|assistant|>
"""
    
    return final_prompt


def assemble_rag_prompt_gemini(
    system_file_path: str,
    filtered_context_str: str,
    user_query_str: str
) -> str:
    """
    Assemble the full Gemini-style RAG prompt using modular JSON components.

    Parameters
    ----------
    system_file_path : str
        Path to the system JSON file (e.g., prompts.json).
    filtered_context_str : str
        Retrieved context (e.g., from FAISS or embeddings search).
    user_query_str : str
        User goal or instruction (e.g., network configuration task).

    Returns
    -------
    str
        The full text prompt ready for Gemini API input.
    """
    # --- Load modular prompt parts ---
    parts = load_prompt_parts(system_file_path)

    # --- Combine contextual data blocks ---
    combined_context = build_context_block(
         filtered_context_str
    ) or "No contextual data available."

    # --- Construct modular Gemini prompt ---
    modular_template = f"""
{parts['SYSTEM_PROMPT']}

{parts['CRITICAL_RULE_INSTRUCTION']}

{parts['TASK_INSTRUCTION']}

{parts['SITUATION_HEADER']}
{parts['CONFIG_GOAL_HEADER']}
{user_query_str.strip()}

{parts['CONTEXT_BASE_HEADER']}
{combined_context.strip()}

{parts['SCHEMA_HEADER']}
{parts['JSON_SCHEMA_BODY']}
"""

    return modular_template.strip()
