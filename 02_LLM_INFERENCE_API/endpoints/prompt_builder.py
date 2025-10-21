import os
import json
from rag_logic.code_aware_filter import filter_and_refine_context


def load_system_instructions(system_source_path: str) -> str:
    """
    Load system instructions from a JSON file (expects key 'SYSTEM_PROMPT')
    or from a plain text file if not JSON.
    """
    if not os.path.exists(system_source_path):
        raise FileNotFoundError(f"System instruction source not found: {system_source_path}")
    _, ext = os.path.splitext(system_source_path)
    if ext.lower() == ".json":
        with open(system_source_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # We are now loading the SYSTEM_PROMPT from the provided JSON set
            return (data.get("SYSTEM_PROMPT") or "").strip()
    # Fallback to plain text file
    with open(system_source_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_context_block(
    filtered_context_str: str,
    use_code_filter: bool = True
) -> str | None:
    """
    Build the procedural context block using code-aware filtering.
    
    NOTE: Definitive knowledge is no longer handled here.
    The retrieval orchestrator (LLM-007) decides whether to use definitive facts
    or proceed to full RAG BEFORE this function is called.
    
    Args:
        filtered_context_str: FAISS procedural chunks (to be code-filtered)
        use_code_filter: Apply code-aware filtering to procedural context
    
    Returns:
        Formatted procedural context string or None if no context provided
    """
    
    if not filtered_context_str or not filtered_context_str.strip():
        return None
    
    # Apply code-aware filtering if enabled
    if use_code_filter:
        # Split by chunks if multiple are concatenated
        raw_chunks = filtered_context_str.split('\n---\n')
        refined = filter_and_refine_context(raw_chunks)
        if refined:
            return "## PROCEDURAL CONTEXT (Documentation/Examples):\n" + refined
        return None
    else:
        return "## PROCEDURAL CONTEXT (Documentation/Examples):\n" + filtered_context_str.strip()
    # Build the combined context block once using functional/trust labels.
    # Returns a string or None if no context is provided.

"""
    blocks = []
    if factual_data_str and factual_data_str.strip():
        # This now matches the 'Target Devices and Details' from the RIGID_PROMPT_TEMPLATE
        blocks.append("## TARGET DEVICES AND DETAILS (Factual/PostgreSQL Data):\n" + factual_data_str.strip())
    if filtered_context_str and filtered_context_str.strip():
        # This now matches the 'Protocol Facts' and 'VPP Error Report' from the RIGID_PROMPT_TEMPLATE
        blocks.append("## PROTOCOL FACTS AND ERRORS (Content/FAISS Data):\n" + filtered_context_str.strip())
    return "\n\n".join(blocks) if blocks else None


# --- START OF CUSTOM PROMPT TEMPLATE ---
# This static text defines the rigid rules and schema, incorporating RAG data dynamically.

RIGID_PROMPT_TEMPLATE = 
"""
"""
# **CRITICAL RULE:** Your entire response MUST be a single, complete JSON array. This array MUST be the ONLY text in your response. Do not include any natural language introduction, explanations, notes, or apologies. Do not include the JSON markdown delimiters (`json ... `).

# TASK: Generate the configuration commands necessary to fulfill the requested Goal for all specified devices.

# SITUATION & CONSTRAINTS (Data Injected by Python):

1. **Configuration Goal (User Query):**
{user_query_str}

2. **Contextual Knowledge Base (RAG Data):**
{combined_context}

# REQUIRED JSON SCHEMA:
The response MUST be a JSON array containing an object for EACH device that requires configuration changes, strictly conforming to this structure. Note that the command list should be executed sequentially, starting from the device's global configuration mode (e.g., `configure terminal`).

[
  {{
    "device_name": "string (e.g., R1 or SW1)",
    "protocol": "string (e.g., OSPF, VLAN, BGP, INTERFACE-IP)",
    "configuration_mode_commands": [
      "string (The first command, typically 'configure terminal' if needed)",
      "string (The second command)",
      // ... continue with all commands in the correct sequence
    ],
    "verification_command": "string (e.g., show ip ospf neighbor or show vlan brief)"
  }}
  // ... continue for all other devices
]
"""
# --- END OF CUSTOM PROMPT TEMPLATE ---


def assemble_rag_prompt(
    system_file_path: str,
    filtered_context_str: str,
    user_query_str: str,
    use_code_filter: bool = True
) -> str:
    """
    Assembles the final RAG prompt following the Blueprint structure.
    For local models with chat templates (Zephyr, Llama, etc.)
    Assembles the final RAG prompt for full RAG path.
    
    NOTE: Only called when retrieval orchestrator routes to "full_rag".
    Definitive answers bypass this function entirely.
    
    Args:
        system_file_path: Path to system instructions
        filtered_context_str: FAISS procedural chunks
        user_query_str: User's query
        use_code_filter: Apply code-aware filtering
    
    Returns:
        Formatted prompt for local models with chat templates
    """
    
    # Load system instructions
    system_instructions = load_system_instructions(system_file_path)
    
    # Build procedural context only
    combined_context = build_context_block(filtered_context_str, use_code_filter) or "No contextual data available."
    
    # Fill the static template with dynamic data
    filled_template = RIGID_PROMPT_TEMPLATE.format(
        user_query_str=user_query_str,
        combined_context=combined_context
    )

    # Assemble final prompt following chat template structure (e.g., Llama/Zephyr)
    # The system instructions are placed inside the <|system|> tag.
    # Assemble final prompt
    final_prompt = f"""<|system|>
{system_instructions}
</s>
<|user|>
{filled_template}
</s>
<|assistant|>
"""
    
    return final_prompt


def assemble_rag_prompt_gemini(
    system_file_path: str,
    filtered_context_str: str,
    user_query_str: str,
    use_code_filter: bool = True
) -> str:
    """
    Assembles RAG prompt specifically for Gemini (no chat template tags).
    Assembles RAG prompt for Gemini (full RAG path only).
    
    NOTE: Only called when retrieval orchestrator routes to "full_rag".
    
    Args:
        system_file_path: Path to system instructions
        filtered_context_str: FAISS procedural chunks
        user_query_str: User's query
        use_code_filter: Apply code-aware filtering
    
    Returns:
        Formatted prompt for Gemini without chat template tags
    """
    
    # Load system instructions
    system_instructions = load_system_instructions(system_file_path)
    
    # Build procedural context only
    combined_context = build_context_block(filtered_context_str, use_code_filter) or "No contextual data available."
    
    # Fill the static template with dynamic data
    filled_template = RIGID_PROMPT_TEMPLATE.format(
        user_query_str=user_query_str,
        combined_context=combined_context
    )

    # Assemble prompt for Gemini (clean format, using the system instruction loaded from file)
    # Assemble prompt for Gemini
    final_prompt = f"""{system_instructions}

---

{filled_template}

---

Please provide your response below:"""
    
    return final_prompt
