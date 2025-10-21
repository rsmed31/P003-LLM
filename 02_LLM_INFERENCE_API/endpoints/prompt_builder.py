import os
import json


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


def build_context_block(factual_data_str: str, filtered_context_str: str) -> str | None:
    """
    Build the combined context block once using functional/trust labels.
    Returns a string or None if no context is provided.
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

RIGID_PROMPT_TEMPLATE = """
**CRITICAL RULE:** Your entire response MUST be a single, complete JSON array. This array MUST be the ONLY text in your response. Do not include any natural language introduction, explanations, notes, or apologies. Do not include the JSON markdown delimiters (`json ... `).

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
    factual_data_str: str,
    filtered_context_str: str,
    user_query_str: str
) -> str:
    """
    Assembles the final RAG prompt following the Blueprint structure.
    For local models with chat templates (Zephyr, Llama, etc.)
    """
    
    # Load system instructions (JSON or text)
    system_instructions = load_system_instructions(system_file_path)
    
    # Build contextual knowledge base once
    combined_context = build_context_block(factual_data_str, filtered_context_str) or "No contextual data available."
    
    # Fill the static template with dynamic data
    filled_template = RIGID_PROMPT_TEMPLATE.format(
        user_query_str=user_query_str,
        combined_context=combined_context
    )

    # Assemble final prompt following chat template structure (e.g., Llama/Zephyr)
    # The system instructions are placed inside the <|system|> tag.
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
    factual_data_str: str,
    filtered_context_str: str,
    user_query_str: str
) -> str:
    """
    Assembles RAG prompt specifically for Gemini (no chat template tags).
    """
    
    # Load system instructions (JSON or text)
    system_instructions = load_system_instructions(system_file_path)
    
    # Build contextual knowledge base once
    combined_context = build_context_block(factual_data_str, filtered_context_str) or "No contextual data available."
    
    # Fill the static template with dynamic data
    filled_template = RIGID_PROMPT_TEMPLATE.format(
        user_query_str=user_query_str,
        combined_context=combined_context
    )

    # Assemble prompt for Gemini (clean format, using the system instruction loaded from file)
    final_prompt = f"""{system_instructions}

---

{filled_template}

---

Please provide your response below:"""
    
    return final_prompt
