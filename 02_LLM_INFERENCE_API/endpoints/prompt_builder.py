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
        blocks.append("## DEFINITIVE KNOWLEDGE (Verified Facts):\n" + factual_data_str.strip())
    if filtered_context_str and filtered_context_str.strip():
        blocks.append("## PROCEDURAL CONTEXT (Documentation/Examples):\n" + filtered_context_str.strip())
    return "\n\n".join(blocks) if blocks else None


def assemble_rag_prompt(
    system_file_path: str,
    factual_data_str: str,
    filtered_context_str: str,
    user_query_str: str
) -> str:
    """
    Assembles the final RAG prompt following the Blueprint structure.
    For local models with chat templates (Zephyr, Llama, etc.)
    
    Args:
        system_file_path: Path to prompts_run.txt containing system instructions
        factual_data_str: PostgreSQL factual retrieval results
        filtered_context_str: FAISS code-aware filtered chunks
        user_query_str: User's raw natural language query
    
    Returns:
        Formatted prompt string ready for LLM inference
    """
    
    # Load system instructions (JSON or text)
    system_instructions = load_system_instructions(system_file_path)
    
    # Build contextual knowledge base once
    combined_context = build_context_block(factual_data_str, filtered_context_str) or "No contextual data available."
    
    # Assemble final prompt following chat template structure
    final_prompt = f"""<|system|>
{system_instructions}
</s>
<|user|>
# CONTEXTUAL KNOWLEDGE BASE:
{combined_context}

# USER COMMAND:
{user_query_str}
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
    
    Args:
        system_file_path: Path to prompts_run.txt containing system instructions
        factual_data_str: PostgreSQL factual retrieval results
        filtered_context_str: FAISS code-aware filtered chunks
        user_query_str: User's raw natural language query
    
    Returns:
        Formatted prompt string for Gemini
    """
    
    # Load system instructions (JSON or text)
    system_instructions = load_system_instructions(system_file_path)
    
    # Build contextual knowledge base once
    combined_context = build_context_block(factual_data_str, filtered_context_str) or "No contextual data available."
    
    # Assemble prompt for Gemini (clean format without chat tags)
    final_prompt = f"""{system_instructions}

---

# CONTEXTUAL KNOWLEDGE BASE:
{combined_context}

---

# USER COMMAND:
{user_query_str}

---

Please provide your response below:"""
    
    return final_prompt
