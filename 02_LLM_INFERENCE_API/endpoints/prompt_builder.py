import os
import json
from code_aware_filter import filter_and_refine_context


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


def assemble_rag_prompt(
    system_file_path: str,
    filtered_context_str: str,
    user_query_str: str,
    use_code_filter: bool = True
) -> str:
    """
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
    
    # Assemble final prompt
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
    filtered_context_str: str,
    user_query_str: str,
    use_code_filter: bool = True
) -> str:
    """
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
    
    # Assemble prompt for Gemini
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
