import os


def assemble_rag_prompt(
    system_file_path: str,
    factual_data_str: str,
    filtered_context_str: str,
    user_query_str: str
) -> str:
    """
    Assembles the final RAG prompt following the Blueprint structure.
    
    Args:
        system_file_path: Path to prompts_run.txt containing system instructions
        factual_data_str: PostgreSQL factual data retrieval results
        filtered_context_str: FAISS code-aware filtered chunks
        user_query_str: User's raw natural language query
    
    Returns:
        Formatted prompt string ready for LLM inference
    """
    
    # Load system instructions from file
    if not os.path.exists(system_file_path):
        raise FileNotFoundError(f"System instruction file not found: {system_file_path}")
    
    with open(system_file_path, 'r', encoding='utf-8') as f:
        system_instructions = f.read().strip()
    
    # Construct contextual knowledge base
    contextual_knowledge = []
    
    if factual_data_str and factual_data_str.strip():
        contextual_knowledge.append("## PostgreSQL Facts:\n" + factual_data_str.strip())
    
    if filtered_context_str and filtered_context_str.strip():
        contextual_knowledge.append("## FAISS Retrieved Context:\n" + filtered_context_str.strip())
    
    # Combine context sections
    combined_context = "\n\n".join(contextual_knowledge) if contextual_knowledge else "No contextual data available."
    
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
