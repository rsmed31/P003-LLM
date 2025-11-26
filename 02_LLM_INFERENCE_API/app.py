import json
import requests
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import os

from endpoints.inference import (
    generate, 
    CONFIG, 
    RETRIEVAL_SERVICE_URL, 
    get_chunk_count_for_query,
    load_system_instructions
)
from endpoints.prompt_builder import assemble_rag_prompt, assemble_rag_prompt_gemini

try:
    from retrieval import RetrievalOrchestrator
    _TEST_ORCHESTRATOR = RetrievalOrchestrator()
except Exception:
    _TEST_ORCHESTRATOR = None

app = FastAPI(
    title="Team 2 - getAnswer API",
    version="1.0.0",
    description="Endpoint used by Team 3 to get the generated answer or commands from Team 2."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

class GetAnswerResponse(BaseModel):
    model: str
    response: list
    rag_enabled: bool  # Changed from loopback

class TestResponse(BaseModel):
    query: str
    model: str
    rag_enabled: bool
    retrieval_service: str
    chunks_requested: int
    chunks_retrieved: int
    chunks_preview: list
    filtered_context: str
    prompt_preview: str
    full_prompt_length: int
    would_send_to: str
    retrieval_error: Optional[str] = None

@app.get("/health")
async def health():
    return {"ok": True}

@app.get("/v1/getAnswer", response_model=GetAnswerResponse, tags=["Inference"])
async def get_answer(
    q: str = Query(..., min_length=3, description="User query to process"),
    model: str = Query("llama", description="Target model to use", pattern="^(gemini|llama)$"),
    rag: str = Query("on", description="RAG mode: 'on' for retrieval-augmented, 'off' for direct inference", pattern="^(on|off)$")
):
    """
    Team 3 calls this endpoint to get a generated answer or configuration commands.
    """
    try:
        # Convert rag on/off to loopback boolean (inverse)
        loopback = (rag == "off")
        
        raw = generate(query=q, model_name=model, loopback=loopback)
        result = json.loads(raw)
        
        model_name = result.get("model", "unknown_model")
        response_data = result.get("response", [])
        
        if not response_data:
            raise HTTPException(status_code=502, detail="Empty response from model")
        
        # Validate schema compliance for human-readable errors
        for i, device in enumerate(response_data):
            if not isinstance(device, dict):
                raise HTTPException(
                    status_code=502, 
                    detail=f"Device {i}: Expected object, got {type(device).__name__}. Model output malformed."
                )
            
            missing = []
            if "device_name" not in device:
                missing.append("device_name")
            if "configuration_mode_commands" not in device:
                missing.append("configuration_mode_commands")
            if "protocol" not in device:
                missing.append("protocol")
            if "intent" not in device:
                missing.append("intent")
            
            if missing:
                raise HTTPException(
                    status_code=502,
                    detail=f"Device '{device.get('device_name', f'index_{i}')}': Missing required fields: {', '.join(missing)}. Check model prompt schema enforcement."
                )
            
            if not device["configuration_mode_commands"]:
                raise HTTPException(
                    status_code=502,
                    detail=f"Device '{device['device_name']}': Empty configuration_mode_commands array. Model must generate actual commands."
                )
        
        return {"model": model_name, "response": response_data, "rag_enabled": not loopback}
    
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Model returned invalid JSON: {str(e)}. Raw output likely contained text outside JSON array."
        )
    except HTTPException:
        raise
    except ValueError as ve:
        # Catch validation errors from inference.py and make them readable
        error_msg = str(ve)
        if "missing required keys" in error_msg.lower():
            raise HTTPException(
                status_code=502,
                detail=f"Schema validation failed in inference layer: {error_msg}. Model prompt must enforce all required fields: device_name, configuration_mode_commands, protocol, intent."
            )
        raise HTTPException(status_code=502, detail=f"Validation error: {error_msg}")
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Internal error: {str(e)}. Check server logs for details."
        )

@app.get("/test", response_model=TestResponse, tags=["Inference", "Debug"])
async def test_prompt_building(
    q: str = Query(..., min_length=3, description="Query to test (no model call)"),
    model: str = Query("llama", description="Model type", pattern="^(gemini|llama)$")
):
    """
    DEBUG: Test RAG pipeline without calling the model.
    """
    try:
        config = CONFIG.get(model, CONFIG["llama"])
        supports_rag = config.get("supports_rag", False)
        chunk_count = get_chunk_count_for_query(q, config)
        
        filtered_context = ""
        chunks_retrieved = 0
        chunks_preview = []
        retrieval_error = None
        
        if supports_rag and RETRIEVAL_SERVICE_URL:
            try:
                response = requests.get(
                    f"{RETRIEVAL_SERVICE_URL}/chunks/query",
                    params={"query": q, "limit": chunk_count},
                    timeout=1000
                )
                if response.status_code == 200:
                    data = response.json()
                    if data.get("found", False):
                        results = data.get("results", [])
                        chunks = [item["text"] for item in results if "text" in item][:chunk_count]
                        
                        if model == "llama":
                            filtered_context = "\n---\n".join(chunks) if chunks else ""
                            chunks_retrieved = len(chunks)
                            chunks_preview = [
                                {"index": i+1, "preview": chunk[:200] + "...", "type": "raw"}
                                for i, chunk in enumerate(chunks[:3])
                            ]
                        else:
                            if chunks and _TEST_ORCHESTRATOR:
                                _TEST_ORCHESTRATOR.add_chunks(chunks)
                                correlation_result = _TEST_ORCHESTRATOR.retrieve_with_correlation(q, len(chunks))
                                
                                code_chunks = [c["chunk"] for c in correlation_result["chunks"] if c["type"] == "code"]
                                theory_chunks = [c["chunk"] for c in correlation_result["chunks"] if c["type"] == "theory"]
                                
                                context_parts = []
                                if code_chunks:
                                    context_parts.append("## CODE-AWARE CONTEXT:\n" + "\n---\n".join(code_chunks))
                                if theory_chunks:
                                    context_parts.append("## THEORETICAL CONTEXT:\n" + "\n---\n".join(theory_chunks))
                                
                                filtered_context = "\n\n".join(context_parts) if context_parts else ""
                                chunks_retrieved = len(correlation_result["chunks"])
                                chunks_preview = [
                                    {
                                        "index": i+1,
                                        "preview": c["chunk"][:200] + "...",
                                        "type": c["type"]
                                    }
                                    for i, c in enumerate(correlation_result["chunks"][:3])
                                ]
                    else:
                        retrieval_error = "No matching chunks found"
                else:
                    retrieval_error = f"Retrieval service returned status {response.status_code}"
            except Exception as e:
                retrieval_error = f"Failed to retrieve context: {str(e)}"
        
        prompts_path = os.path.join(os.path.dirname(__file__), "prompts", "prompts.json")
        
        if model == "gemini":
            full_prompt = assemble_rag_prompt_gemini(prompts_path, filtered_context, q)
            api_endpoint = "https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent"
        else:
            full_prompt = assemble_rag_prompt(prompts_path, filtered_context, q)
            api_endpoint = config.get('api_link', 'http://localhost:11434/api/generate')
        
        return {
            "query": q,
            "model": model,
            "rag_enabled": supports_rag,
            "retrieval_service": RETRIEVAL_SERVICE_URL or "N/A",
            "chunks_requested": chunk_count if supports_rag else 0,
            "chunks_retrieved": chunks_retrieved,
            "chunks_preview": chunks_preview,
            "filtered_context": filtered_context,
            "prompt_preview": full_prompt[:500] + "..." if len(full_prompt) > 500 else full_prompt,
            "full_prompt_length": len(full_prompt),
            "would_send_to": api_endpoint,
            "retrieval_error": retrieval_error
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")
