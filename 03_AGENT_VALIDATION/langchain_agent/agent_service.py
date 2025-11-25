import os, json, logging
import warnings
# Suppress Pydantic V1 compatibility warning
warnings.filterwarnings("ignore", message=".*Pydantic V1 functionality.*", category=UserWarning)

from typing import Dict, Any, Optional, List, Tuple, Set
import requests
import datetime, pathlib
from langchain_core.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# LangChain (no agent reasoning needed; we use LCEL + an LLM only for the final explanation)
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

# ========= CONFIG LOAD FROM JSON =========
HERE = os.path.dirname(os.path.abspath(__file__))

# Load environment variables from .env file
load_dotenv(os.path.join(HERE, ".env"))

CONFIG_PATH = os.path.join(HERE, "config.json")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)
else:
    cfg = {}

def get_cfg(key, default=None):
    return cfg.get(key) or os.getenv(key, default)

T1_BASE_URL = get_cfg("T1_BASE_URL", "http://t1:8000").rstrip("/")
T2_BASE_URL = get_cfg("T2_BASE_URL", "http://t2:8002").rstrip("/")
T3_BASE_URL = get_cfg("T3_BASE_URL", "http://t3:8003").rstrip("/")

# Endpoints
T1_QA_LOOKUP = get_cfg("T1_ENDPOINT_QA_LOOKUP", "/qa/query")
T1_WRITE = get_cfg("T1_ENDPOINT_WRITE", "/qa")
T2_GENERATE = get_cfg("T2_ENDPOINT_GENERATE", "/v1/getAnswer")
T3_VALIDATE = get_cfg("T3_ENDPOINT_VALIDATE", "/evaluate")

TIMEOUT = float(get_cfg("HTTP_TIMEOUT", 90))
LOG_LEVEL = get_cfg("LOG_LEVEL", "INFO").upper()

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("agent-v2")

LOG_DIR = get_cfg("LOG_DIR", os.path.join(HERE, "logs"))
LOG_EVENTS = get_cfg("LOG_EVENTS", "1") in ("1", "true", "True", "yes")
# Load intent types from config (full candidate set)
INTENT_TYPES_CONFIG = set(get_cfg("INTENT_TYPES_SUPPORTED", [
    "adjacency", "connectivity", "reachability", "interface", "policy", "redundancy"
]))

pathlib.Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
_pipeline_log_path = os.path.join(LOG_DIR, "pipeline.log")

class PipelineCallback(BaseCallbackHandler):
    """LangChain callback that appends JSONL events for tracing."""
    def _write(self, event: str, data: Dict[str, Any]):
        try:
            payload = {
                "ts": datetime.datetime.utcnow().isoformat() + "Z",
                "event": event,
                **data
            }
            with open(_pipeline_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as e:
            log.warning(f"Callback write failed: {e}")

    def on_chain_start(self, serialized, inputs, **kwargs):
        if LOG_EVENTS:
            self._write("chain_start", {"name": serialized.get("name"), "inputs": inputs})

    def on_chain_end(self, outputs, **kwargs):
        if LOG_EVENTS:
            self._write("chain_end", {"outputs": outputs})

    def on_llm_start(self, serialized, prompts, **kwargs):
        if LOG_EVENTS:
            self._write("llm_start", {"prompts": prompts})

    def on_llm_end(self, response, **kwargs):
        if LOG_EVENTS:
            self._write("llm_end", {"response": getattr(response, "generations", [])})

    def on_llm_error(self, error, **kwargs):
        if LOG_EVENTS:
            self._write("llm_error", {"error": str(error)})

    def on_chain_error(self, error, **kwargs):
        if LOG_EVENTS:
            self._write("chain_error", {"error": str(error)})

_callbacks = [PipelineCallback()]

# ========= HTTP helper =========
def _post(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(url, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def _get(url: str) -> Dict[str, Any]:
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def _probe_t1() -> bool:
    """Check if Team 1 service is reachable via health check."""
    if not T1_BASE_URL:
        return False
    try:
        r = requests.get(f"{T1_BASE_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

_TERMINAL_NOOPS = {"configure terminal", "end", "exit"}

def _clean_commands(cmds: List[str]) -> List[str]:
    """Remove terminal/no-op lines and keep order."""
    out = []
    for c in cmds or []:
        c = (c or "").strip()
        if not c or c.lower() in _TERMINAL_NOOPS:
            continue
        out.append(c)
    return out

def _intents_to_reach(intents: List[dict]) -> List[dict]:
    """
    Convert Team2 'adjacency' intents into simple reach checks for T3.
    Example input intent:
      {"type":"adjacency","endpoints":[{"role":"router","id":"R1"},{"role":"router","id":"R2"}]}
    Output:
      [{"src":"R1","dst":"R2"}]
    """
    reach: List[dict] = []
    seen: Set[Tuple[str, str]] = set()
    for it in intents or []:
        if (it or {}).get("type") != "adjacency":
            continue
        eps = (it or {}).get("endpoints") or []
        if len(eps) != 2:
            continue
        a = (eps[0] or {}).get("id")
        b = (eps[1] or {}).get("id")
        if not a or not b:
            continue
        key = tuple(sorted([a, b]))
        if key in seen:
            continue
        seen.add(key)
        # Arbitrarily choose direction a->b; T3 can test bidirectional if needed
        reach.append({"src": a, "dst": b})
    return reach

def _categorize_intents(intents: List[dict], active_types: Set[str]) -> Dict[str, List[dict]]:
    """Group intents by active type set; ignore unknown types."""
    buckets: Dict[str, List[dict]] = {k: [] for k in active_types}
    for it in intents or []:
        t = (it or {}).get("type", "").lower()
        if t in buckets:
            buckets[t].append(it)
    return buckets

def _intents_to_reach_all(intent_groups: Dict[str, List[dict]]) -> List[dict]:
    """
    Build reach list from connectivity/adjacency/reachability intents.
    Each intent must have >=2 endpoints with 'id'.
    """
    reach: List[dict] = []
    seen: Set[Tuple[str, str]] = set()
    for key in ("adjacency", "connectivity", "reachability"):
        for it in intent_groups.get(key, []):
            eps = (it or {}).get("endpoints") or []
            # Use first two distinct IDs
            ids = [e.get("id") for e in eps if isinstance(e, dict) and e.get("id")]
            if len(ids) < 2:
                continue
            a, b = ids[0], ids[1]
            pair = tuple(sorted([a, b]))
            if pair in seen:
                continue
            seen.add(pair)
            reach.append({"src": a, "dst": b})
    return reach

def _parse_t2_response(raw_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Normalize Team 2's response into a list of device configs.
    Handles multiple formats:
    1. {"response": [{"device_name": "R1", ...}]}  <- expected
    2. {"response": "bash\nconfigure terminal..."}  <- string fallback
    3. [{"device_name": "R1", ...}]                <- direct list
    """
    log.info(f"[T2 Parse] Raw response keys: {list(raw_data.keys())}")
    
    # Case 1: {"response": [...]}
    if "response" in raw_data:
        resp = raw_data["response"]
        
        # Case 1a: List of devices (expected)
        if isinstance(resp, list):
            log.info(f"[T2 Parse] Found list with {len(resp)} devices")
            return resp
        
        # Case 1b: String (bash script or plain text)
        if isinstance(resp, str):
            log.warning("[T2 Parse] Response is string format, attempting to parse as single device")
            # Try to extract device name from string
            device_name = "UnknownDevice"
            lines = [line.strip() for line in resp.split('\n') if line.strip()]
            
            # Look for hostname command
            for line in lines:
                if line.startswith("hostname "):
                    device_name = line.split("hostname ", 1)[1].strip()
                    break
            
            return [{
                "device_name": device_name,
                "configuration_mode_commands": lines,
                "protocol": "UNKNOWN",
                "intent": []
            }]
    
    # Case 2: Direct list at root level
    if isinstance(raw_data, list):
        log.info(f"[T2 Parse] Root-level list with {len(raw_data)} devices")
        return raw_data
    
    # Case 3: Fallback - no recognizable format
    log.error(f"[T2 Parse] Unrecognized format. Keys: {list(raw_data.keys())}")
    raise HTTPException(
        status_code=502, 
        detail=f"T2 response format not recognized. Keys: {list(raw_data.keys())}, Type: {type(raw_data)}"
    )

FALLBACK_LOOPBACK_PREFIX = get_cfg("FALLBACK_LOOPBACK_PREFIX", "10.255")
FALLBACK_MAX_PAIRWISE_REACH = int(get_cfg("FALLBACK_MAX_PAIRWISE_REACH", 20))

# ---- Helper bootstrap (ensure rename helpers exist before use) ----
def _requires_router_names(query: str) -> bool:
    q = (query or "").lower()
    return "router" in q or "routers" in q

def _rename_if_switch(query: str, original: str, index: int) -> str:
    if _requires_router_names(query) and original.lower().startswith("switch"):
        return f"R{index+1}"
    return original

# If accidentally overridden or deleted later, rebind safely
if "_rename_if_switch" not in globals():
    def _rename_if_switch(query: str, original: str, index: int) -> str:
        return original

def _build_evaluate_payload_from_t2(data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        resp = _parse_t2_response(data)
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"[T2 Parse] Unexpected error: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to parse T2 response: {e}")
    if not resp:
        raise HTTPException(status_code=502, detail="T2 returned empty device list")

    changes: Dict[str, List[str]] = {}
    collected_intents: List[dict] = []
    original_query = data.get("_original_query", "")

    for idx, dev in enumerate(resp):
        raw_name = dev.get("device_name") or f"Device{idx+1}"
        q_orig = data.get("_original_query", "")
        name = _rename_if_switch(q_orig, raw_name, idx)
        raw_cmds = dev.get("configuration_mode_commands") or []
        cmds = _clean_commands(raw_cmds)
        cmds = _normalize_cmds(cmds)
        cmds = _filter_orphan_area(cmds)
        cmds = _ensure_no_shutdown_on_all_interfaces(cmds)
        changes[name] = cmds
        collected_intents.extend(dev.get("intent") or [])
        log.info(f"[T2 Parse] Device {raw_name}->{name}: cmds={len(cmds)} intents={len(dev.get('intent', []))}")

    # Determine active intent types
    discovered_types = { (it.get("type") or "").lower() for it in collected_intents if isinstance(it, dict) }
    must_keep = {"connectivity", "reachability"}
    active_types = (INTENT_TYPES_CONFIG & discovered_types) | (must_keep & INTENT_TYPES_CONFIG)
    if "adjacency" in active_types and "adjacency" not in discovered_types and "connectivity" in active_types:
        active_types.discard("adjacency")
    log.info(f"[IntentTypes] config={sorted(INTENT_TYPES_CONFIG)} discovered={sorted(discovered_types)} active={sorted(active_types)}")

    intent_groups = _categorize_intents(collected_intents, active_types)

    # Reach from intents (NO fallback generation)
    reach = _intents_to_reach_all(intent_groups)
    
    # Log if no reach pairs found - this is now intentional, not an error
    if not reach:
        log.warning(f"[T2 Parse] No reach intents found. Validation will test only control plane and topology.")

    # Build device configs directly from commands (NO synthetic injection)
    device_configs = {}
    for name, cmds in changes.items():
        device_configs[name] = _render_device_config(name, cmds)

    # Build joined_config
    parts = []
    for name, cfg_text in device_configs.items():
        parts.append(f"! ===== BEGIN {name} =====\n{cfg_text}! ===== END {name} =====")
    joined_config = "\n".join(parts).strip() + "\n"

    evaluate_payload = {
        "changes": changes,
        "snapshot": {
            "configs": device_configs
        },
        "intent": {
            "reach": reach,
            "interface": intent_groups.get("interface", []),
            "policy": intent_groups.get("policy", []),
            "redundancy": intent_groups.get("redundancy", []),
            "connectivity": intent_groups.get("connectivity", [])  # Ensure connectivity is always included
        },
        "meta": {
            "devices": len(changes),
            "raw_intents": len(collected_intents),
            "active_types": list(active_types),
            "reach_pairs": len(reach),
            "synthetic_injection": False  # Changed from fallback_used
        }
    }
    log.info(f"[T2 Parse] Final payload devices={len(changes)} reach={len(reach)} configs_raw={len(device_configs)}")
    return {"evaluate_payload": evaluate_payload, "joined_config": joined_config}

# ========= Team calls =========
def call_t1_qa_lookup(query: str) -> Dict[str, Any]:
    """
    T1: Q&A lookup via GET.
    Expected response:
      { "found": true, "answer": "..." }  OR  { "found": false }
    """
    # Added pre-probe
    if not _probe_t1():
        log.warning("T1 unreachable (health probe failed); skipping QA lookup")
        return {"found": False}
    
    try:
        url = f"{T1_BASE_URL}{T1_QA_LOOKUP}"
        params = {"text": query, "threshold": 0.3}
        resp = requests.get(url, params=params, timeout=TIMEOUT)
        resp.raise_for_status()

        data = resp.json()
        found = bool(data.get("found", False))
        if found and not isinstance(data.get("answer", None), str):
            # sanitize weird payloads
            return {"found": False}
        return {"found": found, "answer": data.get("answer")}

    except Exception as e:
        log.warning(f"T1 QA lookup failed (continuing to T2): {e}")
        return {"found": False}

RECHECK_MAX = int(get_cfg("RECHECK_MAX_ATTEMPTS", 1))
LOOPBACK_ON_FAIL = bool(get_cfg("LOOPBACK_ON_FAIL", True))
_CONFIG_MEMORY: Dict[str, List[str]] = {}

def _fuse_device_cmds(existing: List[str], new: List[str]) -> List[str]:
    seen = set(existing)
    fused = existing[:]
    for cmd in new:
        if cmd not in seen:
            fused.append(cmd)
            seen.add(cmd)
    return fused

def _fuse_changes(changes: Dict[str, List[str]]) -> Dict[str, List[str]]:
    for dev, cmds in changes.items():
        prev = _CONFIG_MEMORY.get(dev, [])
        _CONFIG_MEMORY[dev] = _fuse_device_cmds(prev, cmds)
    return {d: _CONFIG_MEMORY[d] for d in changes.keys()}

def call_t2_generate(query: str, model: str = "gemini", rag_enabled: bool = True) -> Dict[str, Any]:
    """
    Call Team 2 config generation.
    
    Args:
        query: User query
        model: Model to use (gemini/llama)
        rag_enabled: True for RAG, False for direct inference
    """
    # Convert rag_enabled to rag parameter (on/off)
    rag_param = "on" if rag_enabled else "off"
    params = {"q": query, "model": model, "rag": rag_param}
    url = f"{T2_BASE_URL}{T2_GENERATE}"
    log.info(f"[T2] Calling {url} rag={rag_param} model={model}")

    try:
        resp = requests.get(url, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
    except requests.Timeout:
        raise HTTPException(status_code=504, detail=f"T2 request timed out after {TIMEOUT}s")
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"T2 request failed: {e}")

    try:
        raw = resp.json()
        raw["_original_query"] = query
        log.info(f"[T2] Response received, parsing...")
    except Exception as e:
        log.error(f"[T2] Invalid JSON response: {resp.text[:500]}")
        raise HTTPException(status_code=502, detail=f"T2 returned invalid JSON: {e}")

    try:
        norm = _build_evaluate_payload_from_t2(raw)
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"[T2] Normalization failed: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to normalize T2 response: {e}")
    
    if not norm["evaluate_payload"]["changes"]:
        raise HTTPException(status_code=502, detail="T2 produced no device changes")

    return norm

def call_t3_validate(evaluate_payload: Dict[str, Any], retries: int = 2) -> Dict[str, Any]:
    log.info(f"[T3] Sending evaluate payload: {json.dumps(evaluate_payload)[:500]}...")
    
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return _post(f"{T3_BASE_URL}{T3_VALIDATE}", evaluate_payload)
        except requests.HTTPError as e:
            body = e.response.text if e.response is not None else ""
            log.error(f"[T3] Attempt {attempt}/{retries} HTTPError {e} body={body[:500]}")
            
            if LOG_EVENTS:
                PipelineCallback()._write("t3_http_error", {
                    "attempt": attempt,
                    "status": e.response.status_code if e.response else None,
                    "body": body[:500],
                    "payload_keys": list(evaluate_payload.keys()),
                    "changes_devices": list(evaluate_payload.get("changes", {}).keys())
                })
            
            # Parse error from validator if possible
            try:
                err_json = json.loads(body)
                if "stage" in err_json and "error" in err_json:
                    log.error(f"[T3] Validation stage={err_json['stage']}, error={err_json['error']}")
            except Exception:
                pass
            
            last_error = e
            
            # Don't retry on 4xx (client error)
            if e.response and 400 <= e.response.status_code < 500:
                break
                
        except Exception as e:
            log.error(f"[T3] Attempt {attempt}/{retries} Unexpected error: {e}")
            if LOG_EVENTS:
                PipelineCallback()._write("t3_error", {"attempt": attempt, "error": str(e)})
            last_error = e
    
    # All retries exhausted
    raise HTTPException(
        status_code=502,
        detail=f"T3 validation failed after {retries} attempts: {last_error}"
    )

def call_t1_write(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Write validated config back to T1's knowledge base."""
    if not T1_BASE_URL:
        return {"status": "SKIPPED", "reason": "no_t1"}
    
    # Transform payload to match T1 /qa POST endpoint schema
    # Expected: {"question": str, "answer": str}
    write_payload = {
        "question": payload.get("query", ""),
        "answer": payload.get("config", "")
    }
    
    try:
        log.info(f"[T1 Write] Sending payload keys: {list(write_payload.keys())}")
        resp = _post(f"{T1_BASE_URL}{T1_WRITE}", write_payload)
        log.info(f"[T1 Write] Success: {resp}")
        return {"status": "OK", "response": resp}
    except requests.HTTPError as e:
        body = e.response.text if e.response is not None else ""
        log.error(f"[T1 Write] HTTPError {e.response.status_code if e.response else 'unknown'}: {body[:300]}")
        return {"status": "ERROR", "error": f"HTTP {e.response.status_code if e.response else 'unknown'}", "detail": body[:200]}
    except Exception as e:
        log.warning(f"T1 write failed: {e}")
        return {"status": "ERROR", "error": str(e)}

# ========= LLM for final explanation =========
GROQ_MODEL = get_cfg("GROQ_MODEL", "llama-3.3-70b-versatile")

llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0,
    api_key=get_cfg("GROQ_API_KEY")
)

log.info(f"Using Groq model: {GROQ_MODEL}")

verdict_prompt = PromptTemplate.from_template(
    """You are a network assistant. Using the inputs, produce a concise verdict and explanation.

Inputs:
- User query: {query}
- Generated config (from Team 2): <<CONFIG>>
{config}
<</CONFIG>>
- Validation result (from Team 3): {validation_json}

Task:
1) State clearly whether the config is RIGHT or NOT (PASS/FAIL) based on validation.status.
2) Explain briefly WHY (use validation.report).
3) If FAIL, suggest one concrete fix (one line).
4) Keep it under 6 lines total.

Return plain text."""
)
verdict_chain = verdict_prompt | llm | StrOutputParser()

# ========= LangChain pipeline =========
def _structure_final_from_t1(x: Dict[str, Any]) -> Dict[str, Any]:
    """When T1 QA hits, we stop and return the cached/KB answer."""
    return {
        "source": "T1_QA",
        "query": x["query"],
        "answer": x["qa"]["answer"],
        "explanation": "Returned from Q&A knowledge base (cache hit).",
        "config": None,
        "validation": None,
        "write_back": {"status": "SKIPPED", "reason": "T1_qa_hit"}
    }

def _structure_final_from_t1_verified(x: Dict[str, Any], verdict_text: str, validation: Dict[str, Any], config_text: str) -> Dict[str, Any]:
    return {
        "source": "T1_QA_VERIFIED",
        "query": x["query"],
        "answer": x["qa"]["answer"],
        "config": config_text,
        "validation": validation,
        "verdict_text": verdict_text,
        "write_back": {"status": "SKIPPED", "reason": "cached_manual_verify"}
    }

def _parse_verdict_status(verdict_text: str) -> str:
    """
    Extract PASS/FAIL status from LLM verdict text.
    Returns 'PASS' or 'FAIL' based on the FIRST word of the verdict.
    """
    if not isinstance(verdict_text, str):
        return "FAIL"
    
    # Get the first word/line of the verdict text (most reliable indicator)
    first_line = verdict_text.strip().split('\n')[0].strip().upper()
    first_word = first_line.split()[0] if first_line else ""
    
    # Check first word explicitly
    if first_word == "PASS":
        return "PASS"
    elif first_word == "FAIL":
        return "FAIL"
    
    # Fallback: check if first line starts with these keywords
    if first_line.startswith("PASS"):
        return "PASS"
    elif first_line.startswith("FAIL"):
        return "FAIL"
    
    # Default to FAIL if unclear
    return "FAIL"

def _structure_final_from_t2_t3(x: Dict[str, Any]) -> Dict[str, Any]:
    verdict_status = x.get("verdict_status") or _parse_verdict_status(x.get("verdict_text", ""))
    write_back = {"status": "SKIPPED", "reason": "ai_verdict_fail"}
    if verdict_status == "PASS":
        write_back = call_t1_write({"query": x["query"], "config": x["config"], "validation": x["validation"]})
        # persist memory only when passing
        for dev, cmds in (x.get("fused_changes") or {}).items():
            _CONFIG_MEMORY[dev] = cmds
    
    return {
        "source": "T2_T3",
        "query": x["query"],
        "config": x["config"],
        "validation": x["validation"],
        "verdict_text": x.get("verdict_text"),
        "verdict_status": verdict_status,
        "write_back": write_back,
        "meta": x.get("meta", {})
    }

# Gate 1: normalize input to dict shape
start = RunnableLambda(lambda x: {"query": x["query"] if isinstance(x, dict) else x})

# Gate 2: T1 QA lookup
with_t1_qa = start.assign(qa=lambda x: call_t1_qa_lookup(x["query"]))

# Branch: if qa.found -> stop; else -> run T2/T3 + verdict
def _maybe_run_t2_t3(x: Dict[str, Any]) -> Dict[str, Any]:
    if x["qa"].get("found"):
        if x.get("verify_cached"):
            cached_cfg = _extract_config_from_answer(x["qa"]["answer"])
            if cached_cfg:
                eval_payload = _build_payload_from_cached_config(cached_cfg)
                validation = call_t3_validate(eval_payload)
                verdict_text = verdict_chain.invoke({
                    "query": x["query"],
                    "config": cached_cfg,
                    "validation_json": json.dumps(validation, ensure_ascii=False)
                }, config={"callbacks": _callbacks})
                return {"final": _structure_final_from_t1_verified(x, verdict_text, validation, cached_cfg)}
        return {"final": _structure_final_from_t1(x)}

    attempts = 0
    loopback_used = False

    # Primary generation (with RAG based on global setting)
    rag_enabled = x.get("rag_enabled", True)  # Default to RAG enabled
    norm1 = call_t2_generate(x["query"], model=get_cfg("T2_DEFAULT_MODEL", "gemini"), rag_enabled=rag_enabled)
    fused1 = _fuse_changes(norm1["evaluate_payload"]["changes"])
    norm1["evaluate_payload"]["changes"] = fused1
    val1 = call_t3_validate(norm1["evaluate_payload"])
    attempts += 1
    verdict_text1 = verdict_chain.invoke({
        "query": x["query"],
        "config": norm1["joined_config"],
        "validation_json": json.dumps(val1, ensure_ascii=False)
    }, config={"callbacks": _callbacks})
    verdict_status1 = _parse_verdict_status(verdict_text1)

    chosen_norm = norm1
    chosen_val = val1
    chosen_vtext = verdict_text1
    chosen_vstatus = verdict_status1
    chosen_fused = fused1

    # Loopback fallback only if FAIL AND loopback_enabled flag is set
    loopback_enabled = x.get("loopback_enabled", LOOPBACK_ON_FAIL)
    if verdict_status1 == "FAIL" and loopback_enabled:
        loopback_used = True
        # Force RAG off for loopback attempt
        norm2 = call_t2_generate(x["query"], model=get_cfg("T2_DEFAULT_MODEL", "gemini"), rag_enabled=False)
        fused2 = _fuse_changes(norm2["evaluate_payload"]["changes"])
        norm2["evaluate_payload"]["changes"] = fused2
        val2 = call_t3_validate(norm2["evaluate_payload"])
        attempts += 1
        verdict_text2 = verdict_chain.invoke({
            "query": x["query"],
            "config": norm2["joined_config"],
            "validation_json": json.dumps(val2, ensure_ascii=False)
        }, config={"callbacks": _callbacks})
        verdict_status2 = _parse_verdict_status(verdict_text2)

        if verdict_status2 == "PASS":
            chosen_norm = norm2
            chosen_val = val2
            chosen_vtext = verdict_text2
            chosen_vstatus = verdict_status2
            chosen_fused = fused2

    return {
        "query": x["query"],
        "config": chosen_norm["joined_config"],
        "validation": chosen_val,
        "verdict_text": chosen_vtext,
        "verdict_status": chosen_vstatus,
        "fused_changes": chosen_fused,
        "meta": {
            "attempts": attempts,
            "loopback_used": loopback_used,
            "rag_primary": rag_enabled,
            "loopback_enabled": loopback_enabled,
            "devices": list(chosen_fused.keys())
        }
    }

# Final chain assembly (adjusted to pass verdict_status)
AgentChain: RunnableSequence = (
    with_t1_qa
    .assign(maybe=_maybe_run_t2_t3)
    .assign(result=lambda x: x["maybe"]["final"] if "final" in x["maybe"]
            else _structure_final_from_t2_t3({**x, **x["maybe"]}))
    .pick("result")
)

def run_agent(query: str, verify_cached: bool = False, rag_enabled: bool = True, loopback_enabled: bool = None) -> Dict[str, Any]:
    """
    Run the agent pipeline.
    
    Args:
        query: User query
        verify_cached: Whether to verify cached answers
        rag_enabled: Enable RAG (retrieval-augmented generation)
        loopback_enabled: Enable loopback fallback on failure (None = use config default)
    """
    if loopback_enabled is None:
        loopback_enabled = LOOPBACK_ON_FAIL
    
    out = AgentChain.invoke({
        "query": query,
        "verify_cached": verify_cached,
        "rag_enabled": rag_enabled,
        "loopback_enabled": loopback_enabled
    }, config={"callbacks": _callbacks})
    
    log.info("Agent v2 result: %s", json.dumps(out, indent=2))
    if LOG_EVENTS:
        PipelineCallback()._write("agent_result", {
            "query": query,
            "source": out.get("source"),
            "has_config": bool(out.get("config")),
            "verify_cached": verify_cached,
            "rag_enabled": rag_enabled,
            "loopback_enabled": loopback_enabled
        })
    return out

def test_t2_t3_pipeline():
    """
    Manual test for Team 2 + Team 3 integration.
    Allows choosing the model (llama or gemini).
    """
    # Example query
    query = "Generate OSPF area 0 config for 3 Cisco routers with loopbacks."
    model = "gemini"  # 🧠 change to "gemini" or "llama" to test different models

    print(f"🔹 Calling Team 2 (config generation) using model='{model}' ...")
    gen = call_t2_generate(query, model=model)
    print(json.dumps(gen, indent=2, ensure_ascii=False))

    print("\n🔹 Calling Team 3 (validation)...")
    val = call_t3_validate(gen["evaluate_payload"])
    print(json.dumps(val, indent=2, ensure_ascii=False))

    print("\n✅ Test complete")

if __name__ == "__main__":
    import argparse
    import traceback

    parser = argparse.ArgumentParser(
        prog="Agent Orchestrator v2",
        description="Run the full pipeline (T0 -> T2 -> T3 -> LLM verdict -> T1 optional) with detailed diagnostics."
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Generate OSPF area 0 config for 3 Cisco routers with loopback"
    )
    args = parser.parse_args()
    
    # Simple CLI runner
    try:
        print("Agent Orchestrator v2 - Query:", args.query)
        res = run_agent(args.query)
        print("Result:", json.dumps(res, indent=2))
    except Exception as e:
        print("Error:", str(e))
        traceback.print_exc()

def _normalize_cmds(cmds: List[str]) -> List[str]:
    """Trim whitespace, drop empty lines."""
    return [c.strip() for c in cmds if c and c.strip()]

def _filter_orphan_area(cmds: List[str]) -> List[str]:
    if not cmds:
        return []

    parent_prefixes = (
        "router ",           # router ospf 1 / router bgp 65000 / etc.
        "interface ",        # interface GigabitEthernet0/1
        "network ",          # network statement (OSPF/EIGRP)
        "address-family ",   # BGP address-family blocks
        "line ",             # line vty 0 4
        "ip route",          # static route base (rare modifier patterns)
    )

    modifier_prefixes = (
        "area ",               # OSPF area spec
        "metric ",             # routing metric adjustments
        "cost ",               # OSPF cost
        "passive-interface ",  # interface passive directive
        "timers ",             # generic timers line
        "redistribute ",       # redistribution line
        "default-information ",# OSPF default origination addition
        "route-map ",          # route-map (if orphan - rarely useful alone)
        "neighbor ",           # BGP/EIGRP neighbor directive
        "distance ",           # admin distance tuning
    )

    out: List[str] = []
    for line in cmds:
        raw = line.strip()
        low = raw.lower()

        # Detect orphan modifier
        is_modifier = any(low.startswith(pref) for pref in modifier_prefixes)
        if is_modifier:
            if out:
                prev = out[-1]
                prev_low = prev.lower()

                # If previous is a parent directive and does not contain the modifier already, merge
                if any(prev_low.startswith(p) for p in parent_prefixes) and low not in prev_low:
                    out[-1] = f"{prev} {raw}"  # append inline
                    continue  # merged -> skip adding standalone modifier
            # If cannot merge, drop the orphan modifier silently
            continue

        # Normal line, keep
        out.append(raw)

    return out

def _render_device_config(name: str, cmds: List[str]) -> str:
    """
    Render final per-device config text with proper IOS structure.
    Adds section separators for better Batfish parsing.
    """
    lines = [f"hostname {name}", "!"]
    
    # Group commands by section
    current_section = []
    for cmd in cmds:
        stripped = cmd.strip()
        if not stripped:
            continue
        
        # Start of new section (interface, router, etc.)
        if stripped.lower().startswith(("interface ", "router ", "line ")):
            if current_section:
                lines.extend(current_section)
                lines.append("!")
                current_section = []
            current_section.append(cmd)
        else:
            current_section.append(cmd)
    
    # Add remaining section
    if current_section:
        lines.extend(current_section)
        lines.append("!")
    
    # Ensure end marker
    lines.append("end")
    
    return "\n".join(lines) + "\n"

def _looks_like_config(line: str) -> bool:
    """Heuristic to decide if a line is a configuration command."""
    l = line.strip().lower()
    return (
        l.startswith(("interface ", "router ", "hostname ", "ip address ", "vlan ", "switchport ", "access-list ", "ipv6 ", "no "))
        or l.startswith("network ")
    )

def _extract_config_from_answer(answer: str) -> Optional[str]:
    """
    Attempt to extract config-like block from a cached QA answer.
    Returns joined config text or None if insufficient signal.
    """
    if not isinstance(answer, str):
        return None
    lines = [ln.rstrip() for ln in answer.splitlines() if ln.strip()]
    cfg_lines = [ln for ln in lines if _looks_like_config(ln)]
    # Require a minimum number of config lines
    if len(cfg_lines) < 3:
        return None
    # Ensure a hostname if absent
    if not any(l.lower().startswith("hostname ") for l in cfg_lines):
        cfg_lines.insert(0, "hostname CachedDevice")
    return "\n".join(cfg_lines) + "\n"

def _build_payload_from_cached_config(config_text: str) -> Dict[str, Any]:
    """
    Build a minimal evaluate_payload structure from a single cached config text.
    """
    cmds = [c for c in (config_text.splitlines()) if c.strip()]
    # Strip leading hostname into device name if present
    dev_name = "CachedDevice"
    if cmds and cmds[0].lower().startswith("hostname "):
        dev_name = cmds[0].split(None, 1)[1].strip()
        cmds = cmds[1:]
    changes = {dev_name: cmds}
    device_configs = {dev_name: "hostname " + dev_name + "\n" + "\n".join(cmds) + "\n"}
    return {
        "changes": changes,
        "snapshot": {"configs": device_configs},
        "intent": {
            "reach": [],  # unknown from cached answer
            "interface": [],
            "policy": [],
            "redundancy": []
        },
        "meta": {
            "devices": 1,
            "raw_intents": 0,
            "active_types": [],
            "reach_pairs": 0,
            "fallback_used": False,
            "source": "cached_answer"
        }
    }

def _ensure_no_shutdown_on_all_interfaces(cmds: List[str]) -> List[str]:
    """
    Ensure every interface block ends with 'no shutdown' if missing.
    Handles both Loopback and physical interfaces.
    """
    out = []
    in_interface = False
    interface_has_no_shut = False
    
    for line in cmds:
        l = line.strip().lower()
        
        # Detect interface block start
        if l.startswith("interface "):
            # Finalize previous interface if needed
            if in_interface and not interface_has_no_shut:
                out.append(" no shutdown")
            
            in_interface = True
            interface_has_no_shut = False
            out.append(line)
        
        # Check for existing no shutdown
        elif in_interface and l == "no shutdown":
            interface_has_no_shut = True
            out.append(line)
        
        # End of interface block (router/line/hostname/etc)
        elif l.startswith(("router ", "line ", "hostname ", "!", "end", "exit")):
            if in_interface and not interface_has_no_shut:
                out.append(" no shutdown")
            in_interface = False
            out.append(line)
        
        else:
            out.append(line)
    
    # Handle last interface
    if in_interface and not interface_has_no_shut:
        out.append(" no shutdown")
    
    return out
