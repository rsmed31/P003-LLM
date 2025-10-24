#!/usr/bin/env python3
import os, json, logging
from typing import Dict, Any, Optional
import requests

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# LangChain (no agent reasoning needed; we use LCEL + an LLM only for the final explanation)
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import os, json, logging
from typing import List, Tuple, Set

# ========= CONFIG LOAD FROM JSON =========
HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)
else:
    cfg = {}

def get_cfg(key, default=None):
    return cfg.get(key) or os.getenv(key, default)

T0_BASE_URL = get_cfg("T0_BASE_URL", "http://t0:8000").rstrip("/")
T2_BASE_URL = get_cfg("T2_BASE_URL", "http://t2:8002").rstrip("/")
T3_BASE_URL = get_cfg("T3_BASE_URL", "http://t3:8003").rstrip("/")
T1_BASE_URL = get_cfg("T1_BASE_URL", "").rstrip("/")

# Endpoints
T0_QA_LOOKUP = get_cfg("T0_ENDPOINT_QA_LOOKUP", "/qa_lookup")
T2_GENERATE = get_cfg("T2_ENDPOINT_GENERATE", "/generate_config")
T3_VALIDATE = get_cfg("T3_ENDPOINT_VALIDATE", "/evaluate")
T1_WRITE = get_cfg("T1_ENDPOINT_WRITE", "/verify_write")

TIMEOUT = float(get_cfg("HTTP_TIMEOUT", 90))
LOG_LEVEL = get_cfg("LOG_LEVEL", "INFO").upper()

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("agent-v2")

# ========= HTTP helper =========
def _post(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(url, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def _get(url: str) -> Dict[str, Any]:
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

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

def _build_evaluate_payload_from_t2(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform Team2 output into T3 /evaluate payload:
      {
        "changes": {"R1": [...], "R2": [...]},
        "intent": {"reach": [{"src":"R1","dst":"R2"}, ...]}
      }
    Also returns a human-readable 'joined_config' for logging/LLM verdict.
    """
    resp = data.get("response")
    if not isinstance(resp, list) or not resp:
        raise HTTPException(status_code=502, detail=f"T2 payload not understood: keys={list(data.keys())}")

    changes: Dict[str, List[str]] = {}
    all_intents: List[dict] = []

    for dev in resp:
        name = dev.get("device_name") or f"device_{len(changes)+1}"
        cmds = _clean_commands(dev.get("configuration_mode_commands") or [])
        if not cmds:
            # still allow empty, but warn
            logging.warning(f"[T2] {name} produced no actionable commands.")
        changes[name] = cmds
        all_intents.extend(dev.get("intent") or [])

    # Convert intents → reach list
    reach = _intents_to_reach(all_intents)

    # Build a readable joined config (optional, for LLM/reporting)
    parts = []
    for name, cmds in changes.items():
        body = "\n".join(cmds)
        parts.append(f"! ===== BEGIN {name} =====\nhostname {name}\n{body}\n! ===== END {name} =====")
    joined_config = "\n".join(parts).strip() + "\n"

    # Final payload for T3
    evaluate_payload = {
        "changes": changes,
        "intent": {"reach": reach} if reach else {"reach": []}
    }
    return {"evaluate_payload": evaluate_payload, "joined_config": joined_config}


# ========= Team calls =========
def call_t0_lookup(query: str) -> Dict[str, Any]:
    """
    T0: Q&A lookup.
    Expected response:
      { "hit": true, "answer": "..." }  OR  { "hit": false }
    """
    try:
        data = _post(f"{T0_BASE_URL}{T0_QA_LOOKUP}", {"query": query})
        hit = bool(data.get("hit", False))
        if hit and not isinstance(data.get("answer", None), str):
            # sanitize weird payloads
            return {"hit": False}
        return {"hit": hit, "answer": data.get("answer")}
    except Exception as e:
        log.warning(f"T0 lookup failed (continuing to T2): {e}")
        return {"hit": False}

def call_t2_generate(query: str, model: str = "gemini") -> Dict[str, Any]:
    """
    Calls Team 2's GET endpoint (/v1/getAnswer) with a configurable model.
    Returns normalized payload for Team 3.
    """
    params = {"q": query, "model": model}
    url = f"{T2_BASE_URL}{T2_GENERATE}"

    resp = requests.get(url, params=params, timeout=TIMEOUT)
    resp.raise_for_status()

    try:
        raw = resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"T2 returned invalid JSON: {e}")

    norm = _build_evaluate_payload_from_t2(raw)
    if not norm["evaluate_payload"]["changes"]:
        raise HTTPException(status_code=502, detail="T2 produced no device changes")

    return norm




def call_t3_validate(evaluate_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sends the normalized payload to T3's /evaluate.
    Expected reply: { "status": "PASS" | "FAIL", "report": "...", ... }
    """
    return _post(f"{T3_BASE_URL}{T3_VALIDATE}", evaluate_payload)


def call_t1_write(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not T1_BASE_URL:
        return {"status": "SKIPPED", "reason": "no_t1"}
    try:
        _post(f"{T1_BASE_URL}{T1_WRITE}", payload)
        return {"status": "OK"}
    except Exception as e:
        log.warning(f"T1 write failed: {e}")
        return {"status": "ERROR", "error": str(e)}

# ========= LLM for final explanation =========
# Uses a small, deterministic model to “explain verdict”
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0,
    api_key=get_cfg("GROQ_API_KEY")
)

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

# ========= LangChain pipeline (deterministic with a conditional gate) =========
def _structure_final_from_t0(x: Dict[str, Any]) -> Dict[str, Any]:
    """When T0 hits, we stop and return the cached/KB answer."""
    return {
        "source": "T0",
        "query": x["query"],
        "answer": x["qa"]["answer"],
        "explanation": "Returned from Q&A knowledge base (cache hit).",
        "config": None,
        "validation": None,
        "write_back": {"status": "SKIPPED", "reason": "T0_hit"}
    }

def _structure_final_from_t2_t3(x: Dict[str, Any]) -> Dict[str, Any]:
    """When T0 misses, we go through T2/T3 and synthesize a verdict."""
    status = (x["validation"] or {}).get("status", "UNKNOWN")
    wb = {"status": "SKIPPED", "reason": "validation_failed"}
    if status == "PASS":
        wb = call_t1_write({"query": x["query"], "config": x["config"], "validation": x["validation"]})
    return {
        "source": "T2_T3",
        "query": x["query"],
        "config": x["config"],
        "validation": x["validation"],
        "verdict_text": x.get("verdict_text"),
        "write_back": wb
    }

# Gate 1: normalize input to dict shape
start = RunnableLambda(lambda x: {"query": x["query"] if isinstance(x, dict) else x})

# Gate 2: T0 lookup
with_t0 = start.assign(qa=lambda x: call_t0_lookup(x["query"]))

# Branch: if qa.hit -> stop; else -> run T2/T3 + verdict
def _maybe_run_t2_t3(x: Dict[str, Any]) -> Dict[str, Any]:
    if x["qa"].get("hit"):
        # Short-circuit: return final from T0
        return {"final": _structure_final_from_t0(x)}
    # Else: T2 generate
    cfg = call_t2_generate(x["query"])
    # T3 validate
    val = call_t3_validate(cfg)
    # LLM verdict
    vtext = verdict_chain.invoke({
        "query": x["query"],
        "config": cfg,
        "validation_json": json.dumps(val, ensure_ascii=False)
    })
    return {"query": x["query"], "config": cfg, "validation": val, "verdict_text": vtext}

AgentChain: RunnableSequence = (
    with_t0
    .assign(maybe=_maybe_run_t2_t3)
    .assign(result=lambda x:
            x["maybe"]["final"] if "final" in x["maybe"]
            else _structure_final_from_t2_t3({**x, **x["maybe"]})
    )
    .pick("result")
)

def run_agent(query: str) -> Dict[str, Any]:
    out = AgentChain.invoke({"query": query})
    log.info("Agent v2 result: %s", json.dumps(out, indent=2))
    return out

# ========= FastAPI wrapper =========
app = FastAPI(title="Agent Orchestrator v2 (LangChain)", version="2.0.0")

class RunAgentReq(BaseModel):
    query: str

@app.post("/run_agent")
def run_agent_api(req: RunAgentReq):
    return run_agent(req.query)


def test_t2_t3_pipeline():
    """
    Manual test for Team 2 + Team 3 integration.
    Allows choosing the model (llama or gemini).
    """
    # Example query
    query = "Generate OSPF area 0 config for 3 Cisco routers with loopbacks."
    model = "llama"  # 🧠 change to "gemini" if you want to test the other model

    print(f"🔹 Calling Team 2 (config generation) using model='{model}' ...")
    gen = call_t2_generate(query, model=model)  # pass the model here
    print(json.dumps(gen, indent=2, ensure_ascii=False))

    print("\n🔹 Calling Team 3 (validation)...")
    val = call_t3_validate(gen["evaluate_payload"])
    print(json.dumps(val, indent=2, ensure_ascii=False))

    print("\n✅ Test complete")


if __name__ == "__main__":
    print(json.dumps(run_agent("Generate OSPF area 0 config for 3 Cisco routers with loopbacks."), indent=2))