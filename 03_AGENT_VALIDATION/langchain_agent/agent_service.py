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

# ========= CONFIG LOAD FROM JSON =========
CONFIG_PATH = "config.json"
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

def call_t2_generate(query: str) -> str:
    data = _post(f"{T2_BASE_URL}{T2_GENERATE}", {"query": query})
    cfg = data.get("config")
    if not isinstance(cfg, str) or not cfg.strip():
        raise HTTPException(status_code=502, detail=f"T2 invalid response: {data}")
    return cfg

def call_t3_validate(config_text: str) -> Dict[str, Any]:
    """
    Expected:
      { "status": "PASS" | "FAIL", "report": "...", ... }
    """
    return _post(f"{T3_BASE_URL}{T3_VALIDATE}", {"config": config_text})

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

if __name__ == "__main__":
    print(json.dumps(run_agent("Generate OSPF area 0 config for 3 Cisco routers with loopbacks."), indent=2))
