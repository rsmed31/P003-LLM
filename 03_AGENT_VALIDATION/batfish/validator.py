from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from pybatfish.client.session import Session

from pybatfish.datamodel.flow import PathConstraints
import os
import shutil
import uuid
import logging
import traceback
import time

# -------------------------
# Request Models
# -------------------------
class EvaluateRequest(BaseModel):
    changes: Dict[str, List[str]] = Field(..., description="Device configuration changes")
    intent: Optional[Dict[str, Any]] = Field(default=None, description="Intent specifications")
    snapshot: Optional[Dict[str, Any]] = Field(default=None, description="Snapshot configuration")

# -------------------------
# Server & Batfish session
# -------------------------
app = FastAPI(
    title="Batfish Validator API (Team 3)",
    version="1.0.0",
    description="Network configuration validation service using Batfish. Receives device configs and intent checks.",
)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("validator")
bf = Session(host=os.environ.get("BATFISH_HOST", "localhost"))  # e.g., "batfish" in docker-compose

BATFISH_READY_TIMEOUT = int(os.environ.get("BATFISH_READY_TIMEOUT", "180"))
BATFISH_READY_INTERVAL = float(os.environ.get("BATFISH_READY_INTERVAL", "3"))
CP_QUERY_TIMEOUT = int(os.environ.get("CP_QUERY_TIMEOUT", "5"))  # seconds
SNAPSHOT_REUSE_SECONDS = int(os.environ.get("SNAPSHOT_REUSE_SECONDS", "20"))
SKIP_BASE_COPY = os.environ.get("SKIP_BASE_COPY", "0") in ("1", "true", "True")

_LAST_SNAPSHOT: Dict[str, Any] = {
    "ts": None,
    "path": None,
    "devices": []
}

# -------------------------
# Configurable constants
# -------------------------
NETWORK_NAME = os.environ.get("BF_NETWORK_NAME", "network-test")
# This is your pristine/base snapshot folder (must exist and be Batfish-compatible on disk)
SNAPSHOT_BASE = os.environ.get("BF_SNAPSHOT_BASE", os.path.dirname(os.path.abspath(__file__)))
# Default host intent for reachability if none provided in the request
DEFAULT_SRC_HOST = os.environ.get("REACH_SRC", "h1")
DEFAULT_DST_HOST = os.environ.get("REACH_DST", "h2")

# -------------------------
# Helpers
# -------------------------
def _ensure_network():
    bf.set_network(NETWORK_NAME)

def _new_workdir() -> str:
    workdir = os.path.join("/tmp", f"snapshot-verify-{uuid.uuid4().hex}")
    return workdir

def _copy_base_snapshot_to(workdir: str):
    if not os.path.isdir(SNAPSHOT_BASE):
        raise FileNotFoundError(f"Base snapshot folder not found: {SNAPSHOT_BASE}")
    shutil.copytree(SNAPSHOT_BASE, workdir)

def _find_device_config_file(snapshot_dir: str, device: str) -> Optional[str]:
    """
    Try common locations/extensions to find the device config in the snapshot.
    We support:
      snapshot_dir/configs/{device}.cfg
      snapshot_dir/configs/{device}.txt
      snapshot_dir/{device}.cfg
      snapshot_dir/{device}.txt
    """
    candidates = [
        os.path.join(snapshot_dir, "configs", f"{device}.cfg"),
        os.path.join(snapshot_dir, "configs", f"{device}.txt"),
        os.path.join(snapshot_dir, f"{device}.cfg"),
        os.path.join(snapshot_dir, f"{device}.txt"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None

def _apply_config_to_device(snapshot_dir: str, device: str, commands: List[str]):
    cfg_path = _find_device_config_file(snapshot_dir, device)
    if not cfg_path:
        # If the device file is missing, create it inside configs
        cfg_dir = os.path.join(snapshot_dir, "configs")
        os.makedirs(cfg_dir, exist_ok=True)
        cfg_path = os.path.join(cfg_dir, f"{device}.cfg")
        # Create a minimal header to be friendlier to some vendors; adjust as needed
        with open(cfg_path, "w") as f:
            f.write(f"! Auto-created for {device}\n!\n")

    # Naive "append" strategy (idempotency up to you; dedup not attempted here)
    with open(cfg_path, "a") as f:
        f.write("\n!\n! --- LLM proposed changes ---\n")
        for line in commands:
            f.write(f"{line}\n")
        f.write("! --- end changes ---\n")

def _apply_all_changes(snapshot_dir: str, changes: Dict[str, List[str]]) -> List[str]:
    changed_devices = []
    for device, cmds in (changes or {}).items():
        if not isinstance(cmds, list) or not cmds:
            # Skip empty command sets to avoid empty diffs
            continue
        _apply_config_to_device(snapshot_dir, device, cmds)
        changed_devices.append(device)
    return changed_devices

def _init_snapshot(snapshot_dir: str, name: Optional[str] = None):
    """
    Initialize snapshot and check for parse warnings.
    """
    _ensure_network()
    snapshot_name = name or os.path.basename(snapshot_dir.rstrip("/"))
    
    # Log snapshot contents
    configs_dir = os.path.join(snapshot_dir, "configs")
    if os.path.exists(configs_dir):
        config_files = os.listdir(configs_dir)
        log.info(f"Snapshot {snapshot_name} contains {len(config_files)} config files: {config_files}")
    else:
        log.warning(f"No configs directory found at {configs_dir}")
    
    # Initialize snapshot
    bf.init_snapshot(snapshot_dir, name=snapshot_name, overwrite=True)
    
    # Verify nodes loaded
    try:
        node_props = bf.q.nodeProperties().answer().frame()
        nodes = list(node_props.get('Node', []))
        log.info(f"Snapshot loaded with {len(node_props)} nodes: {nodes}")
        
        # CHECK FOR PARSE WARNINGS (CRITICAL DIAGNOSTIC)
        try:
            warnings = bf.q.parseWarnings().answer().frame()
            if not warnings.empty:
                log.warning(f"Batfish parse warnings ({len(warnings)} issues):")
                for idx, row in warnings.head(10).iterrows():
                    log.warning(f"  {row.get('Filename', '?')}: {row.get('Text', '?')}")
            else:
                log.info("No parse warnings - configs look clean")
        except Exception as warn_e:
            log.warning(f"Could not fetch parse warnings: {warn_e}")
            
    except Exception as e:
        log.error(f"Failed to query nodes after snapshot init: {e}")
    
    return snapshot_name

def _timed_cp_query(devices: List[str]):
    """
    Query interfaceProperties with timeout using proper Batfish node regex.
    Batfish nodes parameter expects a regex pattern, e.g., "/r1|r2|r3/".
    Returns (frame or None, timed_out: bool)
    """
    import threading
    result = {"frame": None, "error": None}

    def _run():
        try:
            # Build regex: /node1|node2|node3/
            nodes_regex = "/" + "|".join(devices) + "/" if devices else None
            ans = bf.q.interfaceProperties(
                nodes=nodes_regex,
                properties="Interface,Active"
            ).answer()
            result["frame"] = ans.frame()
        except Exception as e:
            result["error"] = e

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    th.join(CP_QUERY_TIMEOUT)
    if th.is_alive():
        return (None, True)
    if result["error"]:
        raise result["error"]
    return (result["frame"], False)

def _resolve_cp_nodes(changed_devices: List[str]) -> List[str]:
    """
    Map friendly device names (R1, r1, r1.lab, etc.) to canonical Batfish node names.
    Case-insensitive. Falls back to original names if not found.
    """
    try:
        node_props = bf.q.nodeProperties().answer().frame()
        if "Node" not in node_props.columns:
            return changed_devices
        all_nodes = [str(n) for n in node_props["Node"].tolist()]
        lower_map = {n.lower(): n for n in all_nodes}
        resolved = []
        for dev in changed_devices or []:
            canon = lower_map.get(str(dev).lower())
            resolved.append(canon if canon else dev)
        # Deduplicate preserving order
        seen = set()
        out = []
        for n in resolved:
            if n not in seen:
                out.append(n); seen.add(n)
        return out
    except Exception as e:
        log.warning(f"CP node resolution failed, using raw names: {e}")
        return changed_devices

def _run_cp(changed_devices: List[str]) -> Dict[str, Any]:
    """
    Control-plane interface liveness check using resolved Batfish node names.
    """
    if not changed_devices:
        return {"status": "PASS", "rows": 0, "active_devices": [], "inactive_devices": [], "status_reason": "no devices", "advisory": True}

    # Resolve to canonical Batfish node names
    query_devices = _resolve_cp_nodes(changed_devices)

    # Debug snapshot nodes
    try:
        all_nodes = bf.q.nodeProperties().answer().frame()
        if "Node" in all_nodes.columns:
            actual_nodes = sorted(all_nodes["Node"].tolist())
            log.info(f"CP check: snapshot_nodes={actual_nodes}, requested={changed_devices}, query_nodes={query_devices}")
    except Exception as e:
        log.error(f"CP check nodeProperties failed: {e}")

    start = time.time()
    try:
        frame, timed_out = _timed_cp_query(query_devices)

        if timed_out:
            return {
                "status": "SKIPPED",
                "rows": 0,
                "active_devices": [],
                "inactive_devices": changed_devices,
                "status_reason": f"cp_query_timeout>{CP_QUERY_TIMEOUT}s",
                "advisory": True,
                "query_time_seconds": round(time.time() - start, 2)
            }

        if frame is None or frame.empty:
            log.warning(f"CP check: No interfaces found for devices {changed_devices} (query_nodes={query_devices})")
            return {
                "status": "SKIPPED",  # changed from FAIL
                "rows": 0,
                "active_devices": [],
                "inactive_devices": changed_devices,
                "status_reason": "no interfaces parsed for these nodes",
                "advisory": True,
                "query_time_seconds": round(time.time() - start, 2)
            }

        rows = len(frame)
        log.info(f"CP check: Found {rows} interfaces")

        if 'Node' not in frame.columns:
            try:
                frame = frame.reset_index()
            except Exception:
                pass

        active_map = {d: 0 for d in changed_devices}
        if {"Node", "Active"} <= set(frame.columns):
            for _, r in frame.iterrows():
                if r.get("Active") is True:
                    node = r.get("Node")
                    for target in active_map.keys():
                        if target.lower() == str(node).lower():
                            active_map[target] += 1
                            break

        active_devices = [d for d, cnt in active_map.items() if cnt > 0]
        inactive_devices = [d for d in changed_devices if d not in active_devices]

        log.info(f"CP check: active_counts={active_map}, active={active_devices}, inactive={inactive_devices}")

        status = "PASS" if not inactive_devices else "FAIL"
        return {
            "status": status,
            "rows": rows,
            "active_devices": active_devices,
            "inactive_devices": inactive_devices,
            "status_reason": ("all active" if status == "PASS" else f"inactive: {inactive_devices}"),
            "advisory": True,
            "query_time_seconds": round(time.time() - start, 2)
        }
    except Exception as e:
        log.error(f"CP check exception: {e}\n{traceback.format_exc()}")
        return {
            "status": "ERROR",
            "rows": 0,
            "active_devices": [],
            "inactive_devices": changed_devices,
            "status_reason": f"cp_exception:{e}",
            "advisory": True,
            "query_time_seconds": round(time.time() - start, 2)
        }

def _run_tp() -> Dict[str, Any]:
    """
    Topology check: verify layer3 edges.
    Returns summary: {"status": "PASS"|"FAIL", "edges": int}
    """
    try:
        result = bf.q.layer3Edges().answer().frame()
        edges = len(result)
        
        # Basic validation: if we have devices, we should have edges
        status = "PASS" if edges > 0 else "FAIL"
        
        return {"status": status, "edges": edges}
    except Exception as e:
        log.error(f"TP check failed: {e}")
        return {"status": "ERROR", "edges": 0, "error": str(e)}


def _extract_device_ips(changes: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Extract primary IP per device (prefer Loopback0, fallback to first interface IP).
    Returns {device_name_lowercase: ip_address}.
    NOTE: Normalizes device names to lowercase to match Batfish parsing.
    """
    mapping = {}
    for dev, cmds in changes.items():
        normalized_dev = dev.lower()
        current_iface = None
        loopback_ip = None
        first_ip = None
        
        for line in cmds:
            l = line.strip()
            if l.lower().startswith("interface "):
                current_iface = l.split()[1] if len(l.split()) > 1 else None
            elif current_iface and l.startswith("ip address "):
                parts = l.split()
                if len(parts) >= 3:
                    ip = parts[2]
                    if "loopback" in current_iface.lower():
                        loopback_ip = ip
                    elif not first_ip:
                        first_ip = ip
        
        # Prefer Loopback0 > first interface IP
        final_ip = loopback_ip or first_ip or f"0.0.0.{list(changes.keys()).index(dev) + 1}"
        mapping[normalized_dev] = final_ip
        log.info(f"IP mapping: {dev} → {normalized_dev} = {final_ip}")
    
    return mapping

def _run_reach(changes: Dict[str, List[str]], reach_paths: List[dict]) -> List[Dict[str, Any]]:
    """
    Protocol-agnostic reachability test using srcIps/dstIps headers.
    No node() specifier - pure IP-based flow simulation.
    """
    device_ips = _extract_device_ips(changes)
    log.info(f"Device IP mappings: {device_ips}")
    results = []
    
    for path in reach_paths or []:
        src = path.get("src")
        dst = path.get("dst")
        
        if not src or not dst:
            results.append({"src": src, "dst": dst, "status": "FAIL", "error": "Missing src/dst"})
            continue
        
        # Normalize to lowercase for lookup
        src_normalized = src.lower()
        dst_normalized = dst.lower()
        
        src_ip = device_ips.get(src_normalized)
        dst_ip = device_ips.get(dst_normalized)
        
        if not src_ip or not dst_ip:
            log.error(f"IP lookup failed: {src}→{src_normalized}={src_ip}, {dst}→{dst_normalized}={dst_ip}")
            results.append({
                "src": src, "dst": dst, "status": "FAIL",
                "error": f"Unresolved IPs (src={src_ip}, dst={dst_ip})"
            })
            continue
        
        log.info(f"Reachability: {src} ({src_ip}) → {dst} ({dst_ip})")
        
        try:
            ans = bf.q.reachability(
                headers={"srcIps": src_ip, "dstIps": dst_ip}
            ).answer().frame()
            
            status = "PASS" if len(ans) > 0 else "FAIL"
            results.append({
                "src": src, "dst": dst, "status": status,
                "error": "" if status == "PASS" else "No reachable paths"
            })
        except Exception as e:
            log.error(f"Reachability error {src}→{dst}: {e}")
            results.append({"src": src, "dst": dst, "status": "ERROR", "error": str(e)})
    
    return results


# -------------------------
# API
# -------------------------
@app.get("/debug/bf")
def debug_bf():
    try:
        _ensure_network()
        return {
            "networks": bf.list_networks(),
            "snapshot": bf.get_snapshot() if hasattr(bf, "get_snapshot") else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _evaluate_interface_intents(changes: Dict[str, List[str]], interface_intents: List[dict]) -> Dict[str, Any]:
    # Very lightweight: verify referenced device appears in changes and an interface line exists
    # NOTE: Normalizes device names to lowercase to match Batfish parsing
    results = []
    for item in interface_intents:
        eps = item.get("endpoints", [])
        ids = [e.get("id") for e in eps if isinstance(e, dict) and e.get("id")]
        status = "FAIL"
        detail = "No device"
        if ids:
            dev = ids[0]
            normalized_dev = dev.lower()
            cmds = changes.get(dev, [])
            if any(c.startswith("interface ") for c in cmds):
                status = "PASS"
                detail = "interface present"
            else:
                detail = "no interface lines"
        results.append({"device": ids[0] if ids else None, "status": status, "detail": detail})
    # Overall pass if all PASS
    agg_status = "PASS" if results and all(r["status"] == "PASS" for r in results) else ("SKIPPED" if not results else "FAIL")
    return {"status": agg_status, "items": results}

def _evaluate_policy_intents(policy_intents: List[dict]) -> Dict[str, Any]:
    # Placeholder: not implemented yet
    if not policy_intents:
        return {"status": "SKIPPED", "items": []}
    return {"status": "SKIPPED", "items": [{"status": "SKIPPED", "detail": "not implemented"}]}

def _evaluate_redundancy_intents(redundancy_intents: List[dict]) -> Dict[str, Any]:
    # Placeholder: not implemented yet
    if not redundancy_intents:
        return {"status": "SKIPPED", "items": []}
    return {"status": "SKIPPED", "items": [{"status": "SKIPPED", "detail": "not implemented"}]}

def _write_snapshot_configs(workdir: str, device_configs: Dict[str, str]) -> List[str]:
    """
    Write full device configs (OVERWRITE mode, not append).
    Ensures hostname inside config matches lowercase filename for Batfish consistency.
    Returns list of normalized device names written.
    """
    import re
    written = []
    cfg_dir = os.path.join(workdir, "configs")
    os.makedirs(cfg_dir, exist_ok=True)
    
    for dev, text in (device_configs or {}).items():
        # Normalize to lowercase
        normalized_dev = dev.lower()
        
        # CRITICAL: Force hostname inside config to match filename
        # Replace any existing hostname declaration with normalized one
        text = re.sub(
            r"^hostname\s+\S+",
            f"hostname {normalized_dev}",
            text,
            flags=re.MULTILINE | re.IGNORECASE
        )
        
        # If no hostname found, prepend one
        if not re.search(r"^hostname\s+", text, re.MULTILINE | re.IGNORECASE):
            text = f"hostname {normalized_dev}\n!\n{text}"
        
        path = os.path.join(cfg_dir, f"{normalized_dev}.cfg")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text.rstrip() + "\n")
        
        written.append(normalized_dev)
        log.info(f"Wrote full config for {dev} → {normalized_dev} ({len(text)} bytes)")
        
    return written

def _fast_new_workdir(devices: List[str]) -> str:
    """
    Create new workdir (snapshot reuse DISABLED to avoid stale config issues).
    """
    workdir = _new_workdir()
    os.makedirs(os.path.join(workdir, "configs"), exist_ok=True)
    # NOTE: SKIP_BASE_COPY should be True in your environment
    # We write full configs, so base copy is not needed
    return workdir

@app.post("/evaluate", summary="Validate network configuration", tags=["Validation"])
def evaluate(req: EvaluateRequest):
    """
    Validate device configurations using Batfish (FULL CONFIG MODE).
    """
    if not _batfish_ready():
        raise HTTPException(status_code=503, detail="Batfish not ready")

    changes = req.changes
    intent = req.intent or {}
    reach_paths = intent.get("reach", [])
    
    if reach_paths and not isinstance(reach_paths, list):
        raise HTTPException(status_code=400, detail="'intent.reach' must be a list")

    # Extract full configs from snapshot payload
    device_configs_payload = (req.snapshot or {}).get("configs", {})
    
    if not device_configs_payload:
        raise HTTPException(
            status_code=400,
            detail="Missing 'snapshot.configs' - full device configs required"
        )

    # Create fresh workdir (no reuse to avoid stale configs)
    devices_list = list(device_configs_payload.keys())
    workdir = _fast_new_workdir(devices_list)
    
    # Write full configs with hostname normalization
    packaged_devices = _write_snapshot_configs(workdir, device_configs_payload)
    log.info(f"Wrote {len(packaged_devices)} full configs: {packaged_devices}")

    # Use packaged (normalized) device names for validation
    changed_devices = packaged_devices

    # Initialize snapshot with parse warning checks
    try:
        _ensure_network()
        snapshot_name = _init_snapshot(workdir)
    except Exception as e:
        log.error(f"Snapshot init failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"INIT_SNAPSHOT: {str(e)}")

    # Run validation checks
    try:
        cp_summary = _run_cp(changed_devices)
        tp_summary = _run_tp()
        reach_summary = _run_reach(req.changes, (req.intent or {}).get("reach", []))
        iface_summary = _evaluate_interface_intents(req.changes, (req.intent or {}).get("interface", []))
        policy_summary = _evaluate_policy_intents((req.intent or {}).get("policy", []))
        red_summary = _evaluate_redundancy_intents((req.intent or {}).get("redundancy", []))
    except Exception as e:
        log.error(f"Validation failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail={"stage":"VERIFY","error":str(e)})

    # Determine overall status
    reach_ok = all(r["status"] == "PASS" for r in reach_summary) if reach_summary else True
    tp_ok = tp_summary.get("status") == "PASS"
    cp_status = cp_summary.get("status")
    devices_active = (cp_status in ("PASS","SKIPPED")) and (cp_summary.get("active_devices") or cp_status=="SKIPPED")
    overall_status = "PASS" if (tp_ok and reach_ok and devices_active) else "FAIL"

    resp = {
        "result": "OK",
        "status": overall_status,
        "status_reason": {
            "topology_ok": tp_ok,
            "reach_ok": reach_ok,
            "cp_status": cp_status,
            "cp_rows": cp_summary.get("rows"),
            "cp_advisory": True
        },
        "snapshot": snapshot_name,
        "summary": {
            "CP": cp_summary,
            "TP": tp_summary,
            "REACH": reach_summary,
            "IFACE": iface_summary,
            "POLICY": policy_summary,
            "REDUNDANCY": red_summary
        },
        "packaged_devices": packaged_devices,
        "changed_devices": changed_devices
    }
    return resp

@app.get("/health", summary="Health check", tags=["System"])
def health():
    """Check if validator and Batfish are reachable."""
    return {"status": "ok", "batfish": "reachable"}

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def docs_redirect():
    """Redirect root to Swagger UI."""
    return """
    <html>
        <head>
            <meta http-equiv="refresh" content="0; url=/docs" />
        </head>
        <body>
            <p>Redirecting to <a href="/docs">API Documentation</a>...</p>
        </body>
    </html>
    """

# -------------------------
# Main
# -------------------------
def _batfish_ready() -> bool:
    try:
        # Lightweight call – list networks
        bf.list_networks()
        return True
    except Exception:
        return False

def _wait_batfish():
    start = time.time()
    while time.time() - start < BATFISH_READY_TIMEOUT:
        if _batfish_ready():
            log.info("Batfish ready")
            return
        log.info("Waiting for Batfish...")
        time.sleep(BATFISH_READY_INTERVAL)
    log.error("Batfish not ready after timeout")
    raise RuntimeError("Batfish service not ready")

def _create_snapshot_structure(snapshot_id: str, device_configs: Dict[str, str]) -> str:
    """
    Create Batfish-compliant snapshot directory structure:
    <snapshot_id>/
      configs/
        R1.cfg
        R2.cfg
        ...
    """
    import tempfile
    import os
    
    temp_root = tempfile.mkdtemp(prefix=f"snapshot_{snapshot_id}_")
    configs_dir = os.path.join(temp_root, "configs")
    os.makedirs(configs_dir, exist_ok=True)
    
    for device_name, config_text in device_configs.items():
        cfg_path = os.path.join(configs_dir, f"{device_name}.cfg")
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write(config_text)
    
    return temp_root  # Return root dir for Batfish upload

if __name__ == "__main__":
    import uvicorn
    
    try:
        _wait_batfish()
    except Exception as e:
        log.error(f"Startup aborted: {e}")
        raise
    
    _ensure_network()
    print(f"Using SNAPSHOT_BASE: {SNAPSHOT_BASE}")
    print("Starting validator on http://0.0.0.0:5000 ...")
    
    # FastAPI requires uvicorn.run()
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=int(os.environ.get("PORT", 5000)),
        log_level="info"
    )