from flask import Flask, request, jsonify
from pybatfish.client.session import Session
from pybatfish.question import bfq
from pybatfish.datamodel.flow import PathConstraints
import os
import shutil
import uuid
from typing import Dict, List, Optional

# -------------------------
# Server & Batfish session
# -------------------------
app = Flask(_name_)
bf = Session(host=os.environ.get("BATFISH_HOST", "localhost"))  # e.g., "batfish" in docker-compose

# -------------------------
# Configurable constants
# -------------------------
NETWORK_NAME = os.environ.get("BF_NETWORK_NAME", "network-test")
# This is your pristine/base snapshot folder (must exist and be Batfish-compatible on disk)
SNAPSHOT_BASE = os.environ.get("BF_SNAPSHOT_BASE", "./snapshot")
# Default host intent for reachability if none provided in the request
DEFAULT_SRC_HOST = os.environ.get("REACH_SRC", "host1")
DEFAULT_DST_HOST = os.environ.get("REACH_DST", "host2")

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
    Initialize (or re-initialize) the Batfish snapshot from the given directory.
    """
    _ensure_network()
    snapshot_name = name or os.path.basename(snapshot_dir.rstrip("/"))
    bf.init_snapshot(snapshot_dir, name=snapshot_name, overwrite=True)
    return snapshot_name

def _run_cp(changed_devices: List[str]):
    """
    Configuration Properties: we query interfaceProperties.
    If 'changed_devices' is provided, we filter on those nodes; otherwise we return for all nodes.
    """
    q = bfq.interfaceProperties()
    if changed_devices:
        q = q.filter("?node", "in", changed_devices)
    ans = q.answer()
    frame = ans.frame() if ans else None
    rows = 0 if (frame is None or frame.empty) else len(frame)
    return {
        "status": "PASS" if rows >= 0 else "FAIL",  # CP here is informational; you can tighten criteria if desired
        "rows": rows
    }

def _run_tp():
    """
    Topology (L3). We return the edge count (informational).
    """
    ans = bfq.layer3Topology().answer()
    frame = ans.frame() if ans else None
    edges = 0 if (frame is None or frame.empty) else len(frame)
    return {
        "status": "PASS",   # Treat presence/absence as informational; change to policy-based PASS/FAIL if needed
        "edges": edges
    }

def _run_reach(paths: List[Dict[str, str]]):
    """
    Reachability for each requested src/dst pair. If none provided, default host1->host2.
    """
    results = []
    if not paths:
        paths = [{"src": DEFAULT_SRC_HOST, "dst": DEFAULT_DST_HOST}]

    for p in paths:
        src = p.get("src", DEFAULT_SRC_HOST)
        dst = p.get("dst", DEFAULT_DST_HOST)
        start = f"enter({src})"
        end = f"enter({dst})"

        try:
            ans = bfq.reachability(
                pathConstraints=PathConstraints(startLocation=start, endLocation=end)
            ).answer()
            frame = ans.frame() if ans else None
            # If the answer frame has at least one row, Batfish found reachable flows (depending on query settings)
            reachable = False if (frame is None or frame.empty) else True
            results.append({
                "src": src,
                "dst": dst,
                "status": "PASS" if reachable else "FAIL"
            })
        except Exception as e:
            results.append({
                "src": src,
                "dst": dst,
                "status": "ERROR",
                "error": str(e)
            })
    return results

# -------------------------
# API
# -------------------------
@app.route("/evaluate", methods=["POST"])
def evaluate():
    """
    Expected input (LLM team only provides changes; verifier runs all checks):
    {
      "changes": {
        "R1": ["interface g0/0", "description from-llm"],
        "R2": ["router ospf 100", "network 10.0.0.0 0.0.0.255 area 0"]
      },
      "intent": {
        "reach": [{"src":"host1","dst":"host2"}]  // optional; defaults used if omitted
      }
    }
    """
    payload = request.get_json(silent=True) or {}
    changes = payload.get("changes", {}) or {}
    intent = payload.get("intent", {}) or {}
    reach_paths = intent.get("reach", []) if isinstance(intent.get("reach", []), list) else []

    # 1) Prepare a fresh work snapshot
    workdir = _new_workdir()
    try:
        _copy_base_snapshot_to(workdir)
    except Exception as e:
        return jsonify({"result": "ERROR", "stage": "COPY_BASE_SNAPSHOT", "error": str(e)}), 400

    # 2) Apply LLM changes (if any) onto the work snapshot
    try:
        changed_devices = _apply_all_changes(workdir, changes)
    except Exception as e:
        # Clean up temp workdir before returning
        shutil.rmtree(workdir, ignore_errors=True)
        return jsonify({"result": "ERROR", "stage": "APPLY_CHANGES", "error": str(e)}), 400

    # 3) Initialize the snapshot in Batfish
    try:
        snapshot_name = _init_snapshot(workdir)
    except Exception as e:
        shutil.rmtree(workdir, ignore_errors=True)
        return jsonify({"result": "ERROR", "stage": "INIT_SNAPSHOT", "error": str(e)}), 400

    # 4) Run CP, TP, Reachability
    try:
        cp_summary = _run_cp(changed_devices)
        tp_summary = _run_tp()
        reach_summary = _run_reach(reach_paths)
    except Exception as e:
        shutil.rmtree(workdir, ignore_errors=True)
        return jsonify({"result": "ERROR", "stage": "VERIFY", "error": str(e)}), 500

    # 5) Summarize + cleanup
    resp = {
        "result": "OK",
        "snapshot": snapshot_name,
        "summary": {
            "CP": cp_summary,
            "TP": tp_summary,
            "REACH": reach_summary
        }
    }

    # Remove the temp snapshot folder. If you prefer keeping it for debugging, comment this out.
    shutil.rmtree(workdir, ignore_errors=True)

    return jsonify(resp), 200

# -------------------------
# Health
# -------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "up"}), 200

# -------------------------
# Main
# -------------------------
if _name_ == "_main_":
    # Ensure network exists (no-op if already created)
    _ensure_network()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))