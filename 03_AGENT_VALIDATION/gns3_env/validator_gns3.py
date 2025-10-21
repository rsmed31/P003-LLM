import os
import re
import json
from dotenv import load_dotenv
from gns3_api import get_project, get_console_port, start_all_nodes
from vpcs_telnet import run_vpcs_cmd

load_dotenv()

H1 = os.getenv("GNS3_NODE_H1", "H1")
H2_IP = "192.168.3.10"  # cible MVP

def validate_policy(config_text: str = None) -> dict:
    """Validation MVP: ping H2 depuis H1. Retourne PASS/FAIL + détails."""
    project = get_project()
    if not project:
        return {"status": "FAIL", "reason": "PROJECT_NOT_FOUND"}

    pid = project["project_id"]
    start_all_nodes(pid)

    try:
        port_h1 = get_console_port(pid, H1)
    except Exception as e:
        return {"status": "FAIL", "reason": "NODE_H1_NOT_FOUND", "error": str(e)}

    out = run_vpcs_cmd("127.0.0.1", port_h1, f"ping {H2_IP}")
    success = bool(re.search(r"(icmp_seq|bytes from)", out, flags=re.IGNORECASE))

    if success:
        return {"status": "PASS", "details": {"cmd": f"ping {H2_IP}", "raw": out}}
    return {"status": "FAIL", "reason": "REACHABILITY", "details": {"cmd": f"ping {H2_IP}", "raw": out}}

if __name__ == "__main__":
    print(json.dumps(validate_policy(), indent=2))
