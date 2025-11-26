"""Validator backend using GNS3 instead of Batfish."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from typing import Any, Dict

from dotenv import load_dotenv

from gns3_api import get_console_port, get_project, start_all_nodes
from vpcs_telnet import run_vpcs_cmd

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOG = logging.getLogger(__name__)

H1_NAME = os.getenv("GNS3_NODE_H1", "H1")
H2_IP = os.getenv("GNS3_H2_IP", "192.168.3.10")
GNS3_PROJECT = os.getenv("GNS3_PROJECT", "P003_OSPF_GNS3")


def _project_required() -> Dict[str, Any]:
    project = get_project(GNS3_PROJECT)
    if not project:
        return {"status": "FAIL", "reason": "PROJECT_NOT_FOUND"}
    return project


def validate_policy(config_text: str | None = None) -> Dict[str, Any]:
    """Validate reachability H1->H2 using GNS3 VPCS ping."""
    project = _project_required()
    if project.get("status") == "FAIL":
        return project  # type: ignore[return-value]

    project_id = project["project_id"]
    try:
        start_all_nodes(project_id)
    except Exception as exc:  # pragma: no cover - safety net
        LOG.error("Unable to start nodes: %s", exc)
        return {"status": "FAIL", "reason": "START_NODES_FAILED", "error": str(exc)}

    try:
        console_port = get_console_port(project_id, H1_NAME)
    except Exception as exc:
        LOG.error("Unable to get console for %s: %s", H1_NAME, exc)
        return {"status": "FAIL", "reason": "NODE_H1_NOT_FOUND", "error": str(exc)}

    cmd = f"ping {H2_IP}"
    try:
        output = run_vpcs_cmd(
            "127.0.0.1",
            console_port,
            cmd,
            wait=float(os.getenv("PING_WAIT", "2")),
        )
    except Exception as exc:  # pragma: no cover - runtime telnet errors
        LOG.error("Telnet to %s failed: %s", H1_NAME, exc)
        return {"status": "FAIL", "reason": "CONSOLE_ERROR", "error": str(exc)}

    success = bool(re.search(r"(icmp_seq|bytes from)", output, flags=re.IGNORECASE))
    details = {"cmd": cmd, "raw": output}
    if success:
        return {"status": "PASS", "details": details}
    return {"status": "FAIL", "reason": "REACHABILITY", "details": details}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate reachability H1 -> H2 in GNS3.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs.")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    result = validate_policy()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
