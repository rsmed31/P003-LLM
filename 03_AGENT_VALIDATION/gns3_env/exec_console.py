"""Run a one-off command on a VPCS node console."""

from __future__ import annotations

import argparse
import logging
import os

from dotenv import load_dotenv

from gns3_api import get_console_port, get_project
from vpcs_telnet import run_vpcs_cmd

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOG = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute a command on a VPCS console.")
    parser.add_argument("--node", required=True, help="Node name (e.g. H1)")
    parser.add_argument("--cmd", required=True, help="Command to send (e.g. ping 192.168.3.10)")
    parser.add_argument(
        "--project",
        default=os.getenv("GNS3_PROJECT", "P003_OSPF_GNS3"),
        help="Project name (default: env GNS3_PROJECT or P003_OSPF_GNS3)",
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=float(os.getenv("PING_WAIT", "2")),
        help="Seconds to wait for output after sending the command",
    )
    args = parser.parse_args()

    project = get_project(args.project)
    if not project:
        raise SystemExit(f"Project {args.project} not found.")

    console_port = get_console_port(project["project_id"], args.node)
    output = run_vpcs_cmd("127.0.0.1", console_port, args.cmd, wait=args.wait)
    print(output.strip())


if __name__ == "__main__":
    main()
