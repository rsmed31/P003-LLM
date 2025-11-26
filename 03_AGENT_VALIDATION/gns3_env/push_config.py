"""Push an FRR config to a router node via telnet and vtysh."""

from __future__ import annotations

import argparse
import logging
import os
import time
import telnetlib
from typing import Iterable

from dotenv import load_dotenv

from gns3_api import get_console_port, get_project

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOG = logging.getLogger(__name__)
TELNET_TIMEOUT = float(os.getenv("TELNET_TIMEOUT", "5"))


def _send_lines(tn: telnetlib.Telnet, lines: Iterable[str], delay: float = 0.1) -> None:
    for line in lines:
        tn.write(line.encode("ascii") + b"\n")
        time.sleep(delay)


def push_config(node: str, filepath: str, project_name: str, force: bool = False) -> None:
    project = get_project(project_name)
    if not project:
        raise SystemExit(f"Project {project_name} not found.")

    console_port = get_console_port(project["project_id"], node)
    LOG.info("Pushing config %s to %s (console %s)", filepath, node, console_port)

    if not force:
        confirm = input(f"Send {len(lines)} lines to {node} on {project_name}? [y/N]: ").strip().lower()
        if confirm not in {"y", "yes"}:
            LOG.info("Aborted by user.")
            return

    with open(filepath, "r", encoding="utf-8") as handle:
        lines = [line.strip("\n") for line in handle if line.strip() and not line.strip().startswith("!")]

    tn = telnetlib.Telnet("127.0.0.1", console_port, timeout=TELNET_TIMEOUT)
    tn.read_until(b"#", timeout=TELNET_TIMEOUT)
    _send_lines(tn, ["vtysh"])
    tn.read_until(b">", timeout=TELNET_TIMEOUT)
    _send_lines(tn, ["configure terminal"])
    tn.read_until(b"(config)", timeout=TELNET_TIMEOUT)
    _send_lines(tn, lines)
    _send_lines(tn, ["write", "exit", "exit"])
    tn.close()
    LOG.info("Config push completed for %s", node)


def main() -> None:
    parser = argparse.ArgumentParser(description="Push FRR config to a router node.")
    parser.add_argument("--node", required=True, help="Router node name (e.g. R1)")
    parser.add_argument("--file", required=True, help="Path to frr.conf to push")
    parser.add_argument(
        "--project",
        default=os.getenv("GNS3_PROJECT", "P003_OSPF_GNS3"),
        help="Project name (default: env GNS3_PROJECT or P003_OSPF_GNS3)",
    )
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt.")
    args = parser.parse_args()
    push_config(args.node, args.file, args.project, force=args.yes)


if __name__ == "__main__":
    main()
