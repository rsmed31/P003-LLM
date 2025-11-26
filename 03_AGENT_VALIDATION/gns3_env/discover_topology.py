"""Discover the GNS3 project topology and export it to topology/topology.json."""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Dict, List

from dotenv import load_dotenv

from gns3_api import get_nodes, get_project

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOG = logging.getLogger(__name__)

OUTPUT_PATH = os.path.join("topology", "topology.json")


def _export_topology(project: Dict[str, str], nodes: List[Dict[str, str]]) -> str:
    os.makedirs("topology", exist_ok=True)
    data = {
        "project_id": project["project_id"],
        "project_name": project["name"],
        "nodes": [
            {"name": node.get("name"), "node_id": node.get("node_id"), "console": node.get("console")}
            for node in nodes
        ],
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    return OUTPUT_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GNS3 topology to topology/topology.json")
    parser.add_argument(
        "--project",
        default=os.getenv("GNS3_PROJECT", "P003_OSPF_GNS3"),
        help="GNS3 project name (default: env GNS3_PROJECT or P003_OSPF_GNS3)",
    )
    args = parser.parse_args()

    project = get_project(args.project)
    if not project:
        raise SystemExit(f"Project {args.project} not found. Check .env.")

    project_id = project["project_id"]
    nodes = get_nodes(project_id)
    output_path = _export_topology(project, nodes)
    print(output_path)


if __name__ == "__main__":
    main()
