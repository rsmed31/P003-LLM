import os
import json
from dotenv import load_dotenv
from gns3_api import get_project, get_nodes

load_dotenv()

OUT_PATH = "topology/topology.json"

def main():
    project = get_project()
    if not project:
        raise SystemExit("Projet GNS3 introuvable (check .env).")

    pid = project["project_id"]
    nodes = get_nodes(pid)

    data = {
        "project_id": pid,
        "project_name": project["name"],
        "nodes": [
            {"name": n["name"], "node_id": n["node_id"], "console": n.get("console")}
            for n in nodes
        ],
    }

    os.makedirs("topology", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[OK] Topology exported → {OUT_PATH}")

if __name__ == "__main__":
    main()
