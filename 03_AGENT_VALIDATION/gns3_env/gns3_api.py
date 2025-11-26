import os
import requests
from dotenv import load_dotenv

load_dotenv()

GNS3_URL = os.getenv("GNS3_URL", "http://localhost:3080/v2")
PROJECT_NAME = os.getenv("GNS3_PROJECT", "P003_OSPF_GNS3")

def _url(path: str) -> str:
    return f"{GNS3_URL}{path}"

def get_projects():
    r = requests.get(_url("/projects"))
    r.raise_for_status()
    return r.json()

def get_project(project_name: str = PROJECT_NAME):
    for p in get_projects():
        if p.get("name") == project_name:
            return p
    return None

def get_nodes(project_id: str):
    r = requests.get(_url(f"/projects/{project_id}/nodes"))
    r.raise_for_status()
    return r.json()

def get_node(project_id: str, node_name: str):
    for n in get_nodes(project_id):
        if n.get("name") == node_name:
            return n
    return None

def get_console_port(project_id: str, node_name: str) -> int:
    node = get_node(project_id, node_name)
    if not node:
        raise RuntimeError(f"Node {node_name} not found.")
    return int(node.get("console"))

def start_all_nodes(project_id: str):
    for n in get_nodes(project_id):
        requests.post(_url(f"/projects/{project_id}/nodes/{n['node_id']}/start"))
    print("[OK] All nodes started.")