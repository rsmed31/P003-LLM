"""Thin helper over the GNS3 REST API used by the validator tools."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth

load_dotenv()

LOG = logging.getLogger(__name__)

GNS3_URL = os.getenv("GNS3_URL", "http://localhost:3080/v2").rstrip("/")
GNS3_USERNAME = os.getenv("GNS3_USERNAME")
GNS3_PASSWORD = os.getenv("GNS3_PASSWORD")


def _auth() -> Optional[HTTPBasicAuth]:
    if GNS3_USERNAME and GNS3_PASSWORD:
        return HTTPBasicAuth(GNS3_USERNAME, GNS3_PASSWORD)
    return None


def _request(method: str, path: str, **kwargs: Any) -> requests.Response:
    url = f"{GNS3_URL}/{path.lstrip('/')}"
    LOG.debug("GNS3 %s %s", method, url)
    timeout = kwargs.pop("timeout", 10)
    resp = requests.request(method, url, auth=_auth(), timeout=timeout, **kwargs)
    if resp.status_code >= 400:
        LOG.error("GNS3 API error %s on %s: %s", resp.status_code, url, resp.text)
        resp.raise_for_status()
    return resp


def get_projects() -> List[Dict[str, Any]]:
    """Return the list of projects visible to the API."""
    resp = _request("GET", "/projects")
    projects = resp.json()
    LOG.info("Discovered %d GNS3 projects", len(projects))
    return projects


def get_project(project_name: str) -> Optional[Dict[str, Any]]:
    """Return the project dict matching the name, or None if not found."""
    projects = get_projects()
    for project in projects:
        if project.get("name") == project_name:
            LOG.info("Using project %s (%s)", project_name, project.get("project_id"))
            return project
    LOG.error("Project %s not found", project_name)
    return None


def get_nodes(project_id: str) -> List[Dict[str, Any]]:
    """Return the list of nodes for a project."""
    resp = _request("GET", f"/projects/{project_id}/nodes")
    nodes = resp.json()
    LOG.info("Project %s has %d nodes", project_id, len(nodes))
    return nodes


def get_node(project_id: str, node_name: str) -> Optional[Dict[str, Any]]:
    """Return a node matching the given name."""
    nodes = get_nodes(project_id)
    for node in nodes:
        if node.get("name") == node_name:
            return node
    LOG.error("Node %s not found in project %s", node_name, project_id)
    return None


def get_console_port(project_id: str, node_name: str) -> int:
    """Return the console port for a node, raising if not present."""
    node = get_node(project_id, node_name)
    if not node:
        raise ValueError(f"Node {node_name} not found in project {project_id}")
    console_port = node.get("console")
    if console_port is None:
        raise ValueError(f"Node {node_name} has no console port")
    return int(console_port)


def start_all_nodes(project_id: str) -> bool:
    """Start all nodes in the project (idempotent server-side)."""
    resp = _request("POST", f"/projects/{project_id}/nodes/start")
    LOG.info("Start all nodes response code %s", resp.status_code)
    return resp.status_code < 400
