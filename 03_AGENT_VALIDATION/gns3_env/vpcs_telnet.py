"""Helpers to send commands to VPCS nodes over telnet."""

from __future__ import annotations

import logging
import os
import time
import telnetlib
from typing import Any

from dotenv import load_dotenv

load_dotenv()

LOG = logging.getLogger(__name__)
TELNET_TIMEOUT = float(os.getenv("TELNET_TIMEOUT", "5"))


def run_vpcs_cmd(host: str, port: int, cmd: str, wait: float = 2.0) -> str:
    """Open telnet, send a VPCS command, and return the raw output."""
    LOG.info("Connecting telnet %s:%s to run '%s'", host, port, cmd)
    tn: Any = telnetlib.Telnet(host, port, timeout=TELNET_TIMEOUT)
    tn.read_until(b">", timeout=TELNET_TIMEOUT)
    tn.write(cmd.encode("ascii") + b"\n")
    time.sleep(wait)
    output = tn.read_very_eager().decode("utf-8", errors="ignore")
    tn.close()
    LOG.debug("Telnet raw output: %s", output)
    return output
