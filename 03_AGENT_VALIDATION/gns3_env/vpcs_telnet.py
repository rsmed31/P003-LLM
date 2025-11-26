import telnetlib
import os
import time
from dotenv import load_dotenv

load_dotenv()

def run_vpcs_cmd(host: str, port: int, cmd: str, wait: float = 2.0) -> str:
    """Ouvre un telnet vers VPCS, exécute cmd, retourne la sortie brute."""
    tn = telnetlib.Telnet(host, port, timeout=5)
    tn.read_until(b"> ")
    tn.write((cmd.strip() + "\n").encode())
    time.sleep(wait)
    out = tn.read_very_eager().decode(errors="ignore")
    tn.write(b"exit\n")
    return out