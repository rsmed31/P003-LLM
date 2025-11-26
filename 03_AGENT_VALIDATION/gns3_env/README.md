GNS3 backend pour la Team 3 (Validation & Agent). Il offre le même contrat qu’avec Batfish via `validate_policy` et se sélectionne avec `VALIDATION_BACKEND=gns3`.

## Pré-requis
- GNS3 Desktop + VM ou support Docker avec l’API locale dispo sur `http://localhost:3080/v2`
- Projet importé : `P003_OSPF_GNS3`
- Topo MVP : H1 — R1 — R2 — R3 — H2 avec H2 = 192.168.3.10/24

## Quick start
```bash
cd P003-LLM/03_AGENT_VALIDATION/gns3_env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python discover_topology.py
python validator_gns3.py
```

## Intégration agent (switch Batfish → GNS3)
```bash
export VALIDATION_BACKEND=gns3
```
```python
backend = os.getenv("VALIDATION_BACKEND", "batfish")
if backend == "gns3":
    from 03_AGENT_VALIDATION.gns3_env.validator_gns3 import validate_policy
else:
    from 03_AGENT_VALIDATION.batfish_env.validator import validate_policy
```

## Scripts disponibles
- `discover_topology.py` : exporte `topology/topology.json` (project_id, nodes, consoles)
- `validator_gns3.py` : ping H1 → 192.168.3.10 depuis la topo, retourne JSON PASS/FAIL
- `exec_console.py --node H1 --cmd "ping 192.168.3.10"` : envoie une commande VPCS ponctuelle
- `push_config.py --node R1 --file topology/configs/R1-frr.conf` : pousse un frr.conf via vtysh

## Exemple de réponses
- PASS : `{"status":"PASS","details":{"cmd":"ping 192.168.3.10","raw":"..."}}`
- FAIL reachability : `{"status":"FAIL","reason":"REACHABILITY","details":{...}}`
- FAIL structure : `{"status":"FAIL","reason":"PROJECT_NOT_FOUND"}` …

## Troubleshooting
- API GNS3 : vérifier `http://localhost:3080/v2` (ou ajouter `GNS3_USERNAME/PASSWORD` si auth)
- Noms des nœuds : `GNS3_NODE_H1` / `GNS3_NODE_H2` doivent correspondre dans le projet (H1/H2)
- Ports console absents : relancer les nœuds ou régénérer la topo (`discover_topology.py`)

## Commandes de test
```bash
# 1) Setup
cd P003-LLM/03_AGENT_VALIDATION/gns3_env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# 2) Vérifier l’API GNS3
python discover_topology.py

# 3) Valider (H1 → H2)
python validator_gns3.py

# 4) Exécuter une commande ponctuelle
python exec_console.py --node H1 --cmd "ping 192.168.3.10"

# 5) (Option) Pousser une config FRR
python push_config.py --node R1 --file topology/configs/R1-frr.conf
```
