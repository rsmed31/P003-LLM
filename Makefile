.PHONY: help install install-t1 install-t2 install-t3 run-all run-t1 run-t2 run-t3 stop config clean test

# Default ports
T1_PORT ?= 8000
T2_PORT ?= 8001
T3_PORT ?= 5000

# API URLs (can be overridden)
T1_URL ?= http://localhost:$(T1_PORT)
T2_URL ?= http://localhost:$(T2_PORT)
T3_URL ?= http://localhost:$(T3_PORT)

# Replace per-team venv variables with one global venv
VENV := .venv
PYTHON ?= python

# Cross-platform venv python path + shell adaptations
ifeq ($(OS),Windows_NT)
	VENV_BIN := Scripts
	VENV_PY := $(VENV)\Scripts\python.exe
	ACTIVATE_HINT := $(VENV)\Scripts\activate
	# Ensure we use cmd to allow Windows conditionals
	SHELL := cmd
	SHELLFLAGS := /C
else
	VENV_BIN := bin
	VENV_PY := $(VENV)/bin/python
	ACTIVATE_HINT := source $(VENV)/bin/activate
endif

# Cross-platform sleep helper
ifeq ($(OS),Windows_NT)
	SLEEP = timeout /t
	CLEAR = cls
else
	SLEEP = sleep
	CLEAR = clear
endif

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)P003-LLM Project Manager$(NC)"
	@echo ""
	@echo "$(GREEN)Available targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Configuration:$(NC)"
	@echo "  T1_PORT=$(T1_PORT)"
	@echo "  T2_PORT=$(T2_PORT)"
	@echo "  T3_PORT=$(T3_PORT)"
	@echo ""
	@echo "$(GREEN)Example usage:$(NC)"
	@echo "  make install              # Install all dependencies in venvs"
	@echo "  make config               # Configure API URLs"
	@echo "  make run-all              # Run all services"
	@echo "  make T1_PORT=9000 run-t1  # Run T1 on custom port"

# ========== Global venv creation (fixed) ==========
# Windows and *nix have different existence checks
ifeq ($(OS),Windows_NT)
install-venv: ## Create global virtual environment (Windows)
	@echo "$(GREEN)Creating/refreshing global virtual environment...$(NC)"
	@if not exist "$(VENV)" ($(PYTHON) -m venv $(VENV))
	@$(VENV_PY) -m pip install --upgrade pip
	@echo "$(GREEN)✓ Global venv ready at $(VENV) (activate: $(ACTIVATE_HINT))$(NC)"
else
install-venv: ## Create global virtual environment (*nix)
	@echo "$(GREEN)Creating/refreshing global virtual environment...$(NC)"
	@if [ ! -d "$(VENV)" ]; then \
		$(PYTHON) -m venv $(VENV); \
	fi
	@$(VENV_PY) -m pip install --upgrade pip
	@echo "$(GREEN)✓ Global venv ready at $(VENV) (activate: $(ACTIVATE_HINT))$(NC)"
endif

# ========== Top-level install (T1 is Docker-only) ==========
install: install-venv install-t2 install-t3 ## Install all (T1 is Docker-only)

install-t1: ## Team 1 is Docker-only
	@echo "$(YELLOW)Team 1 uses Docker only – skipping Python installation$(NC)"
	@echo "$(GREEN)✓ Use: make run-t1$(NC)"

install-t2: install-venv ## Install Team 2 deps
	@echo "$(GREEN)Installing Team 2 dependencies in global venv...$(NC)"
	@$(VENV_PY) -m pip install -r 02_LLM_INFERENCE_API/requirements.txt
	@echo "$(GREEN)✓ Team 2 dependencies installed$(NC)"

install-t3: install-venv ## Install Team 3 deps
	@echo "$(GREEN)Installing Team 3 dependencies in global venv...$(NC)"
	@$(VENV_PY) -m pip install -r 03_AGENT_VALIDATION/langchain_agent/requirements.txt
	@$(VENV_PY) -m pip install -r 03_AGENT_VALIDATION/batfish/requirements.txt
	@echo "$(GREEN)✓ Team 3 dependencies installed$(NC)"

# ========== Config (use global venv python) ==========
config: install-venv ## Configure API URLs
	@echo "$(BLUE)Configuring API URLs...$(NC)"
	@$(VENV_PY) -c "import json,sys; \
	p='03_AGENT_VALIDATION/langchain_agent/config.json'; \
	cfg=json.load(open(p)); \
	cfg['T1_BASE_URL']='$(T1_URL)'; \
	cfg['T2_BASE_URL']='$(T2_URL)'; \
	cfg['T3_BASE_URL']='$(T3_URL)'; \
	json.dump(cfg,open(p,'w'),indent=2)"
	@echo "$(GREEN)✓ Updated config.json$(NC)"

config-interactive: ## Interactive configuration wizard
	@echo "$(BLUE)=== Configuration Wizard ===$(NC)"
	@read -p "Team 1 (Data/RAG) URL [$(T1_URL)]: " t1; \
	read -p "Team 2 (LLM) URL [$(T2_URL)]: " t2; \
	read -p "Team 3 (Validation) URL [$(T3_URL)]: " t3; \
	read -p "GROQ API Key (optional): " groq; \
	read -p "Gemini API Key (optional): " gemini; \
	cd 03_AGENT_VALIDATION/langchain_agent && \
	.venv/$(VENV_BIN)/python -c "import json; \
	cfg = json.load(open('config.json')); \
	cfg['T1_BASE_URL'] = '$${t1:-$(T1_URL)}'; \
	cfg['T2_BASE_URL'] = '$${t2:-$(T2_URL)}'; \
	cfg['T3_BASE_URL'] = '$${t3:-$(T3_URL)}'; \
	if '$${groq}': cfg['GROQ_API_KEY'] = '$${groq}'; \
	json.dump(cfg, open('config.json', 'w'), indent=2)"; \
	if [ -n "$${gemini}" ]; then \
		cd ../../02_LLM_INFERENCE_API/models && \
		echo "GEMINI_API_KEY=$${gemini}" > keys.env; \
	fi
	@echo "$(GREEN)✓ Configuration complete$(NC)"

run-all: ## Run all services (sequential, abort on failure)
	@echo "Starting all services..."
	@$(MAKE) run-t1 || (echo "Team 1 failed; aborting."; exit 1)
	@$(SLEEP) 2 >nul 2>&1 || true
	@$(MAKE) run-t2 || (echo "Team 2 failed; aborting."; exit 1)
	@$(SLEEP) 2 >nul 2>&1 || true
	@$(MAKE) run-t3 || (echo "Team 3 failed; aborting."; exit 1)
	@echo "All services started."

run-t1: ## Run Team 1 (Docker)
ifeq ($(OS),Windows_NT)
	@echo "Starting Team 1 (Docker)..."
	@where docker >nul 2>&1 || (echo "ERROR: Docker is not installed. Install Docker Desktop from https://docker.com/products/docker-desktop"; exit 1)
	@docker ps >nul 2>&1 || (echo "ERROR: Docker Desktop is not running. Please start Docker Desktop first."; exit 1)
	@if not exist "01_DATA_ASSETS\postgres_api\docker-compose.yml" (echo "ERROR: Missing docker-compose.yml in 01_DATA_ASSETS\postgres_api"; exit 1)
	@cd 01_DATA_ASSETS\postgres_api && docker compose up --build -d
	@echo "Team 1 running (port $(T1_PORT))"
else
	@echo "Starting Team 1 (Docker)..."
	@command -v docker >/dev/null 2>&1 || { echo "ERROR: Docker is not installed."; exit 1; }
	@docker ps >/dev/null 2>&1 || { echo "ERROR: Docker daemon is not running."; exit 1; }
	@[ -f 01_DATA_ASSETS/postgres_api/docker-compose.yml ] || { echo "ERROR: Missing docker-compose.yml"; exit 1; }
	@cd 01_DATA_ASSETS/postgres_api && docker compose up --build -d
	@echo "Team 1 running (port $(T1_PORT))"
endif

run-t1-logs: ## Show Team 1 Docker logs
	@cd 01_DATA_ASSETS/postgres_api && docker compose logs -f

run-t2: ## Run Team 2 (LLM Inference API)
	@echo "Starting Team 2 on port $(T2_PORT)..."
	@cd 02_LLM_INFERENCE_API && \
		..\$(VENV)\$(VENV_BIN)\python -m uvicorn app:app --host 0.0.0.0 --port $(T2_PORT) --reload

run-t3: ## Run Team 3 (Batfish + Agent or interactive)
	@$(MAKE) run-t3-batfish
ifeq ($(T3_MODE),interactive)
	@$(MAKE) run-t3-interactive
else
	@$(MAKE) run-t3-agent
endif

run-t3-batfish: ## Run Team 3 Batfish validation service with Docker
	@echo "$(GREEN)Starting Batfish validation service...$(NC)"
	@cd 03_AGENT_VALIDATION/batfish && docker compose up -d --build
	@echo "$(GREEN)✓ Batfish + Validator running on port 5000$(NC)"

run-t3-agent: ## Run Team 3 Agent Orchestrator
	@echo "Starting Team 3 Agent on port $(T3_PORT)..."
	@cd 03_AGENT_VALIDATION\langchain_agent && \
		..\..\$(VENV)\$(VENV_BIN)\python -m uvicorn agent_service:app --host 0.0.0.0 --port $(T3_PORT) --reload

run-t3-interactive: ## Run Team 3 Agent Interactive CLI
	@echo "Starting Team 3 Agent (interactive CLI)..."
	@cd 03_AGENT_VALIDATION\langchain_agent && ..\..\$(VENV)\$(VENV_BIN)\python interactive.py

interactive: run-t3-interactive ## Shortcut for interactive CLI only

stop: ## Stop all running services
	@echo "$(RED)Stopping all services...$(NC)"
	@pkill -f "uvicorn.*app:app" || true
	@pkill -f "uvicorn.*agent_service:app" || true
	@cd 01_DATA_ASSETS/postgres_api && docker compose down 2>/dev/null || true
	@cd 03_AGENT_VALIDATION/batfish && docker compose down 2>/dev/null || true
	@echo "$(GREEN)✓ All services stopped$(NC)"

clean: clean-cache clean-venv ## Clean all cache and virtual environments

clean-cache: ## Clean Python cache and temporary files
	@echo "$(YELLOW)Cleaning cache files...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Cache cleanup complete$(NC)"

clean-venv: ## Remove global venv
	@echo "$(YELLOW)Removing global virtual environment...$(NC)"
ifeq ($(OS),Windows_NT)
	@if exist "$(VENV)" rmdir /S /Q "$(VENV)"
else
	@rm -rf "$(VENV)"
endif
	@echo "$(GREEN)✓ Global venv removed$(NC)"

clean-t3: ## Clean Team 3 Docker volumes
	@echo "$(YELLOW)Cleaning Team 3 Batfish volumes...$(NC)"
	@cd 03_AGENT_VALIDATION/batfish && docker compose down -v
	@echo "$(GREEN)✓ Team 3 cleanup complete$(NC)"

test-t2-t3: ## Test Team 2 → Team 3 integration (global venv)
	@echo "Testing T2 → T3 pipeline..."
	@cd 03_AGENT_VALIDATION\langchain_agent && ..\..\$(VENV)\$(VENV_BIN)\python agent_service.py --query "Configure OSPF on 3 routers" --model gemini

test-full: ## Test full pipeline (global venv)
	@echo "Testing full pipeline..."
	@cd 03_AGENT_VALIDATION\langchain_agent && ..\..\$(VENV)\$(VENV_BIN)\python agent_service.py --query "What is OSPF?"

health-check: ## Check if all services are running
	@echo "$(BLUE)Checking service health...$(NC)"
	@curl -s http://localhost:$(T1_PORT)/health && echo "$(GREEN)✓ T1 is running$(NC)" || echo "$(RED)✗ T1 is down$(NC)"
	@curl -s http://localhost:$(T2_PORT)/health && echo "$(GREEN)✓ T2 is running$(NC)" || echo "$(RED)✗ T2 is down$(NC)"
	@curl -s http://localhost:5000/health && echo "$(GREEN)✓ T3 Validator is running$(NC)" || echo "$(RED)✗ T3 Validator is down$(NC)"
	@curl -s http://localhost:$(T3_PORT)/docs && echo "$(GREEN)✓ T3 Agent is running$(NC)" || echo "$(RED)✗ T3 Agent is down$(NC)"

logs-t1: ## Show Team 1 logs (if running via Docker)
	@cd 01_DATA_ASSETS/postgres_api && docker compose logs -f

logs-t3: ## Show Team 3 Batfish logs
	@cd 03_AGENT_VALIDATION/batfish && docker compose logs -f

logs-t3-validator: ## Show validator service logs only
	@cd 03_AGENT_VALIDATION/batfish && docker compose logs -f validator

logs-t3-batfish: ## Show batfish service logs only
	@cd 03_AGENT_VALIDATION/batfish && docker compose logs -f batfish

status: ## Show status
	@echo "$(BLUE)=== Service Status ===$(NC)"
	@ps aux | grep -E "uvicorn.*(app|agent_service)" | grep -v grep || echo "$(YELLOW)No Python services running$(NC)"
	@echo ""
	@echo "$(BLUE)=== Virtual Environment ===$(NC)"
	@test -d $(VENV) && echo "$(GREEN)✓ Global venv exists at ./$(VENV)$(NC)" || echo "$(RED)✗ Global venv missing$(NC)"
	@echo ""
	@echo "$(BLUE)=== Docker Containers ===$(NC)"
	@cd 01_DATA_ASSETS/postgres_api && docker compose ps 2>/dev/null || true
	@cd 03_AGENT_VALIDATION/batfish && docker compose ps 2>/dev/null || true

setup: install config ## Complete setup (install + configure)
	@echo "$(GREEN)✓ Setup complete! Run 'make run-all' to start all services$(NC)"

dev: ## Development mode - run all services with auto-reload
	@echo "$(BLUE)Starting development environment...$(NC)"
	@make run-all

quick-test: ## Quick test query through the agent
	@echo "$(BLUE)Running quick test...$(NC)"
	@curl -X POST http://localhost:$(T3_PORT)/run_agent \
		-H "Content-Type: application/json" \
		-d '{"query":"Configure OSPF area 0 on router R1"}' | python -m json.tool

# Manual venv activation commands (for reference)
activate: ## Show command to activate global venv
	@echo "Windows: .venv\Scripts\activate"
	@echo "Linux/Mac: source .venv/bin/activate"