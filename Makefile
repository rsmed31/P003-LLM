.PHONY: help install install-t1 install-t2 install-t3 run-all run-t1 run-t2 run-t3 stop config clean test

# Default ports
T1_PORT ?= 8000
T2_PORT ?= 8001
T3_PORT ?= 5000

# API URLs (can be overridden)
T1_URL ?= http://localhost:$(T1_PORT)
T2_URL ?= http://localhost:$(T2_URL)
T3_URL ?= http://localhost:$(T3_PORT)

# Python and venv paths
PYTHON := python3
VENV_T1 := 01_DATA_ASSETS/.venv
VENV_T2 := 02_LLM_INFERENCE_API/.venv
VENV_T3 := 03_AGENT_VALIDATION/langchain_agent/.venv
VENV_T3_BATFISH := 03_AGENT_VALIDATION/batfish/.venv

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

install: install-t1 install-t2 install-t3 ## Install all project dependencies in virtual environments

install-t1: ## Install Team 1 (Data/RAG) dependencies
	@echo "$(GREEN)Installing Team 1 dependencies in virtual environment...$(NC)"
	@cd 01_DATA_ASSETS && \
		$(PYTHON) -m venv .venv && \
		.venv/bin/pip install --upgrade pip && \
		.venv/bin/pip install -r requirements.txt
	@echo "$(GREEN)✓ Team 1 dependencies installed in $(VENV_T1)$(NC)"

install-t2: ## Install Team 2 (LLM Inference) dependencies
	@echo "$(GREEN)Installing Team 2 dependencies in virtual environment...$(NC)"
	@cd 02_LLM_INFERENCE_API && \
		$(PYTHON) -m venv .venv && \
		.venv/bin/pip install --upgrade pip && \
		.venv/bin/pip install -r requirements.txt
	@echo "$(GREEN)✓ Team 2 dependencies installed in $(VENV_T2)$(NC)"

install-t3: ## Install Team 3 (Agent/Validation) dependencies
	@echo "$(GREEN)Installing Team 3 dependencies in virtual environments...$(NC)"
	@cd 03_AGENT_VALIDATION/langchain_agent && \
		$(PYTHON) -m venv .venv && \
		.venv/bin/pip install --upgrade pip && \
		.venv/bin/pip install -r requirements.txt
	@cd 03_AGENT_VALIDATION/batfish && \
		$(PYTHON) -m venv .venv && \
		.venv/bin/pip install --upgrade pip && \
		.venv/bin/pip install -r requirements.txt
	@echo "$(GREEN)✓ Team 3 dependencies installed$(NC)"

config: ## Configure API URLs for all services
	@echo "$(BLUE)Configuring API URLs...$(NC)"
	@echo "$(YELLOW)Team 1 URL: $(T1_URL)$(NC)"
	@echo "$(YELLOW)Team 2 URL: $(T2_URL)$(NC)"
	@echo "$(YELLOW)Team 3 URL: $(T3_URL)$(NC)"
	@cd 03_AGENT_VALIDATION/langchain_agent && \
		.venv/bin/python -c "import json; \
		cfg = json.load(open('config.json')); \
		cfg['T1_BASE_URL'] = '$(T1_URL)'; \
		cfg['T2_BASE_URL'] = '$(T2_URL)'; \
		cfg['T3_BASE_URL'] = '$(T3_URL)'; \
		json.dump(cfg, open('config.json', 'w'), indent=2)"
	@echo "$(GREEN)✓ Configuration updated in 03_AGENT_VALIDATION/langchain_agent/config.json$(NC)"

config-interactive: ## Interactive configuration wizard
	@echo "$(BLUE)=== Configuration Wizard ===$(NC)"
	@read -p "Team 1 (Data/RAG) URL [$(T1_URL)]: " t1; \
	read -p "Team 2 (LLM) URL [$(T2_URL)]: " t2; \
	read -p "Team 3 (Validation) URL [$(T3_URL)]: " t3; \
	read -p "GROQ API Key (optional): " groq; \
	read -p "Gemini API Key (optional): " gemini; \
	cd 03_AGENT_VALIDATION/langchain_agent && \
	.venv/bin/python -c "import json; \
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

run-all: ## Run all services (T1, T2, T3) in background
	@echo "$(GREEN)Starting all services...$(NC)"
	@make run-t1 &
	@sleep 2
	@make run-t2 &
	@sleep 2
	@make run-t3 &
	@echo "$(GREEN)✓ All services started$(NC)"
	@echo "$(YELLOW)Access points:$(NC)"
	@echo "  T1 (Data/RAG):  http://localhost:$(T1_PORT)"
	@echo "  T2 (LLM):       http://localhost:$(T2_PORT)/docs"
	@echo "  T3 (Agent):     http://localhost:$(T3_PORT)"

run-t1: ## Run Team 1 (Data/RAG service)
	@echo "$(GREEN)Starting Team 1 on port $(T1_PORT)...$(NC)"
	@cd 01_DATA_ASSETS && \
		.venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port $(T1_PORT) --reload

run-t1-docker: ## Run Team 1 with Docker Compose
	@echo "$(GREEN)Starting Team 1 (PostgreSQL + API) with Docker...$(NC)"
	@cd 01_DATA_ASSETS && docker compose up --build -d
	@echo "$(GREEN)✓ Team 1 running in Docker$(NC)"

run-t2: ## Run Team 2 (LLM Inference API)
	@echo "$(GREEN)Starting Team 2 on port $(T2_PORT)...$(NC)"
	@cd 02_LLM_INFERENCE_API && \
		.venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port $(T2_PORT) --reload

run-t3: run-t3-batfish run-t3-agent ## Run Team 3 (Batfish + Agent)

run-t3-batfish: ## Run Team 3 Batfish validation service with Docker
	@echo "$(GREEN)Starting Batfish validation service...$(NC)"
	@cd 03_AGENT_VALIDATION/batfish && docker compose up -d --build
	@echo "$(GREEN)✓ Batfish + Validator running on port 5000$(NC)"

run-t3-agent: ## Run Team 3 Agent Orchestrator
	@echo "$(GREEN)Starting Team 3 Agent on port $(T3_PORT)...$(NC)"
	@cd 03_AGENT_VALIDATION/langchain_agent && \
		.venv/bin/python -m uvicorn agent_service:app --host 0.0.0.0 --port $(T3_PORT) --reload

stop: ## Stop all running services
	@echo "$(RED)Stopping all services...$(NC)"
	@pkill -f "uvicorn.*app:app" || true
	@pkill -f "uvicorn.*agent_service:app" || true
	@cd 01_DATA_ASSETS && docker compose down 2>/dev/null || true
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

clean-venv: ## Remove all virtual environments
	@echo "$(YELLOW)Removing virtual environments...$(NC)"
	@rm -rf $(VENV_T1) $(VENV_T2) $(VENV_T3) $(VENV_T3_BATFISH) 2>/dev/null || true
	@echo "$(GREEN)✓ Virtual environments removed$(NC)"

clean-t3: ## Clean Team 3 Docker volumes
	@echo "$(YELLOW)Cleaning Team 3 Batfish volumes...$(NC)"
	@cd 03_AGENT_VALIDATION/batfish && docker compose down -v
	@echo "$(GREEN)✓ Team 3 cleanup complete$(NC)"

test-t2-t3: ## Test Team 2 → Team 3 integration
	@echo "$(BLUE)Testing T2 → T3 pipeline...$(NC)"
	@cd 03_AGENT_VALIDATION/langchain_agent && \
		.venv/bin/python agent_service.py --query "Configure OSPF on 3 routers" --model gemini

test-full: ## Test full pipeline (T1 → T2 → T3)
	@echo "$(BLUE)Testing full pipeline...$(NC)"
	@cd 03_AGENT_VALIDATION/langchain_agent && \
		.venv/bin/python agent_service.py --query "What is OSPF?"

health-check: ## Check if all services are running
	@echo "$(BLUE)Checking service health...$(NC)"
	@curl -s http://localhost:$(T1_PORT)/health && echo "$(GREEN)✓ T1 is running$(NC)" || echo "$(RED)✗ T1 is down$(NC)"
	@curl -s http://localhost:$(T2_PORT)/health && echo "$(GREEN)✓ T2 is running$(NC)" || echo "$(RED)✗ T2 is down$(NC)"
	@curl -s http://localhost:5000/health && echo "$(GREEN)✓ T3 Validator is running$(NC)" || echo "$(RED)✗ T3 Validator is down$(NC)"
	@curl -s http://localhost:$(T3_PORT)/docs && echo "$(GREEN)✓ T3 Agent is running$(NC)" || echo "$(RED)✗ T3 Agent is down$(NC)"

logs-t1: ## Show Team 1 logs (if running via Docker)
	@cd 01_DATA_ASSETS && docker compose logs -f

logs-t3: ## Show Team 3 Batfish logs
	@cd 03_AGENT_VALIDATION/batfish && docker compose logs -f

logs-t3-validator: ## Show validator service logs only
	@cd 03_AGENT_VALIDATION/batfish && docker compose logs -f validator

logs-t3-batfish: ## Show batfish service logs only
	@cd 03_AGENT_VALIDATION/batfish && docker compose logs -f batfish

status: ## Show status of all services
	@echo "$(BLUE)=== Service Status ===$(NC)"
	@ps aux | grep -E "uvicorn.*(app|agent_service)" | grep -v grep || echo "$(YELLOW)No services running$(NC)"
	@echo ""
	@echo "$(BLUE)=== Virtual Environments ===$(NC)"
	@test -d $(VENV_T1) && echo "$(GREEN)✓ T1 venv exists$(NC)" || echo "$(RED)✗ T1 venv missing$(NC)"
	@test -d $(VENV_T2) && echo "$(GREEN)✓ T2 venv exists$(NC)" || echo "$(RED)✗ T2 venv missing$(NC)"
	@test -d $(VENV_T3) && echo "$(GREEN)✓ T3 venv exists$(NC)" || echo "$(RED)✗ T3 venv missing$(NC)"
	@echo ""
	@echo "$(BLUE)=== Docker Containers ===$(NC)"
	@cd 01_DATA_ASSETS && docker compose ps 2>/dev/null || true
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
activate-t1: ## Show command to activate Team 1 venv
	@echo "Run: source $(VENV_T1)/bin/activate"

activate-t2: ## Show command to activate Team 2 venv
	@echo "Run: source $(VENV_T2)/bin/activate"

activate-t3: ## Show command to activate Team 3 venv
	@echo "Run: source $(VENV_T3)/bin/activate"
