# P003-LLM: Network Configuration Generation System

Multi-team LLM-powered network configuration generation and validation pipeline.

---

## 📋 Prerequisites

### Required Software

**1. Python 3.9+**
- Minimum version: 3.9
- Recommended: 3.11 or 3.12

**2. Docker Desktop**
- **Windows**: https://docs.docker.com/desktop/install/windows-install/
- **macOS**: https://docs.docker.com/desktop/install/mac-install/
- **Linux**: https://docs.docker.com/desktop/install/linux-install/
- Verify: `docker --version` and `docker-compose --version`
- Required for Team 1 (PostgreSQL) and Team 3 (Batfish)

**3. Make (Build Tool)**
- **Windows**: 
  - Check Install_Make_Windows.md
- **macOS**: Pre-installed with Xcode Command Line Tools
  - If missing: `xcode-select --install`
- **Linux**: Pre-installed or `sudo apt install build-essential`
- Verify: `make --version`

### API Keys Required

**Groq API Key** (Team 3 - LLM Verdict Synthesis)
- Free tier available: https://console.groq.com/keys
- Sign up → Create API key → Copy to clipboard
- Configure in: `03_AGENT_VALIDATION/langchain_agent/config.json`
- Model used: `llama-3.3-70b-versatile` (70B params)

**Gemini API Key** (Team 2 - Config Generation)
- Free tier available: https://aistudio.google.com/app/apikey
- Sign in with Google → Create API key
- Configure in: `02_LLM_INFERENCE_API/models/keys.env`
- Model used: `gemini-2.0-flash` (multimodal)

**Ollama (Optional - Local Llama)** (Team 2 - Model)
- Download: https://ollama.com/download
- Install and run: `ollama serve`
- Pull model: `ollama pull llama3.1:8b`
- Configure in: `02_LLM_INFERENCE_API/models/keys.env`
- Default endpoint: `http://localhost:11434/api/generate`

### System Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8 GB
- Disk: 10 GB free space
- OS: Windows 10/11, macOS 10.15+, Ubuntu 20.04+

**Recommended:**
- CPU: 8+ cores
- RAM: 16 GB
- Disk: 20 GB free space (for Docker images and Batfish snapshots)
- OS: Windows 11, macOS 13+, Ubuntu 22.04+
---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
make install          # Creates global .venv/ and installs all dependencies
```

### 2. Configure API URLs & Keys
```bash
make config            # Auto-configure with localhost defaults
# OR
make config-interactive  # Interactive wizard
```

**Manual Configuration:**
Edit API keys in:
- `03_AGENT_VALIDATION/langchain_agent/config.json` - Add Groq API key
- `02_LLM_INFERENCE_API/models/keys.env` - Add Gemini/Ollama keys

### 3. Start All Services
```bash
make run-all          # Starts Team 1 (Docker), Team 2, Team 3
```

**Note:** Docker Desktop must be running before starting services.

### 4. Interactive Mode (Recommended)
```bash
make interactive      # Launch rich CLI with real-time visualization
```

**Interactive Features:**
- 🎨 Real-time pipeline visualization
- 💬 Chat-like interface with step-by-step progress
- 🔄 Live configuration preview
- ⚡ Runtime model/RAG switching (`model gemini`, `rag off`)
- 📊 Validation results with retry options

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Team 3: Agent Orchestrator               │
│                   (03_AGENT_VALIDATION)                      │
│  • LangChain LCEL Pipeline                                  │
│  • Groq LLM for verdict synthesis                           │
│  • Coordinates all teams + interactive CLI                  │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│   Team 1: RAG   │  │ Team 2: LLM Gen  │  │ Team 3: Validate │
│ (01_DATA_ASSETS)│  │(02_LLM_INFERENCE)│  │  (Simulation)    │
│                 │  │                  │  │                  │
│ • PostgreSQL    │  │ • Gemini/Llama   │  │ • Batfish Engine │
│ • Q&A Lookup    │  │ • RAG on/off     │  │ • Intent checks  │
│ • Write-back    │  │ • Multi-model    │  │ • PASS/FAIL      │
└─────────────────┘  └──────────────────┘  └──────────────────┘
```

### Technology Stack

**Team 1 - Data & RAG:**
- PostgreSQL 15 - https://www.postgresql.org/
- pgvector extension - https://github.com/pgvector/pgvector
- FastAPI - https://fastapi.tiangolo.com/
- Docker Compose - https://docs.docker.com/compose/

**Team 2 - LLM Inference:**
- Google Gemini 2.0 Flash - https://ai.google.dev/gemini-api/docs
- Ollama + Llama 3.1 - https://ollama.com/
- FastAPI - https://fastapi.tiangolo.com/
- ChromaDB (optional) - https://www.trychroma.com/

**Team 3 - Validation:**
- Batfish 2024+ - https://github.com/batfish/batfish
- Docker - https://www.docker.com/
- FastAPI - https://fastapi.tiangolo.com/
- LangChain - https://python.langchain.com/
- Groq Cloud (Llama 3.3 70B) - https://groq.com/

## 📁 Project Structure
```
P003-LLM/
├── .venv/                      # Global virtual environment (Teams 2 & 3)
├── Makefile                    # All automation commands
├── requirements.txt            # Python dependencies list
├── 01_DATA_ASSETS/
│   └── postgres_api/           # Docker-only (Team 1)
│       ├── docker-compose.yml  # PostgreSQL + pgvector setup
│       └── app/                # FastAPI Q&A service
├── 02_LLM_INFERENCE_API/       # Gemini/Llama inference + RAG
│   ├── models/
│   │   ├── keys.env            # API keys configuration
│   │   └── inference.json      # Model settings
│   ├── prompts/
│   │   └── prompts.json        # System prompts & schemas
│   └── endpoints/
│       └── inference.py        # Main generation logic
├── 03_AGENT_VALIDATION/
│   ├── langchain_agent/        # Orchestrator + interactive CLI
│   │   ├── config.json         # Agent configuration + Groq key
│   │   ├── agent_service.py    # Pipeline orchestration
│   │   └── interactive.py      # Rich CLI interface
│   └── batfish/                # Network simulation engine
│       └── docker-compose.yml  # Batfish container setup
└── README.md
```

---

## 📋 Available Commands

### Setup & Installation
```bash
make help              # Show all available commands
make install           # Install all dependencies
make setup             # Complete setup (install + config)
```

### Configuration
```bash
make config                          # Auto-configure with defaults
make config-interactive              # Interactive wizard
make T1_URL=http://host:port config  # Custom URLs
```

### Running Services
```bash
make run-all           # Start all services (T1 Docker, T2, T3)
make run-t1            # Team 1 only (PostgreSQL + API)
make run-t2            # Team 2 only (LLM Inference)
make run-t3            # Team 3 only (Batfish Validation)
make dev               # Development mode (all services with reload)
make interactive       # Interactive agent CLI

# Custom ports
make T2_PORT=9001 T3_PORT=5010 run-all
```

### Testing
```bash
make test-full         # Full pipeline test
make test-t2-t3        # T2→T3 integration test
make quick-test        # Quick API health check
make health-check      # Check all service health
```

### Management & Cleanup
```bash
make status            # Show running services & venv status
make stop              # Stop all services
make clean             # Clean cache files
make clean-venv        # Remove virtual environment
make logs-t1           # Docker logs (Team 1)
make logs-t3           # Batfish logs (Team 3)
```

---

## 🔧 Configuration

### Set Custom API URLs
```bash
make T1_URL=http://192.168.1.100:8000 \
     T2_URL=http://192.168.1.101:8001 \
     T3_URL=http://192.168.1.102:5000 \
     config
```

### Configure API Keys

**Groq API** (Team 3 - LLM Verdict):
- File: `03_AGENT_VALIDATION/langchain_agent/config.json`
- Get key: https://console.groq.com/keys
- Model: `llama-3.3-70b-versatile`
- Free tier: 30 requests/min, 6000 tokens/min

**Gemini API** (Team 2 - Config Generation):
- File: `02_LLM_INFERENCE_API/models/keys.env`
- Get key: https://aistudio.google.com/app/apikey
- Model: `gemini-2.0-flash`
- Free tier: 15 RPM, 1M TPM, 1500 RPD

**Ollama (Optional - Local Llama)**:
- File: `02_LLM_INFERENCE_API/models/keys.env`
- Installation: https://ollama.com/download
- Default: `http://localhost:11434/api/generate`
- Model: `llama3.1:8b` (pull with `ollama pull llama3.1:8b`)

---

## 🐛 Troubleshooting

### Docker Issues

**Docker not running:**
```bash
# Windows: Start Docker Desktop from Start Menu
# macOS: Start Docker.app from Applications
# Linux: sudo systemctl start docker
```

**Port conflicts:**
```bash
# Check what's using ports
netstat -ano | findstr :8000   # Windows
lsof -i :8000                  # macOS/Linux

# Use custom ports
make T1_PORT=9000 T2_PORT=9001 T3_PORT=9002 run-all
```

### Service Issues

**Team 1 (PostgreSQL) won't start:**
- Ensure Docker Desktop is running
- Check port 5432 is free: `netstat -ano | findstr :5432`
- View logs: `make logs-t1`
- Restart: `docker-compose -f 01_DATA_ASSETS/postgres_api/docker-compose.yml restart`

**Team 2 (LLM) API errors:**
- Verify Gemini API key in `02_LLM_INFERENCE_API/models/keys.env`
- Check API quota: https://aistudio.google.com/app/apikey
- Test connection: `curl https://generativelanguage.googleapis.com/v1/models`

**Team 3 (Batfish) validation fails:**
- Ensure Docker is running
- Check Batfish logs: `make logs-t3`
- Verify network configs are valid Cisco IOS format
- Increase timeout if needed: Edit `TIMEOUT` in `agent_service.py`

### API Key Issues

**Groq API rate limit:**
- Free tier: 30 req/min
- Upgrade: https://console.groq.com/settings/billing
- Alternative: Use cached responses or reduce request frequency

**Gemini API quota exceeded:**
- Free tier: 15 RPM, 1500 RPD
- Wait for quota reset (daily)
- Get paid tier: https://ai.google.dev/pricing

---

## 📚 Additional Resources

### Documentation Links

**Batfish:**
- Official Docs: https://batfish.readthedocs.io/
- GitHub: https://github.com/batfish/batfish
- Pybatfish API: https://pybatfish.readthedocs.io/

**LangChain:**
- Docs: https://python.langchain.com/docs/
- Groq Integration: https://python.langchain.com/docs/integrations/llms/groq

**FastAPI:**
- Tutorial: https://fastapi.tiangolo.com/tutorial/
- Deployment: https://fastapi.tiangolo.com/deployment/

**Docker:**
- Get Started: https://docs.docker.com/get-started/
- Compose: https://docs.docker.com/compose/

### Learning Resources

**Network Automation:**
- Batfish Tutorial: https://pybatfish.readthedocs.io/en/latest/notebooks/
- Cisco IOS Reference: https://www.cisco.com/c/en/us/support/ios-nx-os-software/ios-15-4m-t/products-command-reference-list.html

**LLM Development:**
- LangChain Quickstart: https://python.langchain.com/docs/get_started/quickstart
- Prompt Engineering Guide: https://www.promptingguide.ai/

---

## 📌 Version & Notes

**Version:** 2.0.0  
**Last Updated:** 2025  


### Key Features
- ✅ Single global virtual environment (`.venv/`)
- ✅ Interactive CLI with real-time visualization (Rich library)
- ✅ Runtime model switching (Gemini/Llama)
- ✅ RAG toggle (on/off) per query
- ✅ Loopback fallback on validation failure
- ✅ Protocol-agnostic validation with Batfish
- ✅ Multi-model support (Gemini 2.0, Llama 3.1/3.3)