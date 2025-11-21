# P003-LLM: Network Configuration Generation System

Multi-team LLM-powered network configuration generation and validation pipeline.

---

## 🚀 Quick Start

### 1. Install Everything (Creates Virtual Environments)
```bash
make install
```

This creates separate `.venv` directories for each team:
- `01_DATA_ASSETS/.venv`
- `02_LLM_INFERENCE_API/.venv`
- `03_AGENT_VALIDATION/langchain_agent/.venv`
- `03_AGENT_VALIDATION/batfish/.venv`

### 2. Configure API URLs
```bash
# Interactive wizard
make config-interactive

# Or use defaults
make config
```

### 3. Run All Services
```bash
make run-all
```

### 4. Test the Pipeline
```bash
make test-full
```

---

## 🐍 Virtual Environments

Each team has its own isolated Python environment to prevent dependency conflicts.

### Why Virtual Environments?

- ✅ **Isolation**: Each service has its own dependencies
- ✅ **No Conflicts**: Different versions of same packages
- ✅ **Clean System**: Don't pollute global Python
- ✅ **Reproducible**: Exact dependency versions

### Manual Activation

If you need to work on a specific team:

```bash
# Team 1
cd 01_DATA_ASSETS
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Team 2
cd 02_LLM_INFERENCE_API
source .venv/bin/activate

# Team 3 Agent
cd 03_AGENT_VALIDATION/langchain_agent
source .venv/bin/activate

# Team 3 Batfish
cd 03_AGENT_VALIDATION/batfish
source .venv/bin/activate
```

### Check Virtual Environment Status
```bash
make status
```

---

## 📋 Available Commands

### Setup & Installation
```bash
make help              # Show all available commands
make install           # Install all dependencies in venvs
make install-t1        # Install Team 1 only
make install-t2        # Install Team 2 only
make install-t3        # Install Team 3 only
make setup             # Complete setup (install + config)
```

### Configuration
```bash
make config                          # Auto-configure with default URLs
make config-interactive              # Interactive configuration wizard
make T1_URL=http://host:port config  # Configure with custom URLs
```

### Running Services
```bash
make run-all           # Run all services
make run-t1            # Run Team 1 (Data/RAG)
make run-t2            # Run Team 2 (LLM Inference)
make run-t3            # Run Team 3 (Agent)
make dev               # Development mode (all services with reload)

# Custom ports
make T1_PORT=9000 run-t1
make T2_PORT=9001 T3_PORT=9002 run-all
```

### Testing
```bash
make test-full         # Test full pipeline
make test-t2-t3        # Test T2→T3 integration
make quick-test        # Quick API test
make health-check      # Check all services
```

### Management & Cleanup
```bash
make status            # Show running services & venv status
make stop              # Stop all services
make clean             # Clean cache files
make clean-venv        # Remove all virtual environments
make clean-t3          # Clean Team 3 Docker volumes
make logs-t1           # Show Team 1 Docker logs
make logs-t3           # Show Team 3 Batfish logs
```

### Helper Commands
```bash
make activate-t1       # Show command to activate T1 venv
make activate-t2       # Show command to activate T2 venv
make activate-t3       # Show command to activate T3 venv
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Team 3: Agent Orchestrator               │
│                   (03_AGENT_VALIDATION)                      │
│  • LangChain LCEL Pipeline                                  │
│  • Groq LLM for verdict synthesis                           │
│  • Coordinates all teams                                     │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│   Team 1: RAG   │  │ Team 2: LLM Gen  │  │ Team 3: Validate │
│ (01_DATA_ASSETS)│  │(02_LLM_INFERENCE)│  │  (Simulation)    │
│                 │  │                  │  │                  │
│ • PostgreSQL    │  │ • Gemini/Llama   │  │ • Config testing │
│ • Q&A Lookup    │  │ • RAG-enhanced   │  │ • Intent check   │
│ • Write-back    │  │ • Multi-model    │  │ • PASS/FAIL      │
└─────────────────┘  └──────────────────┘  └──────────────────┘
```

---

## 🔧 Custom Configuration

### Set Custom API URLs
```bash
make T1_URL=http://192.168.1.100:8000 \
     T2_URL=http://192.168.1.101:8001 \
     T3_URL=http://192.168.1.102:5000 \
     config
```

### Configure API Keys
Edit files directly or use wizard:
```bash
make config-interactive
```

**Manual configuration:**
- **Groq API**: `03_AGENT_VALIDATION/langchain_agent/config.json`
- **Gemini API**: `02_LLM_INFERENCE_API/models/keys.env`

---

## 📁 Project Structure

```
P003-LLM/
├── .gitignore                  # 🆕 Excludes .venv from git
├── Makefile                    # Master orchestration with venv support
├── 01_DATA_ASSETS/             # Team 1: PostgreSQL + RAG
│   ├── .venv/                  # 🆕 Virtual environment
│   ├── requirements.txt        # Python dependencies
│   ├── docker-compose.yml
│   └── app.py
├── 02_LLM_INFERENCE_API/       # Team 2: LLM Generation
│   ├── .venv/                  # 🆕 Virtual environment
│   ├── requirements.txt        # Python dependencies
│   ├── app.py
│   ├── endpoints/
│   └── models/
└── 03_AGENT_VALIDATION/        # Team 3: Orchestration
    ├── langchain_agent/
    │   ├── .venv/              # 🆕 Virtual environment
    │   ├── requirements.txt    # Python dependencies
    │   ├── config.json         # Auto-configured
    │   └── agent_service.py
    └── batfish/
        ├── .venv/              # 🆕 Virtual environment
        ├── requirements.txt    # Python dependencies
        ├── Dockerfile
        └── docker-compose.yml
```

---

## 🧪 Example Usage

### Complete Setup from Scratch
```bash
# 1. Install all dependencies in virtual environments
make install

# 2. Configure (interactive)
make config-interactive

# 3. Start all services
make run-all

# 4. Wait for services to start (5 seconds)
sleep 5

# 5. Run test
make quick-test
```

### Development Workflow
```bash
# Start services in dev mode
make dev

# In another terminal, test changes
make test-t2-t3

# Check logs and status
make status
```

### Working on Specific Team
```bash
# Activate Team 2 venv
cd 02_LLM_INFERENCE_API
source .venv/bin/activate

# Make changes to code
# ...

# Test manually
python -m uvicorn app:app --reload

# Deactivate when done
deactivate
```

---

## 🐛 Troubleshooting

### Virtual environment not found
```bash
# Reinstall specific team
make install-t1  # or t2, t3

# Or reinstall everything
make clean-venv
make install
```

### Dependencies issues
```bash
# Clean and reinstall
make clean
make clean-venv
make install
```

### Port conflicts
```bash
# Use custom ports
make T1_PORT=9000 T2_PORT=9001 T3_PORT=9002 run-all
```

### Services won't start
```bash
# Check what's running
make status

# Stop everything
make stop

# Try again
make run-all
```

---

## 📞 Team Responsibilities

| Team | Service | Port | Key Files |
|------|---------|------|-----------|
| T1 | Data/RAG | 8000 | `01_DATA_ASSETS/app.py` |
| T2 | LLM Inference | 8001 | `02_LLM_INFERENCE_API/app.py` |
| T3 | Agent + Validation | 5000 | `03_AGENT_VALIDATION/langchain_agent/agent_service.py` |

---

## 📌 Version

**Current Version:** 2.0.0  
**Last Updated:** 2025  
**Makefile Support:** ✅ Added
