# langchain_agent: Orchestration Logic
Contains the agent_flow.py and all tool definitions required to sequence the E2E process, including the VPP loop logic.

# Team 3 - Agent Orchestrator

Orchestration logic that sequences the E2E pipeline: T1 (Q&A + Write) → T2 (LLM) → T3 (Validation).

---

## 🔄 Pipeline Flow

```
User Query
    ↓
┌─────────────────────────────────────┐
│ T1: Q&A Lookup (GET /qa/query)      │ ← Check if answer exists in KB
└─────────────────────────────────────┘
    │
    ├─ FOUND → Return cached answer ✅
    │
    └─ NOT FOUND ↓
       ┌─────────────────────────────────────┐
       │ T2: LLM Generate (GET /v1/getAnswer)│ ← Generate config with RAG
       └─────────────────────────────────────┘
           ↓
       ┌─────────────────────────────────────┐
       │ T3: Validate (POST /evaluate)       │ ← Test config in simulation
       └─────────────────────────────────────┘
           ↓
       ┌─────────────────────────────────────┐
       │ LLM: Verdict Synthesis (Groq)       │ ← Explain PASS/FAIL
       └─────────────────────────────────────┘
           ↓
       ┌─────────────────────────────────────┐
       │ T1: Write-back (POST /qa)           │ ← Store validated config
       └─────────────────────────────────────┘
```

---

## 🤝 Contract Requirements for Team 2

### Expected Response Format

Team 2's `/v1/getAnswer` endpoint **must** return:

```json
{
  "model": "llama",
  "response": [
    {
      "device_name": "R1",
      "protocol": "OSPF",
      "configuration_mode_commands": [
        "router ospf 1",
        "network 10.0.0.0 0.0.0.255 area 0"
      ],
      "intent": [
        {
          "type": "adjacency",
          "endpoints": [
            {"role": "router", "id": "R1"},
            {"role": "router", "id": "R2"}
          ]
        }
      ]
    }
  ]
}
```

### Fallback Support

The agent also handles:
- **String format**: `{"response": "configure terminal\n..."}`
- **Direct list**: `[{"device_name": "R1", ...}]`

### Critical Fields

Each device object must contain:
- `device_name` (string): Router/switch identifier
- `configuration_mode_commands` (list of strings): CLI commands
- `protocol` (string, optional): Protocol type
- `intent` (list, optional): Validation intents

---

## 🚀 Usage

### Interactive Mode (Recommended)

Launch the interactive CLI with real-time pipeline visualization:

```bash
# From project root (activate global venv first)
cd 03_AGENT_VALIDATION\langchain_agent
python interactive.py
```

**Features:**
- 🎨 Beautiful real-time pipeline visualization
- 📊 Step-by-step progress tracking
- 🔄 Live configuration preview
- 💬 Chat-like interface
- ⚡ Fast model switching

**Commands:**
- `<query>` - Process a network configuration query
- `model gemini` - Switch to Gemini model
- `model llama` - Switch to Llama model
- `status` - Show current configuration
- `help` - Show all commands
- `exit` / `quit` - Exit interactive mode

**Example Session:**
```
> configure ospf on 3 routers
```
```bash
# gemini model
model gemini
# llama model
model llama
# show status
status
# exit
exit
```

### Interactive Mode (Recommended)

Launch the interactive CLI with real-time pipeline visualization:

```bash
# From project root (activate global venv first)
cd 03_AGENT_VALIDATION\langchain_agent
python interactive.py
```

**Features:**
- 🎨 Beautiful real-time pipeline visualization
- 📊 Step-by-step progress tracking
- 🔄 Live configuration preview
- 💬 Chat-like interface
- ⚡ Fast model switching

**Commands:**
- `<query>` - Process a network configuration query
- `model gemini` - Switch to Gemini model
- `model llama` - Switch to Llama model
- `status` - Show current configuration
- `help` - Show all commands
- `exit` / `quit` - Exit interactive mode

**Example Session:**
```
> configure ospf on 3 routers
```
```bash
# gemini model
model gemini
# llama model
model llama
# show status
status
# exit
exit
```

### Run Full Pipeline
```bash
python agent_service.py --query "Configure OSPF on 3 routers" --model gemini
```

### Skip T1 Q&A Lookup (Force T2 generation)
```bash
python agent_service.py --query "..." --skip-t1-qa
```

### Skip T1 Write-back
```bash
python agent_service.py --query "..." --skip-t1-write
```

### Use Different Model
```bash
python agent_service.py --query "..." --model llama
```

### Makefile Shortcut
```bash
# From project root
make interactive            # Launch interactive CLI
make T3_MODE=interactive run-t3   # Start Batfish + interactive agent
```

---

## 📝 Configuration

Edit `config.json`:

```json
{
  "T1_BASE_URL": "http://t1:8000",
  "T1_ENDPOINT_QA_LOOKUP": "/qa/query",
  "T1_ENDPOINT_WRITE": "/qa",
  "T2_BASE_URL": "http://t2:8000",
  "T2_ENDPOINT_GENERATE": "/v1/getAnswer",
  "T3_BASE_URL": "http://t3:5000",
  "T3_ENDPOINT_VALIDATE": "/evaluate",
  "HTTP_TIMEOUT": 90,
  "GROQ_API_KEY": "your_api_key_here",
  "T2_DEFAULT_MODEL": "llama"
}
```

---

## 🔍 Debugging

Check logs for these markers:
- `[T2 Parse]` - Response format detection
- `[T2]` - HTTP calls to Team 2
- `[T3]` - Validation requests
- `[T1]` (if implemented) - Q&A and write operations

### Common T3 Validation Errors

**Error: "A Batfish nodeSpec must be a string"**
- **Cause**: Device names passed as list instead of comma-separated string
- **Fix**: Applied in validator.py `_run_cp()` - converts list to string
- **Check**: Verify `changes` dict keys are valid device names (no special chars)

**Error: "No reachable paths found"**
- **Cause**: Devices not connected or missing IP configuration
- **Fix**: Ensure generated config includes interface IPs and routing
- **Debug**: Check base snapshot has proper topology

**Error: "VERIFY failed"**
- **Cause**: Batfish query exception (missing nodes, invalid syntax)
- **Fix**: Check validator logs for detailed stack trace
- **Debug**: Test with simpler config (single device, no intents)

### Common T3 Validation Errors

**Error: "A Batfish nodeSpec must be a string"**
- **Cause**: Device names passed as list instead of comma-separated string
- **Fix**: Applied in validator.py `_run_cp()` - converts list to string
- **Check**: Verify `changes` dict keys are valid device names (no special chars)

**Error: "No reachable paths found"**
- **Cause**: Devices not connected or missing IP configuration
- **Fix**: Ensure generated config includes interface IPs and routing
- **Debug**: Check base snapshot has proper topology

**Error: "VERIFY failed"**
- **Cause**: Batfish query exception (missing nodes, invalid syntax)
- **Fix**: Check validator logs for detailed stack trace
- **Debug**: Test with simpler config (single device, no intents)

### Enable Debug Logging
Edit `config.json`:
```json
{
  "LOG_LEVEL": "DEBUG"
}
```

## 📜 Pipeline Logging (LangChain Callbacks)

The agent writes JSONL trace events to `03_AGENT_VALIDATION/langchain_agent/logs/pipeline.log`.

Enable/disable (default enabled):
```
LOG_EVENTS=1   # enabled
LOG_EVENTS=0   # disabled
LOG_DIR=custom/path  # optional
```

Events captured:
- chain_start / chain_end
- llm_start / llm_end / llm_error
- t3_http_error / t3_error
- agent_result

Tail the log:
```bash
tail -f 03_AGENT_VALIDATION/langchain_agent/logs/pipeline.log
```

Pretty-print last 10 events (jq):
```bash
jq -r '.event + " | " + .ts' 03_AGENT_VALIDATION/langchain_agent/logs/pipeline.log | tail -n 10
```

Filter only LLM prompts:
```bash
grep '"event":"llm_start"' 03_AGENT_VALIDATION/langchain_agent/logs/pipeline.log | jq '.prompts'
```

To disable logging temporarily:
```bash
export LOG_EVENTS=0
python agent_service.py --query "Configure OSPF..."
```

## ⚠️ Known Limitation: OSPF Validation Requires Complete Configs

**Issue:** If Team 2 returns only `router ospf` stanzas without interface IPs, validation will fail (0 edges, no reachability).

**Root Cause:**
- Batfish requires interfaces with IP addresses to build topology
- OSPF network statements must match actual interface IPs
- Loopback-only configs (fallback) cannot form adjacencies