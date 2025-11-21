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

## 📞 Team Contact for Issues

### T1 (Data/RAG Service)
- **Q&A not returning results**: Check threshold in `call_t1_qa_lookup()` (currently 0.3)
- **Write-back failing**: Verify T1's `/qa` POST endpoint accepts validated configs

### T2 (LLM Inference)
- **Invalid JSON**: Check Team 2's model output cleaning in `endpoints/inference.py`
- **Missing fields**: Verify `prompts/prompts.json` schema instructions
- **String responses**: Team 2's model may need better output format training

### T3 (Validation)
- **Validation errors**: Check payload transformation in `_build_evaluate_payload_from_t2()`
- **Intent parsing**: Verify adjacency intent format matches T3 expectations

---

## 🔍 Debugging

Check logs for these markers:
- `[T2 Parse]` - Response format detection
- `[T2]` - HTTP calls to Team 2
- `[T3]` - Validation requests
- `[T1]` (if implemented) - Q&A and write operations

### Enable Debug Logging
Edit `config.json`:
```json
{
  "LOG_LEVEL": "DEBUG"
}
```
