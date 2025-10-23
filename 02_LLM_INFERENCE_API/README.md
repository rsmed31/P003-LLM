# Team 2 - LLM Inference API

Network configuration generation service using LLM models (Gemini & Llama/Ollama) with RAG support.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install fastapi uvicorn google-generativeai requests python-dotenv pydantic
```

### 2. Configure Environment Variables
Edit `models/keys.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
LLAMA_API_LINK=http://192.168.103.100:11434/api/generate
RETRIEVAL_SERVICE_URL=http://your-retrieval-service:5000/api/retrieve
```

### 3. Run the Server
```bash
cd c:\Users\enmoh\Desktop\P003-LLM\02_LLM_INFERENCE_API
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Access Interfaces
- **Swagger UI**: http://localhost:8000/docs
- **Simple Test UI**: http://localhost:8000/ui
- **Health Check**: http://localhost:8000/health

---

## 📁 Project Structure

```
02_LLM_INFERENCE_API/
├── app.py                          # FastAPI server entry point
├── endpoints/
│   ├── __init__.py
│   ├── inference.py                # Core inference logic (MAIN ENTRY)
│   └── prompt_builder.py           # Prompt assembly utilities
├── models/
│   ├── inference.json              # Model configuration
│   ├── keys.env                    # API keys & endpoints
│   ├── retrieval_config.json       # Protocol-specific chunk counts
│   └── zephyr_configurator         # Ollama Modelfile
├── prompts/
│   └── prompts.json                # Modular prompt templates
├── rag_logic/
│   └── code_aware_filter.py        # CLI-aware context filtering
└── retrieval/
    ├── __init__.py
    └── retrieval_orchestrator.py   # TF-IDF retrieval with correlation
```

---

## 🎯 Key Entry Points for Customization

### 1. **Model Selection** (`models/inference.json`)
Configure which models are available and their capabilities:

```json
{
  "gemini": {
    "model": "gemini-2.0-flash",
    "api_key_env": "GEMINI_API_KEY",
    "supports_rag": true
  },
  "llama": {
    "model": "zephyr_configurator",
    "api_link_env": "LLAMA_API_LINK",
    "supports_rag": false
  }
}
```

**Key Fields:**
- `model`: Model identifier (Gemini model name or Ollama model name)
- `supports_rag`: Enable/disable retrieval-augmented generation
- `api_key_env`/`api_link_env`: Environment variable names for credentials

---

### 2. **Prompt Engineering** (`prompts/prompts.json`)
Modify system prompts and output format requirements:

```json
{
  "SYSTEM_PROMPT": "You are a network configuration engine...",
  "CRITICAL_RULE_INSTRUCTION": "Your response MUST start with '[' and end with ']'...",
  "JSON_SCHEMA_BODY": "Return a JSON array where each object..."
}
```

**Critical Sections:**
- `SYSTEM_PROMPT`: Model's role and behavior
- `CRITICAL_RULE_INSTRUCTION`: Strict output format rules
- `JSON_SCHEMA_BODY`: Expected output structure with example

**⚠️ Best Practice:** Always test prompt changes with both models (Gemini & Llama) as they respond differently to instructions.

---

### 3. **RAG Configuration** (`models/retrieval_config.json`)
Adjust how many context chunks to retrieve per protocol:

```json
{
  "default_chunks": 75,
  "protocol_chunks": {
    "ospf": 100,
    "bgp": 120,
    "vlan": 50
  }
}
```

**Usage:** System automatically detects protocol from query and retrieves appropriate chunk count.

---

### 4. **API Endpoints** (`app.py`)

#### Main Inference Endpoint
```python
@app.get("/v1/getAnswer")
async def get_answer(q: str, model: str = "llama"):
    # Delegates to endpoints.inference.generate()
```

**Parameters:**
- `q`: User query (min 3 characters)
- `model`: "gemini" or "llama" (default: "llama")

**Response Format:**
```json
{
  "model": "llama",
  "response": [
    {
      "device_name": "R1",
      "protocol": "OSPF",
      "configuration_mode_commands": ["..."],
      "intent": [...]
    }
  ]
}
```

---

### 5. **Core Inference Logic** (`endpoints/inference.py`)

#### Main Function: `generate(query, model_name)`

**Workflow:**
1. Load model configuration
2. Fetch RAG context from external service (if enabled)
3. Perform local correlation analysis on chunks
4. Assemble prompt using `prompt_builder`
5. Call LLM API (Gemini or Ollama)
6. Clean and validate JSON response
7. Return structured output

**Key Functions to Modify:**

```python
# Change model behavior
def callGemini(model, prompt):
    # Customize Gemini API call parameters
    response = model.generate_content(prompt)
    return response.text.strip()

# Change output cleaning strategy
def clean_model_output(text: str) -> str:
    # Aggressive extraction of JSON array from model response
    # Modify if model wraps output differently
```

**Common Customization Points:**
- **Temperature**: Adjust in Modelfile (`zephyr_configurator`) or Gemini config
- **Stop sequences**: Modify in Modelfile `PARAMETER stop` directives
- **Validation**: Update `parse_and_validate_array()` for schema changes

---

### 6. **Prompt Assembly** (`endpoints/prompt_builder.py`)

#### Key Functions

```python
def assemble_rag_prompt(system_file_path, filtered_context, user_query):
    # Llama/Zephyr chat template format
    # Modify if using different local model

def assemble_rag_prompt_gemini(system_file_path, filtered_context, user_query):
    # Gemini-specific prompt structure
    # Adjust for Gemini's prompt preferences
```

**Template Structure (Llama/Zephyr):**
```
<|system|>
{SYSTEM_PROMPT}
</s>
<|user|>
{CRITICAL_RULE} + {TASK} + {CONTEXT} + {SCHEMA}
</s>
<|assistant|>
```

**Modification Tips:**
- Change chat template markers for different base models
- Adjust context injection point for optimal performance
- Test prompt order variations (context before/after task)

---

### 7. **Ollama Model Configuration** (`models/zephyr_configurator`)

```dockerfile
FROM hf.co/ccaug/zephyr-config-agent-3:F16

PARAMETER temperature 0
PARAMETER stop "<|system|>"
PARAMETER stop "<|user|>"
PARAMETER stop "<|assistant|>"
PARAMETER stop "</s>"

SYSTEM You are a network configuration engine...
```

**Key Parameters:**
- `temperature`: 0 for deterministic, >0 for creative
- `stop`: Prevent model from continuing conversation
- `SYSTEM`: Default system message (overridden by inference.py)

**To Apply Changes:**
```bash
ollama create zephyr_configurator -f models/zephyr_configurator
```

---

## 🔧 Common Modifications

### Change Default Model
Edit `app.py` line 55:
```python
model: str = Query(
    "gemini",  # Change to "gemini" for default
    description="Target model to use"
)
```

### Disable RAG
Edit `models/inference.json`:
```json
"supports_rag": false
```

### Add New Protocol Chunk Count
Edit `models/retrieval_config.json`:
```json
"protocol_chunks": {
  "mpls": 150,
  "your_protocol": 80
}
```

### Change JSON Schema
Edit `prompts/prompts.json` → `JSON_SCHEMA_BODY` section, then test thoroughly with both models.

### Adjust Cleaning Aggressiveness
Edit `endpoints/inference.py` → `clean_model_output()` function to handle model-specific output patterns.

---

## 🐛 Troubleshooting

### "Model output is not valid JSON"
1. Check `clean_model_output()` is extracting correctly
2. Verify prompt in `prompts.json` has strong format instructions
3. Test model directly via Ollama CLI: `ollama run zephyr_configurator`
4. Inspect raw model output (uncomment print statements in `inference.py`)

### Port 8000 in use
Change port in run command:
```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8080
```

### Empty response from model
1. Check API keys in `models/keys.env`
2. Verify Ollama is running: `ollama list`
3. Test external retrieval service connectivity
4. Review logs for HTTP errors

### CORS issues
Already configured for open access. If still blocked, check browser console and verify middleware in `app.py`.

---

## 📊 Testing

### Test via Swagger UI
1. Navigate to http://localhost:8000/docs
2. Expand `/v1/getAnswer`
3. Click "Try it out"
4. Enter query and select model
5. Execute and review response

### Test via Simple UI
1. Navigate to http://localhost:8000/ui
2. Select model from dropdown
3. Enter query in textarea
4. Click "Send"

### Test Programmatically
```python
import requests

response = requests.get(
    "http://localhost:8000/v1/getAnswer",
    params={
        "q": "Configure OSPF area 0 on R1 with router-id 1.1.1.1",
        "model": "llama"
    }
)
print(response.json())
```

---

## 🔐 Security Notes

- **API Keys**: Never commit `keys.env` to version control
- **CORS**: Current config allows all origins - restrict in production
- **Input Validation**: Queries are validated (min 3 chars, max enforced by FastAPI)
- **Rate Limiting**: Not implemented - add middleware for production

---

## 📝 API Reference

### `GET /v1/getAnswer`

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| q | string | Yes | - | User query (min 3 chars) |
| model | string | No | "llama" | "gemini" or "llama" |

**Response:** `200 OK`
```json
{
  "model": "llama",
  "response": [
    {
      "device_name": "R1",
      "protocol": "OSPF",
      "configuration_mode_commands": ["configure terminal", "..."],
      "intent": [...]
    }
  ]
}
```

**Errors:**
- `500`: Model error or processing failure
- `502`: Empty response from model

---

## 🤝 Integration with Other Teams

### For Team 1 (RAG Service)
Expected endpoint format:
```
GET /api/retrieve?query={query}&number={chunk_count}
Response: { "chunks": ["chunk1", "chunk2", ...] }
```

### For Team 3 (Frontend/Orchestration)
Call `/v1/getAnswer` with query and preferred model. Response contains structured JSON array ready for execution pipeline.

---

## 📌 Version

**Current Version:** 1.0.0  
**Last Updated:** 2025  
**Maintainer:** Team 2
