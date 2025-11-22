# Team 2 - LLM Inference API

Network configuration generation service using LLM models (Gemini & Llama/Ollama) with RAG support.

---

## 🚀 Quick Start

### 1. Create Virtual Environment & Install Dependencies
```bash
cd 02_LLM_INFERENCE_API
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
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
source .venv/bin/activate  # If not already activated
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Access Interfaces
- **Swagger UI**: http://localhost:8000/docs
- **Simple Test UI**: http://localhost:8000/ui
- **Health Check**: http://localhost:8000/health

---

## 📋 Using the Makefile (Recommended)

From the project root:

```bash
# Install with venv
make install-t2

# Run the service
make run-t2

# Or with custom port
make T2_PORT=9001 run-t2
```

---

## 🔧 Manual Virtual Environment Setup

### Create Virtual Environment
```bash
python3 -m venv .venv
```

### Activate Virtual Environment

**Linux/macOS:**
```bash
source .venv/bin/activate
```

**Windows:**
```cmd
.venv\Scripts\activate
```

### Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
02_LLM_INFERENCE_API/
├── .venv/                      # Virtual environment (not in git)
├── app.py                      # FastAPI server entry point
├── requirements.txt            # Python dependencies
├── endpoints/
│   ├── __init__.py
│   ├── inference.py            # Core inference logic (MAIN ENTRY)
│   └── prompt_builder.py       # Prompt assembly utilities
├── models/
│   ├── inference.json          # Model configuration
│   ├── keys.env                # API keys & endpoints
│   ├── retrieval_config.json   # Protocol-specific chunk counts
│   └── zephyr_configurator     # Ollama Modelfile
├── prompts/
│   └── prompts.json            # Modular prompt templates
├── rag_logic/
│   └── code_aware_filter.py    # CLI-aware context filtering
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

---

## 📝 API Reference

### `GET /v1/getAnswer`

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| q | string | Yes | - | User query (min 3 chars) |
| model | string | No | "llama" | "gemini" or "llama" |
| rag | string | No | "on" | "on" for RAG, "off" for direct inference |

**Response:** `200 OK`
```json
{
  "model": "llama",
  "rag_enabled": true,
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

**Example Requests:**
```bash
# With RAG (default)
curl "http://localhost:8001/v1/getAnswer?q=configure+ospf&model=gemini"

# Without RAG (direct inference)
curl "http://localhost:8001/v1/getAnswer?q=configure+ospf&model=gemini&rag=off"
```

**Errors:**
- `500`: Model error or processing failure
- `502`: Empty response from model or schema validation failure
  - Missing required fields (device_name, configuration_mode_commands, protocol, intent)
  - Empty configuration_mode_commands array
  - Malformed device objects

**Schema Validation:**
The endpoint performs strict schema validation. All device objects MUST contain:
- `device_name` (string): Device identifier
- `configuration_mode_commands` (array): Non-empty list of CLI commands
- `protocol` (string): Routing protocol (OSPF/BGP/EIGRP/STATIC)
- `intent` (array): List of adjacency/connectivity intents

Missing or incomplete fields will result in HTTP 502 error with detailed explanation.

---

### `GET /test`

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| q | string | Yes | - | Query to test (no model call) |
| model | string | No | "llama" | "gemini" or "llama" |

**Description:** Debug endpoint that shows RAG pipeline behavior without calling the model. Returns:
- Retrieval service status
- Number of chunks retrieved
- Filtered context preview
- Assembled prompt preview
- Target API endpoint

**Response:** `200 OK`
```json
{
  "query": "configure ospf",
  "model": "gemini",
  "rag_enabled": true,
  "retrieval_service": "http://localhost:8000",
  "chunks_requested": 100,
  "chunks_retrieved": 10,
  "chunks_preview": [...],
  "filtered_context": "...",
  "prompt_preview": "...",
  "full_prompt_length": 5432,
  "would_send_to": "https://generativelanguage.googleapis.com/...",
  "retrieval_error": null
}
```

---

### `GET /health`

**Description:** Health check endpoint

**Response:** `200 OK`
```json
{
  "ok": true
}
```
