import json
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel

# Import your existing generate() method from Team 2’s inference module
# It must return a JSON string like: {"model": "<model_name>", "response": "<text or commands>"}
from endpoints.inference import generate

app = FastAPI(
    title="Team 2 - getAnswer API",
    version="1.0.0",
    description="Endpoint used by Team 3 to get the generated answer or commands from Team 2."
)

# Allow cross-origin requests during testing (open to all origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redirect root to Swagger UI
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

# Update response model to handle array properly
class GetAnswerResponse(BaseModel):
    model: str
    response: list  # Changed from str to list to match actual response structure

@app.get("/health")
async def health():
    return {"ok": True}

@app.get("/v1/getAnswer", response_model=GetAnswerResponse, tags=["Inference"])
async def get_answer(
    q: str = Query(
        ...,
        min_length=3,
        description="User query to process",
        examples={
            "simple": {"summary": "What is BGP?", "value": "What is BGP?"},
            "config": {
                "summary": "OSPF configuration",
                "value": "Configure OSPF area 0 between R1 and R2 with router-ids 1.1.1.1 and 2.2.2.2."
            }
        }
    ),
    model: str = Query(
        "llama",
        description="Target model to use",
        pattern="^(gemini|llama)$"
    )
):
    """
    Team 3 calls this endpoint to get a generated answer or configuration commands.
    It delegates the query to Team 2's existing generate() method and returns only {model, response}.
    """
    try:
        # Call internal generate() logic - it returns JSON string
        raw = generate(query=q, model_name=model)

        # Parse the JSON string from generate()
        try:
            result = json.loads(raw)
            model_name = result.get("model", "unknown_model")
            response_data = result.get("response", [])
            
            # Ensure response is a list (array)
            if not isinstance(response_data, list):
                response_data = []
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=500, detail=f"Invalid JSON from model: {str(e)}")

        if not response_data:
            raise HTTPException(status_code=502, detail="Empty response from model")

        # Return properly structured response
        return {"model": model_name, "response": response_data}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Minimal browser test UI
@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
async def ui():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Team 2 - getAnswer Tester</title>
  <style>
    body { font-family: sans-serif; margin: 2rem; }
    label { display: block; margin-top: 1rem; }
    textarea, select, input[type=text] { width: 100%; padding: .5rem; }
    pre { background: #111; color: #0f0; padding: 1rem; overflow:auto; }
    button { margin-top: 1rem; padding: .6rem 1rem; }
  </style>
</head>
<body>
  <h1>getAnswer Tester</h1>
  <p>Try the API here, or open <a href="/docs" target="_blank">Swagger UI</a>.</p>
  <label>Model
    <select id="model">
      <option value="llama" selected>llama</option>
      <option value="gemini">gemini</option>
    </select>
  </label>
  <label>Query
    <textarea id="q" rows="4" placeholder="Type your query...">What is BGP?</textarea>
  </label>
  <button onclick="run()">Send</button>
  <pre id="out"></pre>
<script>
async function run() {
  const q = document.getElementById('q').value;
  const model = document.getElementById('model').value;
  const out = document.getElementById('out');
  out.textContent = 'Loading...';
  try {
    const params = new URLSearchParams({ q, model });
    const res = await fetch(`/v1/getAnswer?${params.toString()}`);
    const txt = await res.text();
    out.textContent = txt;
  } catch (e) {
    out.textContent = 'Error: ' + e;
  }
}
</script>
</body>
</html>
"""
