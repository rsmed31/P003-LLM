import json
from fastapi import FastAPI, HTTPException, Query

# Import your existing generate() method from Team 2’s inference module
# It must return a JSON string like: {"model": "<model_name>", "response": "<text or commands>"}
from endpoints.inference import generate

app = FastAPI(
    title="Team 2 - getAnswer API",
    version="1.0.0",
    description="Endpoint used by Team 3 to get the generated answer or commands from Team 2."
)

@app.get("/health")
async def health():
    return {"ok": True}

@app.get("/v1/getAnswer")
async def get_answer(
    q: str = Query(..., min_length=3, description="User query to process")
):
    """
    Team 3 calls this endpoint to get a generated answer or configuration commands.
    It delegates the query to Team 2's existing generate() method and returns only {model, response}.
    """
    try:
        # Call your internal generate() logic
        raw = generate(model_name="llama", prompt=q)

        # Normalize to clean JSON {model, response}
        try:
            result = json.loads(raw)
            model = result.get("model", "unknown_model")
            response_text = result.get("response", "").strip()
        except Exception:
            # In case generate() returns plain text
            model = "unknown_model"
            response_text = str(raw).strip()

        if not response_text:
            raise HTTPException(status_code=502, detail="Empty response from model")

        return {"model": model, "response": response_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
