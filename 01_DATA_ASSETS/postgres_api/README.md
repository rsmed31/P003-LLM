# Postgres QA API

FastAPI microservice for querying and managing QA records with similarity search support.

## Features

- **GET /qa/query**: Query QA records using text similarity search
- **GET /qa/{id}**: Retrieve a specific QA record by ID
- **POST /qa**: Create new QA records (stub implementation)
- Connection pooling for efficient database access
- Comprehensive error handling and logging
- Input validation with Pydantic models

## Architecture

```
postgres_api/
├── app.py              # FastAPI application and routes
├── db.py               # Database connection pool and query helpers
├── models.py           # Pydantic models for validation
├── config.py           # Configuration management
├── requirements.txt    # Python dependencies
├── test_app.py         # Unit tests
├── init.sql            # Database schema
├── docker-compose.yml  # Docker setup for Postgres
└── .env.example        # Environment variables template
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and update values:

```bash
cp .env.example .env
```

Edit `.env`:

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=postgres
DEFAULT_SIMILARITY_THRESHOLD=0.75
API_HOST=0.0.0.0
API_PORT=8000
```

### 3. Start Database

Using Docker:

```bash
docker-compose up -d
```

The database will automatically initialize with the schema from `init.sql`.

### 4. Enable pg_trgm Extension

Connect to your database and enable the trigram extension for text similarity:

```sql
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

## Running the API

### Development Mode

```bash
uvicorn app:app --reload --port 8000
```

Or:

```bash
python app.py
```

### Production Mode

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### Health Check

```bash
GET /
```

Response:

```json
{
  "status": "healthy",
  "service": "Postgres QA API",
  "version": "1.0.0"
}
```

### Query QA Records

```bash
GET /qa/query?text=your+question&threshold=0.75
```

Parameters:

- `text` (required): Query text to search for
- `threshold` (optional): Minimum similarity score (0.0-1.0, default: 0.75)

Response (match found):

```json
{
  "found": true,
  "data": {
    "id": 1,
    "question": "What is OSPF?",
    "answer": "OSPF is a routing protocol...",
    "lastUpdated": "2025-10-22T10:00:00",
    "score": 0.85
  },
  "message": "Match found"
}
```

Response (no match):

```json
{
  "found": false,
  "data": null,
  "message": "No match found with similarity >= 0.75"
}
```

### Get QA by ID

```bash
GET /qa/{id}
```

Response:

```json
{
  "id": 1,
  "question": "What is OSPF?",
  "answer": "OSPF is a routing protocol...",
  "lastUpdated": "2025-10-22T10:00:00"
}
```

### Create QA Record (Stub)

```bash
POST /qa
Content-Type: application/json

{
  "question": "What is BGP?",
  "answer": "BGP is the Border Gateway Protocol",
  "embedding": [0.1, 0.2, ...] // Optional, 384 dimensions
}
```

Response:

```json
{
  "message": "Request accepted but not persisted (stub implementation)",
  "id": null,
  "status": "not_implemented"
}
```

## Testing

Run unit tests:

```bash
pytest test_app.py -v
```

Run with coverage:

```bash
pytest test_app.py --cov=. --cov-report=html
```

## Testing with cURL

### Query QA

```bash
curl "http://localhost:8000/qa/query?text=What%20is%20OSPF&threshold=0.7"
```

### Get by ID

```bash
curl "http://localhost:8000/qa/1"
```

### Create QA (Stub)

```bash
curl -X POST "http://localhost:8000/qa" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is RIP?",
    "answer": "RIP is a distance-vector routing protocol"
  }'
```

## Similarity Methods

The API uses **text similarity** (pg_trgm extension) by default:

```sql
SELECT id, question, answer,
       similarity(question, %s) AS score
FROM qa
WHERE similarity(question, %s) >= %s
ORDER BY score DESC
LIMIT 1
```

For vector-based similarity (if QA table has embedding column):

```python
# Use query_qa_by_vector_similarity in db.py
result = query_qa_by_vector_similarity(embedding_vector, threshold)
```

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `202`: Accepted (for stub endpoints)
- `400`: Bad request (invalid parameters)
- `404`: Not found
- `422`: Validation error
- `500`: Internal server error

## Logging

Logs are written to stdout with the format:

```
%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

## Implementation Notes

### POST /qa Endpoint

Currently returns a stub response (202 Accepted). To enable database persistence, uncomment the insert logic in `app.py`:

```python
qa_id = insert_qa_record(
    question=payload.question,
    answer=payload.answer,
    embedding=payload.embedding
)

return QACreateResponse(
    message="QA record created successfully",
    id=qa_id,
    status="created"
)
```

### Adding Embedding Support to QA Table

To use vector similarity for QA queries, update the schema:

```sql
ALTER TABLE qa ADD COLUMN embedding VECTOR(384);
CREATE INDEX ON qa USING ivfflat (embedding vector_cosine_ops);
```

Then modify the query endpoint to use `query_qa_by_vector_similarity`.

## License

Part of the P003-LLM project.
