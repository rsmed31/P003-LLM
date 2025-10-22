# Postgres QA API - Quick Start Guide

## What Was Built

A complete FastAPI microservice for querying and managing QA records with the following features:

### Endpoints

1. **GET /** - Health check
2. **GET /qa/query** - Query QA records using text similarity
3. **GET /qa/{id}** - Get specific QA record by ID
4. **POST /qa** - Create QA record (stub implementation)

### Features

- ✅ FastAPI framework with async support
- ✅ PostgreSQL with pgvector extension
- ✅ Connection pooling for efficient DB access
- ✅ Text similarity search using pg_trgm
- ✅ Pydantic models for validation
- ✅ Comprehensive error handling
- ✅ Logging throughout
- ✅ Unit tests with pytest
- ✅ Docker Compose for database
- ✅ Environment variable configuration

## File Structure

```
postgres_api/
├── app.py              # Main FastAPI application
├── db.py               # Database connection pool & queries
├── models.py           # Pydantic request/response models
├── config.py           # Configuration from environment
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (created)
├── .env.example       # Environment template
├── test_app.py        # Unit tests
├── test_endpoints.py  # API endpoint testing script
├── test_endpoints.sh  # Bash test script
├── setup.sh           # Setup script
├── init.sql           # Database schema
├── docker-compose.yml # Docker setup
└── README.md          # Full documentation
```

## Current Status

✅ **Database**: Running in Docker (pgvector-database container)
✅ **API Server**: Ready to start
✅ **Dependencies**: Installed
✅ **Configuration**: Set up

## How to Start

### 1. Start the API Server

```bash
cd /c/Users/Aymane/P003-LLM/01_DATA_ASSETS/postgres_api
C:/Users/Aymane/P003-LLM/env/Scripts/python.exe -m uvicorn app:app --reload --port 8000
```

Or simply:

```bash
python app.py
```

### 2. Access the API

- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
<!-- - **OpenAPI Schema**: http://localhost:8000/openapi.json -->

### 3. Test Endpoints

#### Using Python Script:

```bash
python test_endpoints.py
```

#### Using curl:

```bash
# Health check
curl http://localhost:8000/

# Query QA
curl "http://localhost:8000/qa/query?text=What%20is%20OSPF&threshold=0.7"

# Get by ID
curl http://localhost:8000/qa/1

# Create QA (stub)
curl -X POST http://localhost:8000/qa \
  -H "Content-Type: application/json" \
  -d '{"question": "What is BGP?", "answer": "BGP is a routing protocol"}'
```

#### Run Unit Tests:

```bash
pytest test_app.py -v
```

## API Examples

### Query QA Records

```http
GET /qa/query?text=What%20is%20OSPF&threshold=0.75
```

**Response (Match Found):**

```json
{
  "found": true,
  "data": {
    "id": 1,
    "question": "What is OSPF?",
    "answer": "OSPF is a routing protocol...",
    "score": 0.85
  },
  "message": "Match found"
}
```

**Response (No Match):**

```json
{
  "found": false,
  "data": null,
  "message": "No match found with similarity >= 0.75"
}
```

### Get Specific QA

```http
GET /qa/1
```

**Response:**

```json
{
  "id": 1,
  "question": "What is OSPF?",
  "answer": "OSPF is a routing protocol...",
  "lastUpdated": "2025-10-22T10:00:00"
}
```

### Create QA (Stub)

```http
POST /qa
Content-Type: application/json

{
  "question": "What is BGP?",
  "answer": "BGP is the Border Gateway Protocol"
}
```

**Response:**

```json
{
  "message": "Request accepted but not persisted (stub implementation)",
  "id": null,
  "status": "not_implemented"
}
```

## Database Setup

The database is already running with Docker. To verify:

```bash
docker ps | grep pgvector
```

To stop the database:

```bash
docker-compose down
```

To restart:

```bash
docker-compose up -d
```

## Configuration

Edit `.env` file to change settings:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=postgres
DEFAULT_SIMILARITY_THRESHOLD=0.75
API_HOST=0.0.0.0
API_PORT=8000
```

## Similarity Search Method

Currently using **text similarity** (pg_trgm extension):

```sql
SELECT id, question, answer,
       similarity(question, %s) AS score
FROM qa
WHERE similarity(question, %s) >= %s
ORDER BY score DESC
LIMIT 1
```

## Implementation Notes

### POST /qa is a Stub

The POST endpoint validates input but doesn't persist to database. To enable persistence, uncomment the code in `app.py`:

```python
qa_id = insert_qa_record(
    question=payload.question,
    answer=payload.answer,
    embedding=payload.embedding
)
```

### Adding Vector Similarity

To use embedding-based similarity instead of text similarity, you need to:

1. Add embedding column to QA table:

```sql
ALTER TABLE qa ADD COLUMN embedding VECTOR(384);
```

2. Update the query endpoint to use `query_qa_by_vector_similarity`

## Troubleshooting

### Database Connection Failed

- Check if Docker is running: `docker ps`
- Start database: `docker-compose up -d`
- Check logs: `docker-compose logs -f`

### Port 8000 Already in Use

Change the port in `.env`:

```env
API_PORT=8001
```

### Import Errors

Install dependencies:

```bash
pip install -r requirements.txt
```

## Next Steps

1. ✅ **Testing**: Run `pytest test_app.py -v` to verify all tests pass
2. ✅ **Manual Testing**: Run `python test_endpoints.py` to test all endpoints
3. 🔄 **Enable POST persistence**: Uncomment insert logic in `app.py`
4. 🔄 **Add QA data**: Insert sample questions and answers
5. 🔄 **Add embedding support**: If using vector similarity
6. 🔄 **Deploy**: Set up production deployment

## Summary

Your FastAPI application is fully implemented and ready to use! The database is running, all dependencies are installed, and the API server is ready to start. Use the interactive documentation at http://localhost:8000/docs to explore and test the endpoints.
