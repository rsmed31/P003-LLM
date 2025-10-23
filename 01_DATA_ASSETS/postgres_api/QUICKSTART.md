# Postgres QA API - Quick Start Guide

## Quick Reference

### Start Everything

```bash
# 1. Start the database
docker-compose up -d

# 2. Start the API server
python app.py
```

### Stop Everything

```bash
# Stop API server (Ctrl+C in the terminal)

# Stop database
docker-compose down
```

### Access Points

- API: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- Database: localhost:5432

---

## What Was Built

A complete FastAPI microservice for querying and managing QA records with the following features:

### Endpoints

1. **GET /** - Health check
2. **GET /qa/query** - Query QA records using vector similarity search
3. **GET /qa/{id}** - Get specific QA record by ID
4. **POST /qa** - Create QA record with automatic embedding generation
5. **GET /chunks/query** - Query text chunks using semantic similarity search

### Features

- ✅ FastAPI framework with async support
- ✅ PostgreSQL with pgvector extension for vector similarity search
- ✅ Connection pooling for efficient DB access
- ✅ Vector similarity search using pgvector and sentence-transformers
- ✅ Automatic embedding generation using all-MiniLM-L6-v2 model
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

✅ **Database**: Ready to run in Docker (pgvector-database container)
✅ **API Server**: Ready to start
✅ **Dependencies**: Installed
✅ **Configuration**: Set up

## How to Start

### 1. Start the Database (Docker)

First, make sure Docker is running on your system. Then start the PostgreSQL database with pgvector extension:

```bash
cd /c/Users/Aymane/P003-LLM/01_DATA_ASSETS/postgres_api
docker-compose up -d
```

This will:

- Build a custom PostgreSQL image with pgvector extension
- Create a container named `pgvector-database`
- Initialize the database with the schema from `init.sql`
- Expose PostgreSQL on port 5432
- Store data persistently in a Docker volume

**Verify the database is running:**

```bash
docker ps | grep pgvector
```

You should see the `pgvector-database` container running.

**Check database logs (if needed):**

```bash
docker-compose logs -f
```

### 2. Start the API Server

```bash
cd /c/Users/Aymane/P003-LLM/01_DATA_ASSETS/postgres_api
C:/Users/Aymane/P003-LLM/env/Scripts/python.exe -m uvicorn app:app --reload --port 8000
```

Or simply:

```bash
python app.py
```

### 3. Access the API

- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
<!-- - **OpenAPI Schema**: http://localhost:8000/openapi.json -->

### 4. Test Endpoints

#### Using Python Script:

```bash
python test_endpoints.py
```

#### Using curl:

```bash
# Health check
curl http://localhost:8000/

# Query QA (vector similarity search)
curl "http://localhost:8000/qa/query?text=What%20is%20OSPF&threshold=0.8"

# Get by ID
curl http://localhost:8000/qa/1

# Create QA (with automatic embedding)
curl -X POST http://localhost:8000/qa \
  -H "Content-Type: application/json" \
  -d '{"question": "What is BGP?", "answer": "BGP is a routing protocol"}'

# Query text chunks (semantic search)
curl "http://localhost:8000/chunks/query?query=routing%20protocol&limit=5"
```

#### Run Unit Tests:

```bash
pytest test_app.py -v
```

## API Examples

### Query QA Records

```http
GET /qa/query?text=What%20is%20OSPF&threshold=0.8
```

**Response (Match Found):**

```json
{
  "found": true,
  "answer": "OSPF is a routing protocol...",
  "message": "Match found"
}
```

**Response (No Match):**

```json
{
  "found": false,
  "answer": null,
  "message": "No match found with similarity >= 0.8"
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

### Create QA

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
  "message": "QA record created successfully",
  "status": "created",
  "question": "What is BGP?",
  "answer": "BGP is the Border Gateway Protocol"
}
```

### Query Text Chunks

```http
GET /chunks/query?query=routing%20protocol&limit=5
```

**Response:**

```json
{
  "found": true,
  "results": [
    {
      "chunk_index": 42,
      "text": "OSPF is a routing protocol used for...",
      "similarity": 0.89
    },
    {
      "chunk_index": 15,
      "text": "Routing protocols enable routers to...",
      "similarity": 0.82
    }
  ],
  "count": 2,
  "message": "Found 2 matching chunks"
}
```

## Database Setup

### Docker Setup (Recommended)

The project includes Docker Compose configuration for easy database setup:

**Start the database:**

```bash
docker-compose up -d
```

**Stop the database:**

```bash
docker-compose down
```

**Stop and remove all data:**

```bash
docker-compose down -v
```

**View logs:**

```bash
docker-compose logs -f
```

**Restart the database:**

```bash
docker-compose restart
```

### What's Inside the Docker Setup

- **Base Image**: `pgvector/pgvector:pg18-trixie` - PostgreSQL 18 with pgvector extension
- **Auto-initialization**: The `init.sql` file runs automatically on first startup
- **Persistent Storage**: Data is stored in a Docker volume named `pg_data`
- **Port Mapping**: PostgreSQL is accessible on `localhost:5432`

### Database Schema

The database is automatically initialized with two tables:

1. **qa table**: Stores questions and answers with vector embeddings
2. **text_chunks table**: Stores document chunks with vector embeddings

Both tables use IVFFlat indexing for efficient vector similarity search.

### Connecting to the Database Directly

You can connect to the PostgreSQL database using any PostgreSQL client:

**Using psql (command line):**

```bash
docker exec -it pgvector-database psql -U postgres -d postgres
```

**Using a GUI client (e.g., pgAdmin, DBeaver):**

- Host: `localhost`
- Port: `5432`
- Database: `postgres`
- User: `postgres`
- Password: `postgres`

**Quick SQL queries:**

```sql
-- View all QA records
SELECT id, question, LEFT(answer, 50) FROM qa;

-- View text chunks
SELECT id, chunk_index, LEFT(text, 50) FROM text_chunks LIMIT 10;

-- Check vector extension
SELECT * FROM pg_extension WHERE extname = 'vector';
```

## Configuration

Edit `.env` file to change settings:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=postgres
DEFAULT_SIMILARITY_THRESHOLD=0.8
API_HOST=0.0.0.0
API_PORT=8000
```

## Similarity Search Method

Currently using **vector similarity search** with pgvector extension:

- **QA Search**: Uses pgvector with sentence-transformers (all-MiniLM-L6-v2) model
- **Text Chunks**: Semantic similarity search using vector embeddings
- **Embedding Dimension**: 384 dimensions
- **Distance Metric**: Cosine similarity

The embeddings are automatically generated when creating QA records or stored with text chunks.

### Database Schema

```sql
-- QA table with vector embeddings
CREATE TABLE qa (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL UNIQUE,
    answer TEXT NOT NULL,
    lastUpdated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    embedding VECTOR(384)
);

-- Text chunks table with vector embeddings
CREATE TABLE text_chunks (
    id SERIAL PRIMARY KEY,
    source TEXT,
    chunk_index INT,
    text TEXT,
    embedding VECTOR(384)
);
```

## Implementation Notes

### POST /qa is Fully Implemented

The POST endpoint now fully persists QA records to the database with automatic embedding generation:

```python
# Embeddings are automatically generated using sentence-transformers
success = qaembedding(payload.question, payload.answer)
```

The system uses the **all-MiniLM-L6-v2** model to generate 384-dimensional embeddings for both questions and answers, enabling semantic similarity search.

### Vector Similarity Search

Both QA and text chunk queries use vector similarity:

1. **QA Query** (`/qa/query`): Searches the `qa` table using vector embeddings
2. **Text Chunks Query** (`/chunks/query`): Searches the `text_chunks` table using vector embeddings

The pgvector extension with IVFFlat indexing enables efficient similarity searches.

## Troubleshooting

### Docker Not Running

If you get an error about Docker not being available:

```bash
# Check if Docker is running
docker --version
docker ps
```

**Windows**: Start Docker Desktop
**Linux**: `sudo systemctl start docker`

### Database Connection Failed

- Check if Docker container is running: `docker ps`
- Start database: `docker-compose up -d`
- Check logs: `docker-compose logs -f`
- Verify port 5432 is not in use: `netstat -an | grep 5432` (Linux/Mac) or `netstat -an | findstr 5432` (Windows)

### Port 5432 Already in Use

If you have another PostgreSQL instance running:

**Option 1**: Stop the other PostgreSQL instance
**Option 2**: Change the port in `docker-compose.yml`:

```yaml
ports:
  - "5433:5432" # Use port 5433 instead
```

Then update `.env`:

```env
DB_PORT=5433
```

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

### Container Fails to Start

Check the Docker logs for detailed error messages:

```bash
docker-compose logs
```

Common issues:

- Port 5432 already in use (see above)
- Insufficient disk space
- Docker daemon not running

### Rebuild Docker Image

If you modified the Dockerfile or init.sql:

```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## Next Steps

1. ✅ **Testing**: Run `pytest test_app.py -v` to verify all tests pass
2. ✅ **Manual Testing**: Run `python test_endpoints.py` to test all endpoints
3. ✅ **Add QA data**: Use POST `/qa` to add questions and answers
4. ✅ **Query QA**: Use GET `/qa/query` for semantic similarity search
5. ✅ **Query Chunks**: Use GET `/chunks/query` for document chunk retrieval
6. 🔄 **Deploy**: Set up production deployment with proper security
7. 🔄 **Monitor**: Add monitoring and logging for production use

## Summary

Your FastAPI application is fully implemented with vector similarity search capabilities! The database is running with pgvector extension, all dependencies are installed, and the API server supports:

- **Semantic QA Search**: Query questions and answers using natural language
- **Text Chunk Retrieval**: Search through document chunks with semantic similarity
- **Automatic Embeddings**: All text is automatically converted to 384-dimensional vectors
- **Production Ready**: Complete with error handling, logging, and connection pooling

Use the interactive documentation at http://localhost:8000/docs to explore and test all endpoints.
