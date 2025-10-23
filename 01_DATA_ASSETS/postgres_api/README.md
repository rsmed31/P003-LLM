# Postgres QA API

FastAPI microservice for querying and managing QA records and text chunks using vector-based similarity search.

## Overview

This API provides semantic search capabilities for question-answer pairs and document chunks. It uses AI embeddings to find the most relevant answers to your questions, even when the wording doesn't match exactly.

**Base URL**: `http://localhost:8000`

## Available Endpoints

| Method | Endpoint        | Description                                   |
| ------ | --------------- | --------------------------------------------- |
| GET    | `/`             | Health check                                  |
| GET    | `/qa/query`     | Search for answers by question similarity     |
| GET    | `/qa/{id}`      | Get a specific QA record by ID                |
| POST   | `/qa`           | Create a new QA record                        |
| GET    | `/chunks/query` | Search document chunks by semantic similarity |

## Quick Start

### Prerequisites

- Python 3.8+
- Docker and Docker Compose

### Installation

1. **Clone the repository and navigate to the API directory**

```bash
cd 01_DATA_ASSETS/postgres_api
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Start the database**

```bash
docker-compose up -d
```

4. **Run the API**

```bash
python app.py
```

Or using uvicorn:

```bash
uvicorn app:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

### Verify Installation

Check the health endpoint:

```bash
curl http://localhost:8000
```

Expected response:

```json
{
  "status": "healthy",
  "service": "Postgres QA API",
  "version": "1.0.0"
}
```

## API Reference

### 1. Health Check

Check if the API is running.

**Endpoint**: `GET /`

**Example Request**:

```bash
curl http://localhost:8000
```

**Response**:

```json
{
  "status": "healthy",
  "service": "Postgres QA API",
  "version": "1.0.0"
}
```

---

### 2. Query QA Records

Search for answers using semantic similarity. The API finds the most similar question in the database and returns its answer.

**Endpoint**: `GET /qa/query`

**Parameters**:

- `text` (required): Your question or query text
- `threshold` (optional): Minimum similarity score (0.0-1.0, default: 0.8)

**Example Request**:

```bash
curl "http://localhost:8000/qa/query?text=What%20is%20OSPF&threshold=0.7"
```

**Success Response** (200 OK):

```json
{
  "found": true,
  "answer": "OSPF (Open Shortest Path First) is a routing protocol...",
  "message": "Match found"
}
```

**No Match Response** (200 OK):

```json
{
  "found": false,
  "answer": null,
  "message": "No match found with similarity >= 0.7"
}
```

---

### 3. Get QA by ID

Retrieve a specific QA record by its ID.

**Endpoint**: `GET /qa/{id}`

**Parameters**:

- `id` (path parameter): The ID of the QA record

**Example Request**:

```bash
curl http://localhost:8000/qa/1
```

**Success Response** (200 OK):

```json
{
  "id": 1,
  "question": "What is OSPF?",
  "answer": "OSPF (Open Shortest Path First) is a routing protocol...",
  "lastUpdated": "2025-10-22T10:00:00"
}
```

**Not Found Response** (404):

```json
{
  "detail": "QA record with ID 1 not found"
}
```

---

### 4. Create QA Record

Add a new question-answer pair to the database. The API automatically generates embeddings for semantic search.

**Endpoint**: `POST /qa`

**Request Body**:

```json
{
  "question": "What is BGP?",
  "answer": "BGP (Border Gateway Protocol) is used for routing between autonomous systems"
}
```

**Example Request**:

```bash
curl -X POST http://localhost:8000/qa \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is BGP?",
    "answer": "BGP (Border Gateway Protocol) is used for routing between autonomous systems"
  }'
```

**Success Response** (201 Created):

```json
{
  "message": "QA record created successfully",
  "status": "created",
  "question": "What is BGP?",
  "answer": "BGP (Border Gateway Protocol) is used for routing between autonomous systems"
}
```

**Validation Error** (400/422):

```json
{
  "detail": "Question cannot be empty or only whitespace"
}
```

---

### 5. Query Text Chunks

Search for relevant document chunks using semantic similarity. Returns multiple results ranked by relevance.

**Endpoint**: `GET /chunks/query`

**Parameters**:

- `query` (required): Your search text
- `limit` (optional): Maximum number of results (1-50, default: 5)

**Example Request**:

```bash
curl "http://localhost:8000/chunks/query?query=routing%20protocols&limit=3"
```

**Success Response** (200 OK):

```json
{
  "found": true,
  "results": [
    {
      "chunk_index": 42,
      "text": "OSPF (Open Shortest Path First) is a link-state routing protocol...",
      "similarity": 0.8734
    },
    {
      "chunk_index": 105,
      "text": "Routing protocols determine the best path for data...",
      "similarity": 0.8123
    },
    {
      "chunk_index": 89,
      "text": "BGP is the protocol used for internet routing...",
      "similarity": 0.7956
    }
  ],
  "count": 3,
  "message": "Found 3 matching chunks"
}
```

**No Results Response** (200 OK):

```json
{
  "found": false,
  "results": [],
  "count": 0,
  "message": "No matching chunks found"
}
```

---

## Error Responses

All endpoints return standard HTTP status codes:

| Status Code | Description                      |
| ----------- | -------------------------------- |
| 200         | Success                          |
| 201         | Created (POST /qa)               |
| 400         | Bad request - invalid parameters |
| 404         | Resource not found               |
| 422         | Validation error                 |
| 500         | Internal server error            |

**Error Response Format**:

```json
{
  "detail": "Error description here"
}
```

## Understanding Similarity Scores

The API uses AI embeddings to find semantically similar content. Similarity scores range from 0.0 to 1.0:

- **0.9 - 1.0**: Extremely similar (near-exact match)
- **0.8 - 0.9**: Very similar (highly relevant)
- **0.7 - 0.8**: Similar (likely relevant)
- **Below 0.7**: Less similar (may not be relevant)

**Tips**:

- Start with a threshold of 0.8 for precise matches
- Lower to 0.7 if you need more results
- The API finds matches even if the wording is different

## Usage Examples

### Python

```python
import requests

# Query QA
response = requests.get(
    "http://localhost:8000/qa/query",
    params={"text": "What is OSPF?", "threshold": 0.7}
)
data = response.json()
if data["found"]:
    print(f"Answer: {data['answer']}")

# Create QA
response = requests.post(
    "http://localhost:8000/qa",
    json={
        "question": "What is VLAN?",
        "answer": "A VLAN is a Virtual Local Area Network..."
    }
)
print(response.json())

# Query text chunks
response = requests.get(
    "http://localhost:8000/chunks/query",
    params={"query": "network routing", "limit": 5}
)
results = response.json()
for chunk in results["results"]:
    print(f"[{chunk['similarity']:.2f}] {chunk['text'][:100]}...")
```

### JavaScript

```javascript
// Query QA
fetch("http://localhost:8000/qa/query?text=What%20is%20OSPF&threshold=0.7")
  .then((res) => res.json())
  .then((data) => {
    if (data.found) {
      console.log("Answer:", data.answer);
    }
  });

// Create QA
fetch("http://localhost:8000/qa", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    question: "What is VLAN?",
    answer: "A VLAN is a Virtual Local Area Network...",
  }),
})
  .then((res) => res.json())
  .then((data) => console.log(data));

// Query text chunks
fetch("http://localhost:8000/chunks/query?query=network%20routing&limit=5")
  .then((res) => res.json())
  .then((data) => {
    data.results.forEach((chunk) => {
      console.log(
        `[${chunk.similarity.toFixed(2)}] ${chunk.text.substring(0, 100)}...`
      );
    });
  });
```

### cURL

```bash
# Query QA
curl "http://localhost:8000/qa/query?text=What%20is%20OSPF&threshold=0.7"

# Get QA by ID
curl "http://localhost:8000/qa/1"

# Create QA
curl -X POST "http://localhost:8000/qa" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is VLAN?",
    "answer": "A VLAN is a Virtual Local Area Network..."
  }'

# Query text chunks
curl "http://localhost:8000/chunks/query?query=network%20routing&limit=5"
```

## API Documentation

Interactive API documentation is available when the server is running:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide interactive testing and detailed schema information.

## Troubleshooting

**API won't start**:

- Check if port 8000 is available
- Ensure Docker database is running: `docker ps`
- Verify database connection in `.env` file

**No results from queries**:

- Check if data exists in the database
- Lower the similarity threshold
- Ensure embeddings are generated (automatic for new records)

**Slow responses**:

- First query may be slow (model loading)
- Subsequent queries should be faster
- Consider adding more database indexes for large datasets

## Support

For issues or questions, please refer to the project documentation or contact the development team.
