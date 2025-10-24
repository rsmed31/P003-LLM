"""Unit tests for the Postgres QA API."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app import app
from models import QAResponse


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_db_pool():
    """Mock the database pool to avoid actual database connections."""
    with patch('db.DatabasePool.initialize') as mock_init, \
         patch('db.DatabasePool.close_all') as mock_close:
        yield mock_init, mock_close


class TestHealthEndpoint:
    """Tests for the health check endpoint."""
    
    def test_health_check(self, client):
        """Test the health check endpoint returns 200."""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert "service" in response.json()


class TestQueryQAEndpoint:
    """Tests for GET /qa/query endpoint."""
    
    def test_query_missing_text_parameter(self, client, mock_db_pool):
        """Test query without text parameter returns 422."""
        response = client.get("/qa/query")
        assert response.status_code == 422
    
    def test_query_empty_text_parameter(self, client, mock_db_pool):
        """Test query with empty text parameter returns 422."""
        response = client.get("/qa/query?text=")
        assert response.status_code == 422
    
    def test_query_invalid_threshold(self, client, mock_db_pool):
        """Test query with invalid threshold returns 422."""
        response = client.get("/qa/query?text=test&threshold=1.5")
        assert response.status_code == 422
        
        response = client.get("/qa/query?text=test&threshold=-0.1")
        assert response.status_code == 422
    
    @patch('app.query_qa_by_text_similarity')
    def test_query_with_match(self, mock_query, client, mock_db_pool):
        """Test query that finds a match."""
        mock_query.return_value = {
            'id': 1,
            'question': 'What is OSPF?',
            'answer': 'OSPF is a routing protocol',
            'score': 0.85
        }
        
        response = client.get("/qa/query?text=What%20is%20OSPF")
        assert response.status_code == 200
        data = response.json()
        assert data['found'] is True
        assert data['data']['id'] == 1
        assert data['data']['score'] == 0.85
    
    @patch('app.query_qa_by_text_similarity')
    def test_query_without_match(self, mock_query, client, mock_db_pool):
        """Test query that doesn't find a match."""
        mock_query.return_value = None
        
        response = client.get("/qa/query?text=unknown%20query")
        assert response.status_code == 200
        data = response.json()
        assert data['found'] is False
        assert data['data'] is None
        assert 'No match found' in data['message']
    
    @patch('app.query_qa_by_text_similarity')
    def test_query_database_error(self, mock_query, client, mock_db_pool):
        """Test query when database error occurs."""
        mock_query.side_effect = Exception("Database connection failed")
        
        response = client.get("/qa/query?text=test")
        assert response.status_code == 500
        assert 'Database query failed' in response.json()['detail']


class TestGetQAByIdEndpoint:
    """Tests for GET /qa/{id} endpoint."""
    
    def test_get_invalid_id(self, client, mock_db_pool):
        """Test getting QA with invalid ID returns 422."""
        response = client.get("/qa/0")
        assert response.status_code == 422
        
        response = client.get("/qa/-1")
        assert response.status_code == 422
        
        response = client.get("/qa/invalid")
        assert response.status_code == 422
    
    @patch('app.query_qa_by_id')
    def test_get_existing_record(self, mock_query, client, mock_db_pool):
        """Test getting an existing QA record."""
        mock_query.return_value = {
            'id': 1,
            'question': 'What is VLAN?',
            'answer': 'VLAN is a virtual LAN',
            'lastUpdated': '2025-10-22T10:00:00'
        }
        
        response = client.get("/qa/1")
        assert response.status_code == 200
        data = response.json()
        assert data['id'] == 1
        assert data['question'] == 'What is VLAN?'
    
    @patch('app.query_qa_by_id')
    def test_get_nonexistent_record(self, mock_query, client, mock_db_pool):
        """Test getting a non-existent QA record returns 404."""
        mock_query.return_value = None
        
        response = client.get("/qa/999")
        assert response.status_code == 404
        assert 'not found' in response.json()['detail']
    
    @patch('app.query_qa_by_id')
    def test_get_database_error(self, mock_query, client, mock_db_pool):
        """Test getting QA when database error occurs."""
        mock_query.side_effect = Exception("Database connection failed")
        
        response = client.get("/qa/1")
        assert response.status_code == 500


class TestCreateQAEndpoint:
    """Tests for POST /qa endpoint."""
    
    def test_create_missing_fields(self, client, mock_db_pool):
        """Test creating QA without required fields returns 422."""
        response = client.post("/qa", json={})
        assert response.status_code == 422
        
        response = client.post("/qa", json={"question": "test"})
        assert response.status_code == 422
    
    def test_create_empty_fields(self, client, mock_db_pool):
        """Test creating QA with empty fields returns 422."""
        response = client.post("/qa", json={"question": "", "answer": "test"})
        assert response.status_code == 422
        
        response = client.post("/qa", json={"question": "test", "answer": ""})
        assert response.status_code == 422
    
    def test_create_invalid_embedding_dimension(self, client, mock_db_pool):
        """Test creating QA with wrong embedding dimension returns 422."""
        response = client.post("/qa", json={
            "question": "test",
            "answer": "test answer",
            "embedding": [0.1, 0.2, 0.3]  # Wrong dimension
        })
        assert response.status_code == 422
    
    def test_create_valid_payload(self, client, mock_db_pool):
        """Test creating QA with valid payload (stub implementation)."""
        response = client.post("/qa", json={
            "question": "What is BGP?",
            "answer": "BGP is the Border Gateway Protocol"
        })
        assert response.status_code == 202
        data = response.json()
        assert data['status'] == 'not_implemented'
        assert 'stub' in data['message'].lower()
    
    def test_create_with_embedding(self, client, mock_db_pool):
        """Test creating QA with valid embedding vector."""
        embedding = [0.1] * 384  # 384 dimensions
        response = client.post("/qa", json={
            "question": "What is RIP?",
            "answer": "RIP is a routing protocol",
            "embedding": embedding
        })
        assert response.status_code == 202
        data = response.json()
        assert data['status'] == 'not_implemented'
    
    def test_create_too_long_question(self, client, mock_db_pool):
        """Test creating QA with too long question returns 422."""
        long_question = "x" * 2001  # Exceeds max_length
        response = client.post("/qa", json={
            "question": long_question,
            "answer": "test answer"
        })
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
