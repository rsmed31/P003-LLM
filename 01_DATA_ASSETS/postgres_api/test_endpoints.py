"""
Quick test script to verify API endpoints are working.
Run this after starting the API server.
"""
import requests
import json

API_URL = "http://localhost:8000"


def print_response(title, response):
    """Pretty print response."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Status Code: {response.status_code}")
    print(f"Response:")
    try:
        print(json.dumps(response.json(), indent=2))
    except:
        print(response.text)
    print()


def main():
    print("\n" + "="*60)
    print("Testing Postgres QA API Endpoints")
    print("="*60)
    
    # Test 1: Health Check
    print("\nTest 1: Health Check")
    try:
        response = requests.get(f"{API_URL}/")
        print_response("GET /", response)
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 2: Query QA with text
    print("\nTest 2: Query QA with text")
    try:
        response = requests.get(
            f"{API_URL}/qa/query",
            params={"text": "What is OSPF?", "threshold": 0.7}
        )
        print_response("GET /qa/query?text=What is OSPF?", response)
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 3: Query QA with no match
    print("\nTest 3: Query QA with no match")
    try:
        response = requests.get(
            f"{API_URL}/qa/query",
            params={"text": "Some random text that will not match", "threshold": 0.9}
        )
        print_response("GET /qa/query (no match)", response)
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 4: Get QA by ID
    print("\nTest 4: Get QA by ID (assuming ID 1 exists)")
    try:
        response = requests.get(f"{API_URL}/qa/1")
        print_response("GET /qa/1", response)
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 5: Get QA by non-existent ID
    print("\nTest 5: Get QA by non-existent ID")
    try:
        response = requests.get(f"{API_URL}/qa/999999")
        print_response("GET /qa/999999", response)
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 6: Create QA record (stub)
    print("\nTest 6: Create QA record (stub)")
    try:
        response = requests.post(
            f"{API_URL}/qa",
            json={
                "question": "What is BGP?",
                "answer": "BGP (Border Gateway Protocol) is the routing protocol used to exchange routing information between autonomous systems."
            }
        )
        print_response("POST /qa", response)
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 7: Invalid query - missing text parameter
    print("\nTest 7: Invalid query - missing text parameter")
    try:
        response = requests.get(f"{API_URL}/qa/query")
        print_response("GET /qa/query (no text param)", response)
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 8: Invalid query - invalid threshold
    print("\nTest 8: Invalid query - invalid threshold")
    try:
        response = requests.get(
            f"{API_URL}/qa/query",
            params={"text": "test", "threshold": 1.5}
        )
        print_response("GET /qa/query (invalid threshold)", response)
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 9: Invalid POST - empty question
    print("\nTest 9: Invalid POST - empty question")
    try:
        response = requests.post(
            f"{API_URL}/qa",
            json={"question": "", "answer": "Some answer"}
        )
        print_response("POST /qa (invalid)", response)
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()
