import requests
import json

# API configuration
API_BASE_URL = "http://localhost:8000"

# Test query
query = "Why should OSPF Area 1 be configured as an NSSA instead of a standard stub area when redistributing EIGRP routes?"

print("=" * 80)
print("Testing Text Chunks Query API")
print("=" * 80)
print(f"Query: {query}")
print()

try:
    # Query the API
    response = requests.post(
        f"{API_BASE_URL}/chunks/query",
        json={
            "query": query,
            "limit": 5
        }
    )
    
    print(f"Status Code: {response.status_code}")
    print()
    
    if response.status_code == 200:
        result = response.json()
        
        if result.get("found"):
            print(f"✓ Found {result.get('count')} matching chunks\n")
            
            for chunk in result.get("results", []):
                source = chunk.get("source", "")
                chunk_index = chunk.get("chunk_index", 0)
                text = chunk.get("text", "")
                similarity = chunk.get("similarity", 0.0)
                
                print(f"[{source} | Chunk {chunk_index} | Similarity {similarity:.3f}]")
                print(text[:300], "...")
                print()
        else:
            print("No matching chunks found")
    else:
        print(f"Error: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("❌ Error: Could not connect to API at", API_BASE_URL)
    print("Make sure the API is running with: uvicorn app:app --reload --port 8000")
except Exception as e:
    print(f"❌ Error: {e}")

print("=" * 80)

