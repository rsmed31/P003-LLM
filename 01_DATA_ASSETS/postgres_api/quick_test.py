import requests
import json

print("="*70)
print(" QUICK API TEST - Postgres QA API")
print("="*70)

# Test 1: Health Check
print("\n1. Health Check (GET /)")
try:
    r = requests.get("http://localhost:8000/")
    print(f"   Status: {r.status_code}")
    print(f"   Response: {json.dumps(r.json(), indent=4)}")
except Exception as e:
    print(f"   Error: {e}")

# Test 2: Create a QA record  
print("\n2. Create QA Record (POST /qa)")
try:
    payload = {
        "question": "What is OSPF routing protocol?",
        "answer": "OSPF (Open Shortest Path First) is an interior gateway protocol (IGP) for routing within an autonomous system."
    }
    r = requests.post("http://localhost:8000/qa", json=payload)
    print(f"   Status: {r.status_code}")
    print(f"   Response: {json.dumps(r.json(), indent=4)}")
except Exception as e:
    print(f"   Error: {e}")

# Test 3: Query the record we just created
print("\n3. Get QA by ID (GET /qa/1)")
try:
    r = requests.get("http://localhost:8000/qa/1")
    print(f"   Status: {r.status_code}")
    if r.status_code == 200:
        print(f"   Response: {json.dumps(r.json(), indent=4)}")
    else:
        print(f"   Response: {r.text}")
except Exception as e:
    print(f"   Error: {e}")

# Test 4: Query by text similarity
print("\n4. Query by Text (GET /qa/query)")
try:
    r = requests.get("http://localhost:8000/qa/query", params={"text": "OSPF", "threshold": 0.5})
    print(f"   Status: {r.status_code}")
    print(f"   Response: {json.dumps(r.json(), indent=4)}")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "="*70)
print(" TEST COMPLETED")
print("="*70)
