"""
REST API endpoint for chunk retrieval.
Usage: GET /api/retrieve?query=<text>&number=<k>
Returns: { "chunks": ["chunk1", "chunk2", ...] }
"""

from flask import Flask, request, jsonify
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from retrieval import RetrievalOrchestrator

# Global orchestrator instance
_ORCHESTRATOR = None

def init_retrieval_service(chunks: list = None):
    """Initialize the retrieval orchestrator with chunks."""
    global _ORCHESTRATOR
    _ORCHESTRATOR = RetrievalOrchestrator()
    if chunks:
        _ORCHESTRATOR.add_chunks(chunks)
    return _ORCHESTRATOR

def get_chunks_handler():
    """
    Flask route handler for GET /api/retrieve
    Query params: ?query=<text>&number=<int>
    Returns: JSON with chunks list: { "chunks": [...] }
    """
    if _ORCHESTRATOR is None:
        return jsonify({"error": "Retrieval service not initialized"}), 500
    
    # Extract query parameters
    query = request.args.get('query', '').strip()
    try:
        k = int(request.args.get('number', 5))
    except (ValueError, TypeError):
        k = 5
    
    if not query:
        return jsonify({"error": "Missing 'query' parameter"}), 400
    
    # Retrieve chunks
    results = _ORCHESTRATOR.retrieve_chunks(query, k)
    
    # Return only chunks list (matching external service format)
    chunks = [r["chunk"] for r in results]
    
    return jsonify({
        "chunks": chunks
    }), 200

# Flask app setup
app = Flask(__name__)

@app.route('/api/retrieve', methods=['GET'])
def retrieve_endpoint():
    return get_chunks_handler()

if __name__ == '__main__':
    # Example: Initialize with sample chunks
    sample_chunks = [
        "router ospf 1\n router-id 1.1.1.1\n network 10.0.0.0 0.0.0.255 area 0",
        "interface GigabitEthernet0/0\n ip address 192.168.1.1 255.255.255.0\n no shutdown",
        "router bgp 65000\n neighbor 192.168.1.2 remote-as 65000\n network 10.0.0.0 mask 255.255.255.0"
    ]
    init_retrieval_service(sample_chunks)
    
    print("Starting retrieval API on http://0.0.0.0:5000")
    print("Example: GET http://localhost:5000/api/retrieve?query=bgp%20neighbor&number=3")
    print("Response format: { \"chunks\": [...] }")
    app.run(host='0.0.0.0', port=5000, debug=True)
