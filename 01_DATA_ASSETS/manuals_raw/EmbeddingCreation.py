import json
import psycopg2
from sentence_transformers import SentenceTransformer

# Database Connection
conn = psycopg2.connect(
    dbname="postgres",
    user="postgres",
    password="postgres",
    host="localhost",
    port="5432"
)
cur = conn.cursor()

# Embedding-Model
model = SentenceTransformer("all-MiniLM-L6-v2")

with open("./docs/raw_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Create and insert Embeddings per Chunk
for chunk in chunks:
    text = chunk["text"]
    embedding = model.encode(text).tolist()
    cur.execute("""
        INSERT INTO text_chunks (source, chunk_index, text, embedding)
        VALUES (%s, %s, %s, %s)
    """, (chunk["source"], chunk["chunk_index"], text, embedding))

conn.commit()
cur.close()
conn.close()

print(f"{len(chunks)} Chunks erfolgreich in PostgreSQL gespeichert.")
