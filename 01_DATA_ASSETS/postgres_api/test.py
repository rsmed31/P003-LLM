import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer

conn = psycopg2.connect(
    dbname="postgres",
    user="postgres",
    password="postgres",
    host="localhost",
    port="5432"
)

model = SentenceTransformer("all-MiniLM-L6-v2")

query = "Why should OSPF Area 1 be configured as an NSSA instead of a standard stub area when redistributing EIGRP routes?"
query_emb = model.encode(query)

# SQL-Abfrage mit Ähnlichkeitsvergleich
with conn.cursor() as cur:
    cur.execute("""
        SELECT source, chunk_index, text,
               1 - (embedding <=> %s::vector) AS similarity
        FROM text_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT 5;
    """, (query_emb.tolist(), query_emb.tolist()))

    results = cur.fetchall()

for source, idx, text, sim in results:
    print(f"\n[{source} | Chunk {idx} | Similarity {sim:.3f}]")
    print(text[:300], "...")
