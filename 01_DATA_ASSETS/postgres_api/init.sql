CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS text_chunks (
    id SERIAL PRIMARY KEY,
    source TEXT,
    chunk_index INT,
    text TEXT,
    embedding VECTOR(384)
);

CREATE INDEX IF NOT EXISTS text_chunks_embedding_idx
ON text_chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 1024);

CREATE TABLE IF NOT EXISTS qa (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    lastUpdated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    embedding VECTOR(384)
);

ALTER TABLE qa ADD CONSTRAINT qa_question_unique UNIQUE (question);


CREATE INDEX IF NOT EXISTS qa_embedding_idx
ON qa
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);