-- Migration script to add embedding column to existing qa table
-- Run this if you already have data in your qa table

-- Enable pgvector extension if not already enabled
CREATE EXTENSION IF NOT EXISTS vector;

-- Add embedding column to qa table if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'qa' 
        AND column_name = 'embedding'
    ) THEN
        ALTER TABLE qa ADD COLUMN embedding VECTOR(384);
        RAISE NOTICE 'Added embedding column to qa table';
    ELSE
        RAISE NOTICE 'Embedding column already exists';
    END IF;
END $$;

-- Create index on embedding column if it doesn't exist
CREATE INDEX IF NOT EXISTS qa_embedding_idx
ON qa
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Show table structure
\d qa
