#!/bin/bash
set -e

echo "Starting PostgreSQL via start.sh..."

# Start PostgreSQL in foreground
docker-entrypoint.sh postgres &

PG_PID=$!

# Wait for Postgres to start
sleep 2

echo "Starting Python app..."
python3 app.py &

APP_PID=$!

echo "Running init.sh script..."
# ./manuals_raw/init.sh

# rm -f ./docs/raw_chunks.json

python3 ./manuals_raw/PDFExtractorV2.py
python3 ./manuals_raw/ChunkCreationV4.py
python3 ./manuals_raw/EmbeddingCreation.py
python3 ./test.py

# Wait for both to exit (this will keep container alive)
wait $PG_PID $APP_PID
