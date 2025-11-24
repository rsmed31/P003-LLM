#!/bin/bash
set -e

echo "🚀 Starting PostgreSQL via start.sh..."

# Start PostgreSQL in background
docker-entrypoint.sh postgres &
PG_PID=$!

# --- Wait until PostgreSQL is ready ---
echo "⏳ Waiting for PostgreSQL to be ready..."
until pg_isready -U "$POSTGRES_USER" -h "localhost" -d "$POSTGRES_DB" >/dev/null 2>&1; do
  sleep 2
done

# --- Start your Python app in background ---
echo "▶️ Starting Python app..."
python3 app.py &
APP_PID=$!

# --- Run data preparation scripts ---
echo "🧾 Running document processing scripts..."
set +e  # allow individual scripts to fail without stopping container

python3 ./manuals_raw/PDFExtractorV2.py || echo "⚠️ PDFExtractorV2.py failed"
python3 ./manuals_raw/ChunkCreationV4.py || echo "⚠️ ChunkCreationV4.py failed"
python3 ./manuals_raw/EmbeddingCreation.py || echo "⚠️ EmbeddingCreation.py failed"
#python3 ./test.py || echo "⚠️ test.py failed"

set -e

echo "✅ All post-startup scripts executed."

# --- Keep Postgres and app alive ---
echo "⏳ Waiting for main processes to exit..."
wait $PG_PID $APP_PID
