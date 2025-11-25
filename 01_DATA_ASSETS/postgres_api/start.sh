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

python3 ./manuals_raw/pdf_extractor.py || echo "⚠️ pdf_extractor.py failed"
python3 ./manuals_raw/chunk_creation.py || echo "⚠️ chunk_creation.py failed"
python3 ./manuals_raw/embedding_creation.py || echo "⚠️ embedding_creation.py failed"
#python3 ./test.py || echo "⚠️ test.py failed"

set -e

echo "✅ All post-startup scripts executed."

# --- Keep Postgres and app alive ---
echo "⏳ Waiting for main processes to exit..."
wait $PG_PID $APP_PID
