#!/bin/bash
set -euo pipefail

echo "Running database initialization (init.sh)..."

# Wait for PostgreSQL via Unix socket (since TCP isn't ready yet)
until pg_isready -U "$POSTGRES_USER" -h /var/run/postgresql >/dev/null 2>&1; do
  echo "Waiting for PostgreSQL to be ready (via socket)..."
  sleep 2
done

echo "Running qa_prefill.py..."
python3 /db/qa_prefill.py || echo "⚠️ Prefill script failed, continuing..."

echo "[02-init.sh] Post-init script running"

# Example: ensure extensions (already in 01-init.sql) – safe rechecks
psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "CREATE EXTENSION IF NOT EXISTS pg_trgm;"
psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "CREATE EXTENSION IF NOT EXISTS vector;"

echo "[02-init.sh] Completed"
