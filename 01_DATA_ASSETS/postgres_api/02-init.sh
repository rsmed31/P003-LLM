#!/bin/bash
set -e

echo "Running database initialization (init.sh)..."

# Wait for PostgreSQL via Unix socket (since TCP isn't ready yet)
until pg_isready -U "$POSTGRES_USER" -h /var/run/postgresql >/dev/null 2>&1; do
  echo "Waiting for PostgreSQL to be ready (via socket)..."
  sleep 2
done

echo "Running qa_prefill.py..."
python3 /db/qa_prefill.py || echo "⚠️ Prefill script failed, continuing..."
