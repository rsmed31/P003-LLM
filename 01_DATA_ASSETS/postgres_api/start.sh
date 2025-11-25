#!/bin/bash
set -euo pipefail

echo "[start.sh] launching PostgreSQL cluster + API (pgvector)"
docker-entrypoint.sh postgres &
PG_PID=$!

echo "[start.sh] waiting for PostgreSQL readiness..."
for i in {1..40}; do
  if pg_isready -U "${POSTGRES_USER:-postgres}" -h localhost -d "${POSTGRES_DB:-postgres}" >/dev/null 2>&1; then
    echo "[start.sh] postgres ready"
    break
  fi
  sleep 2
  if [ $i -eq 40 ]; then
    echo "[start.sh] timeout waiting for postgres"; exit 1
  fi
done

BOOTSTRAP=/db/.bootstrap_done
PDF_CHECKSUM=/db/.pdf_checksum
CURRENT_PDF_CHECKSUM=$(find manuals_raw/docs -type f -name "*.pdf" -exec md5sum {} \; 2>/dev/null | sort | md5sum | awk '{print $1}' || echo "none")

# Check if we need to re-index (new PDFs added or first run)
NEEDS_REINDEX=false
if [ ! -f "$BOOTSTRAP" ]; then
  echo "[start.sh] first run - will index all PDFs"
  NEEDS_REINDEX=true
elif [ ! -f "$PDF_CHECKSUM" ] || [ "$(cat $PDF_CHECKSUM)" != "$CURRENT_PDF_CHECKSUM" ]; then
  echo "[start.sh] PDF files changed - will re-index"
  NEEDS_REINDEX=true
else
  echo "[start.sh] no PDF changes detected"
fi

if [ "$NEEDS_REINDEX" = true ]; then
  echo "[start.sh] running indexing pipeline"
  set +e
  [ -f manuals_raw/PDFExtractorV2.py ] && python3 manuals_raw/PDFExtractorV2.py || echo "[warn] PDFExtractorV2.py skipped"
  # Use SmartChunkCreation instead of ChunkCreationV4 for better chunking
  [ -f manuals_raw/SmartChunkCreation.py ] && python3 manuals_raw/SmartChunkCreation.py || echo "[warn] SmartChunkCreation.py skipped"
  [ -f manuals_raw/EmbeddingCreation.py ] && python3 manuals_raw/EmbeddingCreation.py || echo "[warn] EmbeddingCreation.py skipped"
  set -e
  echo "$CURRENT_PDF_CHECKSUM" > "$PDF_CHECKSUM"
  touch "$BOOTSTRAP"
fi

if [ -f app.py ]; then
  API_HOST="${API_HOST:-0.0.0.0}"
  API_PORT="${API_PORT:-8000}"
  echo "[start.sh] starting uvicorn app:app on ${API_HOST}:${API_PORT}"
  python3 -m uvicorn app:app --host "${API_HOST}" --port "${API_PORT}" --workers 1 &
  APP_PID=$!
else
  echo "[start.sh] app.py missing; API not started"
  APP_PID=
fi

trap 'echo "[start.sh] shutdown signal"; kill $PG_PID >/dev/null 2>&1 || true; [ -n "${APP_PID:-}" ] && kill $APP_PID >/dev/null 2>&1 || true; exit 0' INT TERM

echo "[start.sh] running"
if [ -n "${APP_PID:-}" ]; then
  wait $PG_PID $APP_PID
else
  wait $PG_PID
fi
