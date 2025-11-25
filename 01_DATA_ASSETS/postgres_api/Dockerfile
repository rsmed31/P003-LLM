FROM pgvector/pgvector:pg18-trixie

# Install system dependencies including Tesseract OCR for image-based PDFs
RUN apt-get update && apt-get install -y \
    tini \
    python3 \
    python3-pip \
    curl \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --break-system-packages \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --break-system-packages sentence-transformers

RUN pip install --no-cache-dir --break-system-packages \
    fastapi>=0.104.0 \
    uvicorn>=0.24.0 \
    pydantic>=2.0.0 \
    psycopg2-binary>=2.9.9 \
    pgvector>=0.2.0 \
    python-dotenv>=1.0.0 \
    pytest>=7.4.0 \
    httpx>=0.25.0 \
    requests>=2.31.0 \
    PyMuPDF

WORKDIR /db

# Copy ALL application files first
COPY . /db/
COPY 01-init.sql 02-init.sh /docker-entrypoint-initdb.d/
COPY start.sh /db/
RUN sed -i 's/\r$//' /db/start.sh && chmod +x /db/start.sh \
    && sed -i 's/\r$//' /docker-entrypoint-initdb.d/02-init.sh && chmod +x /docker-entrypoint-initdb.d/02-init.sh

# Ensure start.sh present (already copied) and executable; no change to logic
# Optionally install requirements.txt if present (idempotent)
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir --break-system-packages -r requirements.txt || true; fi
EXPOSE 5432 8000
ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["bash","/db/start.sh"]
