FROM pgvector/pgvector:pg18-trixie

RUN apt-get update && apt-get install -y tini python3 python3-pip && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --break-system-packages torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --break-system-packages sentence-transformers


# Install Python packages; allow system installs since we're in a container
RUN pip install --no-cache-dir --break-system-packages \
    fastapi>=0.104.0 \
    uvicorn>=0.24.0 \
    pydantic>=2.0.0 \
    psycopg2-binary>=2.9.9 \
    pgvector>=0.2.0 \
    python-dotenv>=1.0.0 \
    pytest>=7.4.0 \
    httpx>=0.25.0 \
    requests>=2.31.0 PyMuPDF frontend



WORKDIR /db
COPY . /db
COPY init.sql /docker-entrypoint-initdb.d/init.sql

EXPOSE 5432

ENTRYPOINT ["/usr/bin/tini", "--"]

# COPY start.sh /start.sh
RUN chmod +x ./start.sh
CMD ["./start.sh"]