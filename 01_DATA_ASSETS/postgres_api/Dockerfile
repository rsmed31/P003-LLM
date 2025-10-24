FROM pgvector/pgvector:pg18-trixie

RUN apt-get update && apt-get install -y lsof tini python3 python3-pip && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --break-system-packages torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --break-system-packages sentence-transformers


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

COPY 01-init.sql /docker-entrypoint-initdb.d/01-init.sql
COPY 02-init.sh /docker-entrypoint-initdb.d/02-init.sh

EXPOSE 5432 8000

ENTRYPOINT ["/usr/bin/tini", "--"]

RUN chmod +x ./start.sh
CMD ["./start.sh"]
