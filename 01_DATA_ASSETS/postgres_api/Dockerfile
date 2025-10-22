FROM pgvector/pgvector:pg18-trixie

RUN apt-get update && apt-get install -y tini && rm -rf /var/lib/apt/lists/*

# Any .sql or .sh here runs automatically at first container startup
COPY init.sql /docker-entrypoint-initdb.d/init.sql

# Expose PostgreSQL port
EXPOSE 5432

ENTRYPOINT ["/usr/bin/tini", "--"]

# Let the official entrypoint handle postgres startup
CMD ["docker-entrypoint.sh", "postgres"]
