FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY validator.py .
COPY configs/ ./configs/

# Create directory for temporary snapshots
RUN mkdir -p /tmp/snapshots

# Environment variables (can be overridden in docker-compose.yml)
ENV BATFISH_HOST=batfish
ENV BF_NETWORK_NAME=network-test
ENV BF_SNAPSHOT_BASE=/app/configs
ENV PORT=5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Expose port
EXPOSE 5000

# Run the validator
CMD ["python", "validator.py"]
