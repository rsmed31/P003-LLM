#!/bin/bash
# Startup script for Postgres QA API

echo "=== Postgres QA API Setup ==="

# Check if .env exists
if [ ! -f .env ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo "✓ .env file created. Please update it with your settings."
else
    echo "✓ .env file already exists"
fi

# Install dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Check if Docker is running
echo ""
echo "Checking Docker status..."
if ! docker info > /dev/null 2>&1; then
    echo "⚠ Docker is not running. Please start Docker to run the database."
else
    echo "✓ Docker is running"
    
    # Start database
    echo ""
    echo "Starting PostgreSQL database..."
    docker-compose up -d
    
    echo "Waiting for database to be ready..."
    sleep 5
    
    echo "✓ Database is starting"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start the API:"
echo "  uvicorn app:app --reload --port 8000"
echo ""
echo "Or:"
echo "  python app.py"
echo ""
echo "API will be available at: http://localhost:8000"
echo "API documentation: http://localhost:8000/docs"
