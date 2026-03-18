# Use multi-stage build or keep it simple for both apps
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy the rest of the app
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV API_URL=http://localhost:5000/api

# Open ports
EXPOSE 5000
EXPOSE 8501

# Entry point (we can override this in docker-compose)
CMD ["python", "run.py"]
