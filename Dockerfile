# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first to leverage caching
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy only necessary model files
COPY checkpoints/mistral-cv-merged-final checkpoints/mistral-cv-merged-final

# Copy codebase
COPY src src

# Expose port
EXPOSE 8000

# Run the API with FastAPI (production)
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
