FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl unzip git awscli && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Set defaults
ENV AWS_DEFAULT_REGION=us-east-1
ENV MODEL_S3_PATH="s3://test-s3-putobject-presigned/models/mistral-cv-merged-final.zip"

# Build args (include session token)
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_SESSION_TOKEN

# Set env vars at build time
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ENV AWS_SESSION_TOKEN=$AWS_SESSION_TOKEN

RUN mkdir -p checkpoints && \
    aws s3 cp "$MODEL_S3_PATH" /tmp/model.zip && \
    unzip /tmp/model.zip -d checkpoints/ && \
    rm /tmp/model.zip

COPY src src

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
