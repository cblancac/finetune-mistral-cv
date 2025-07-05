# src/config.py
import os
from pathlib import Path

# Model configuration
CHECKPOINT_DIR = Path("checkpoints/mistral-cv-merged-final")

MAX_INPUT_TOKENS = 1524
MAX_OUTPUT_TOKENS = 888
TEMPERATURE = 0.0

# AWS S3 configuration
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
MODEL_S3_PATH = os.getenv(
    "MODEL_S3_PATH", 
    "s3://test-s3-putobject-presigned/models/mistral-cv-merged-final.zip"
)
