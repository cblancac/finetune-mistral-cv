import os

# Load environment variables or set defaults for development/testing
RUNPOD_BASE_URL = os.getenv(
    "RUNPOD_BASE_URL",
    "https://u9odeh9fcwpjzp-8000.proxy.runpod.net/v1"
)

API_KEY = os.getenv("API_KEY", "EMPTY")

MODEL_NAME = "mistral-cv-merged-final"
TOKENIZER_PATH = "checkpoints/mistral-cv-merged-final"

MAX_INPUT_TOKENS = 1524
MAX_OUTPUT_TOKENS = 888
TEMPERATURE = 0.0

# Schema file location (for easier updates)
SCHEMA_FILE = os.path.join(os.path.dirname(__file__), "schemas.py")