import openai
import json
import logging
import time
from transformers import AutoTokenizer

from config import (
    RUNPOD_BASE_URL,
    API_KEY,
    MODEL_NAME,
    TOKENIZER_PATH,
    MAX_INPUT_TOKENS,
    MAX_OUTPUT_TOKENS,
    TEMPERATURE,
)

# Setup logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Load tokenizer once
logger.info(f"Loading tokenizer from '{TOKENIZER_PATH}'")
TOKENIZER = AutoTokenizer.from_pretrained(
    TOKENIZER_PATH,
    trust_remote_code=True,
)

def load_schema() -> dict:
    """Load JSON schema from schemas.py."""
    try:
        from schemas import SCHEMA
        return SCHEMA
    except ImportError as e:
        logger.error(f"Schema import error: {e}")
        raise

SCHEMA = load_schema()

# Construct system prompt
SYSTEM_PROMPT = (
    "You are an API that extracts structured JSON from resumes.\n"
    "Return *only* valid JSON matching exactly this schema:\n"
    f"{json.dumps(SCHEMA, separators=(',', ':'))}"
)

# Initialize OpenAI client
logger.info("Initializing OpenAI client...")
client = openai.OpenAI(
    base_url=RUNPOD_BASE_URL,
    api_key=API_KEY,
)

def truncate_text(text: str) -> str:
    """
    Truncate text to MAX_INPUT_TOKENS tokens to avoid exceeding model limits.
    """
    ids = TOKENIZER.encode(text, add_special_tokens=False)
    if len(ids) > MAX_INPUT_TOKENS:
        ids = ids[:MAX_INPUT_TOKENS]
    return TOKENIZER.decode(
        ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

def extract_cv(cv_text: str) -> dict:
    """
    Send truncated CV text to RunPod-hosted inference endpoint and parse JSON response.
    """
    truncated_text = truncate_text(cv_text)
    
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": truncated_text}
            ],
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=TEMPERATURE,
        )
        duration = time.time() - start_time
        logger.info(f"Inference completed in {duration:.2f}s")
    except Exception as e:
        logger.error(f"Inference request failed: {e}")
        raise

    # Parse response content into JSON
    try:
        extracted_json = json.loads(response.choices[0].message.content)
        return extracted_json
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {e}")
        raise

