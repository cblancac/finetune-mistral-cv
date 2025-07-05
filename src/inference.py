# src/inference.py
import json
import logging
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from config import CHECKPOINT_DIR, MAX_INPUT_TOKENS, MAX_OUTPUT_TOKENS, TEMPERATURE
from schemas import SCHEMA

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

logger.info("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT_DIR,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=MAX_OUTPUT_TOKENS,
    temperature=TEMPERATURE,
    do_sample=False,
)

SYSTEM_PROMPT = (
    "You are an API that extracts structured JSON from resumes.\n"
    "Return *only* valid JSON matching exactly this schema:\n"
    f"{json.dumps(SCHEMA, separators=(',', ':'))}\n\n"
    "Resume text:\n"
)

def truncate_text(text: str) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > MAX_INPUT_TOKENS:
        ids = ids[:MAX_INPUT_TOKENS]
    return tokenizer.decode(ids, skip_special_tokens=True)

def extract_cv(cv_text: str) -> dict:
    prompt = SYSTEM_PROMPT + truncate_text(cv_text)
    start_time = time.time()
    output = generator(prompt, return_full_text=False)[0]['generated_text']
    duration = time.time() - start_time
    logger.info(f"Inference completed in {duration:.2f}s")

    json_start = output.find('{')
    json_end = output.rfind('}') + 1
    json_str = output[json_start:json_end]

    return json.loads(json_str)
