import openai
import json
import time
import logging

from transformers import AutoTokenizer

# Configuración básica de logging con timestamps
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# NEW: load the same tokenizer you used for fine-tuning / merge
TOKENIZER = AutoTokenizer.from_pretrained(
    "checkpoints/mistral-cv-merged-final",        # local folder or HF repo
    trust_remote_code=True,
)
MAX_INPUT_TOKENS = 1524

def truncate_text(text: str) -> str:  # ← NEW
    """
    Keep only the first MAX_INPUT_TOKENS tokens of `text`
    (counted with the Mistral tokenizer) and decode back to UTF-8.
    """
    ids = TOKENIZER(text, add_special_tokens=False)["input_ids"]
    if len(ids) > MAX_INPUT_TOKENS:
        ids = ids[:MAX_INPUT_TOKENS]
    return TOKENIZER.decode(
        ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

# 1️⃣ Medir tiempo de inicialización del cliente
client_init_start = time.time()
logger.info("Initializing OpenAI client for vLLM...")
client = openai.OpenAI(
    #base_url="http://localhost:8000/v1",
    base_url="https://u9odeh9fcwpjzp-8000.proxy.runpod.net/v1",
    api_key="EMPTY"
)
client_init_end = time.time()
logger.info(f"Client initialized in {client_init_end - client_init_start:.3f}s")

# El esquema que has definido
SCHEMA = json.dumps({
    "certifications": "",
    "contact_detail": {
        "age": "",
        "email": "",
        "home_city": "",
        "mobile": "",
        "name": ""
    },
    "education": [{"degree": "", "degree_level": "", "end_date": "", "school_name": "", "start_date": ""}],
    "gender": "",
    "industry": "",
    "skills": [],
    "software_tools": [],
    "work_abroad": "",
    "work_experience": [{"company": "", "end_date": "", "position": "", "start_date": ""}]
}, separators=(",", ":"))  

# Prompt del sistema
SYSTEM_PROMPT = (
    "You are an API that extracts structured JSON from resumes.\n"
    "Return *only* valid JSON matching exactly this schema:\n"
    f"{SCHEMA}"
)

def extract_cv(cv_text):
    # 2️⃣ Antes de la llamada a la inferencia
    logger.info("Preparing to send chat completion request...")

    # NEW: truncate before the call
    tokenizer_start = time.time()
    cv_text = truncate_text(cv_text)
    tokenizer_end = time.time()
    logger.info(f"Tokenization takes {tokenizer_end - tokenizer_start:.3f}s")

    request_start = time.time()
    response = client.chat.completions.create(
        model="mistral-cv-merged-final",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": cv_text}
        ],
        max_tokens=888,
        temperature=0.0,
    )
    
    # 3️⃣ Después de recibir la respuesta
    request_end = time.time()
    logger.info(f"Chat completion request took {request_end - request_start:.3f}s")

    return response.choices[0].message.content

from concurrent.futures import ThreadPoolExecutor, as_completed

def extract_cv_wrapper(cv_text):
    return extract_cv(cv_text)

if __name__ == "__main__":

    import os

    DATASET_DIR = "dataset_txt"
    PARALLEL_JOBS = 8
    CVs_TOTAL = 400
    BATCH_SIZE = PARALLEL_JOBS  # 8 per batch
    LOOPS = CVs_TOTAL // BATCH_SIZE

    logger.info(f"📥 Loading {CVs_TOTAL} CVs from {DATASET_DIR}...")

    load_start = time.time()
    # Load all text files from the directory
    all_cvs = []
    filenames = sorted(os.listdir(DATASET_DIR))[:CVs_TOTAL]
    for fname in filenames:
        path = os.path.join(DATASET_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            all_cvs.append(f.read())
    load_end = time.time()

    logger.info(f"✅ Loaded {len(all_cvs)} CVs in {load_end - load_start:.2f} seconds.")

    # Now, split into batches
    batches = [all_cvs[i:i + BATCH_SIZE] for i in range(0, CVs_TOTAL, BATCH_SIZE)]

    global_start = time.time()
    for loop, batch in enumerate(batches, start=1):
        logger.info(f"===== RUN {loop}/{LOOPS} =====")
        start = time.time()
        results = []

        with ThreadPoolExecutor(max_workers=PARALLEL_JOBS) as ex:
            futures = [ex.submit(extract_cv_wrapper, cv) for cv in batch]
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Error during CV extraction: {e}")

        dur = time.time() - start
        logger.info(f"Run {loop} finished in {dur:.2f} s "
                    f"(processed {len(results)} CVs)")

        # (Optional) Print the first few outputs of each batch
        for idx, res in enumerate(results[:2], start=1):
            print(f"\n=== Run {loop} · Result {idx} ===\n{res}")

    total_duration = time.time() - global_start
    logger.info(f"🏁 All batches processed in {total_duration:.2f} seconds")
