from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

# Define las rutas absolutas
MODEL_DIR = "checkpoints/mistral-cv-finetuned"
BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MERGED_MODEL_DIR = "checkpoints/mistral-cv-merged-final"

print("Loading PEFT (LoRA) model...")
model = AutoPeftModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    torch_dtype=torch.float16
)

print("Merging weights...")
merged_model = model.merge_and_unload()

print(f"Saving merged model at {MERGED_MODEL_DIR}...")
merged_model.save_pretrained(MERGED_MODEL_DIR, safe_serialization=True, max_shard_size="2GB")

# Carga y guarda explícitamente el tokenizer original del modelo base
print("Saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.save_pretrained(MERGED_MODEL_DIR)

print("✅ Model fully merged and tokenizer saved.")

