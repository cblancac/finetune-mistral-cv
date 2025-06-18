from peft import AutoPeftModelForCausalLM
import torch

MODEL_DIR = "workspace/finetune-mistral-cv/checkpoints/mistral-cv-finetuned" # dir original LoRA checkpoints
MERGED_MODEL_DIR = "workspace/finetune-mistral-cv/checkpoints/mistral-cv-merged-final"

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

# Guarda también el tokenizer original
model.tokenizer.save_pretrained(MERGED_MODEL_DIR)

print("✅ Model fully merged and saved.")
