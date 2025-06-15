#!/usr/bin/env python3
# training/finetune_mistral_cv.py

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    AutoConfig,
    BitsAndBytesConfig
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, PeftModel

# --- CONFIGURACIÓN ---
MODEL_ID       = "mistralai/Mistral-7B-Instruct-v0.2"
DATASET_PATH   = "./dataset/dataset.jsonl"
OUTPUT_DIR     = "./checkpoints/mistral-cv-finetuned"
MERGED_DIR     = "./checkpoints/mistral-cv-merged"
MAX_SEQ_LENGTH = 2048
BATCH_SIZE     = 2
GRAD_ACCUM     = 4
EPOCHS         = 3
LEARNING_RATE  = 2e-4

config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
config.init_device = "cuda"
config.parallelization_style = "none"

# --- TOKENIZADOR Y MODELO EN 4-BIT ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print("parallelization_style =", config.parallelization_style)  # <-- TEMPORAL


model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    config=config,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True,
    torch_dtype=torch.float16,
)

# --- CARGAR DATASET JSONL ---
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# --- COLLATOR PARA SOLO COMPLETION ---
collator = DataCollatorForCompletionOnlyLM(
    tokenizer=tokenizer,
    response_template="completion",
    instruction_template="prompt",
    mlm=False
)

# --- CONFIGURACIÓN DE LoRA ---
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# --- ARGUMENTOS DE ENTRENAMIENTO ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    warmup_steps=20,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    save_total_limit=2,
    remove_unused_columns=False,
    report_to="none"
)

# --- ENTRENADOR ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    dataset_text_field="prompt",
    max_seq_length=MAX_SEQ_LENGTH,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=lora_config,
    data_collator=collator,
    formatting_func=lambda ex: f"{ex['prompt']}\n{ex['completion']}"
)

# --- EJECUTAR ENTRENAMIENTO Y MERGE ---
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.train()
    print(f"✅ Fine-tuning completado. Checkpoint guardado en: {OUTPUT_DIR}")

    print("🔁 Cargando modelo con LoRA para merge final...")
    merged_model = PeftModel.from_pretrained(model, OUTPUT_DIR)
    merged_model = merged_model.merge_and_unload()

    print(f"💾 Guardando modelo mergeado en: {MERGED_DIR}")
    os.makedirs(MERGED_DIR, exist_ok=True)
    merged_model.save_pretrained(MERGED_DIR)
    tokenizer.save_pretrained(MERGED_DIR)
    print("✅ Modelo mergeado y guardado listo para inferencia.")
