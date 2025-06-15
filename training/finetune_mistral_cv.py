#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tuning estilo NER para Mistral-7B-Instruct-v0.2.
-   Percepción SOLO sobre completion   (completion-only loss)
-   LoRA r=8 α=16 sobre proyecciones
-   Quant 4-bit NF4 (bnb)
-   Truncación + padding 2048
-   Merge final para inferencia rápida
"""

import os, torch, transformers, datasets
from datasets import load_dataset
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, PeftModel

MODEL_ID       = "mistralai/Mistral-7B-Instruct-v0.2"
DATASET_PATH   = "./dataset/dataset.jsonl"
OUT_DIR        = "./checkpoints/mistral-cv-finetuned"
MERGED_DIR     = "./checkpoints/mistral-cv-merged"
SEQ_LEN        = 2048
BATCH          = 2
ACC_STEPS      = 4
EPOCHS         = 3
LR             = 2e-4

print(f"🔧 Versions → torch {torch.__version__} | transformers {transformers.__version__}")

# --- Config & tokenizer --------------------------------------------------------------------
cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
cfg.init_device = "cuda"
cfg.parallelization_style = "none"

tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tok.pad_token = tok.eos_token
tok.model_max_length = SEQ_LEN      # evita warnings y errores de overflow

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    config=cfg,
    quantization_config=bnb_cfg,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# --- Dataset -------------------------------------------------------------------------------
raw_ds = load_dataset("json", data_files=DATASET_PATH)["train"]
ds      = raw_ds.train_test_split(test_size=0.1, seed=42)

# --- Completion-only collator --------------------------------------------------------------
collator = DataCollatorForCompletionOnlyLM(
    tokenizer          = tok,
    response_template  = "completion",   # campo con la salida etiquetada
    instruction_template = "prompt",     # campo con el texto crudo
    mlm = False
)

# --- LoRA ----------------------------------------------------------------------------------
lora_cfg = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# --- Training args -------------------------------------------------------------------------
train_args = TrainingArguments(
    output_dir                 = OUT_DIR,
    per_device_train_batch_size= BATCH,
    gradient_accumulation_steps= ACC_STEPS,
    learning_rate              = LR,
    num_train_epochs           = EPOCHS,
    warmup_steps               = 20,
    fp16                       = True,
    logging_steps              = 25,
    save_strategy              = "epoch",
    evaluation_strategy        = "epoch",
    save_total_limit           = 2,
    remove_unused_columns      = False,
    report_to                  = "none"
)

# --- Trainer -------------------------------------------------------------------------------
trainer = SFTTrainer(
    model         = model,
    args          = train_args,
    train_dataset = ds["train"],
    eval_dataset  = ds["test"],
    peft_config   = lora_cfg,
    data_collator = collator,
    packing       = False           # mantiene un sample ↔ un ejemplo
)

# --- Train & merge -------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    trainer.train()
    trainer.save_model(OUT_DIR)

    # --- Merge LoRA ➜ full weights (para vLLM/TGI sin overhead) -------------
    print("➡️  Merging LoRA adapters con pesos base…")
    merged = PeftModel.from_pretrained(model, OUT_DIR).merge_and_unload()
    os.makedirs(MERGED_DIR, exist_ok=True)
    merged.save_pretrained(MERGED_DIR)
    tok.save_pretrained(MERGED_DIR)
    print(f"✅ Modelo mergeado listo en {MERGED_DIR}")
