#!/usr/bin/env python3
# training/finetune_mistral_cv.py
# Fine-tune estilo NER (loss sólo en completion) – Mistral-7B int4 + LoRA

import os, torch, transformers, datasets
from datasets import load_dataset
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments, DataCollatorWithPadding
)
from peft import LoraConfig, PeftModel
from accelerate import Accelerator

# ---------- versiones ----------
print(f"🔧 torch {torch.__version__} | transformers {transformers.__version__}")

# ---------- hiperparámetros ----
MODEL_ID   = "mistralai/Mistral-7B-Instruct-v0.2"
DATASET    = "./dataset/dataset.jsonl"
OUT_DIR    = "./checkpoints/mistral-cv-finetuned"
MERGED_DIR = "./checkpoints/mistral-cv-merged"
SEQ_LEN    = 2048
BATCH      = 2
ACC_STEPS  = 4
EPOCHS     = 3
LR         = 2e-4

accel = Accelerator()   # para reproducir -fp16, DDP, etc.

# ---------- modelo 4-bit + tokenizador ----------
cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
bnb = BitsAndBytesConfig(load_in_4bit=True,
                         bnb_4bit_compute_dtype=torch.float16,
                         bnb_4bit_use_double_quant=True,
                         bnb_4bit_quant_type="nf4")

tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tok.pad_token = tok.eos_token
tok.model_max_length = SEQ_LEN

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    config=cfg,
    quantization_config=bnb,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

# ---------- LoRA ----------
lora_cfg = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# ---------- dataset ----------
raw = load_dataset("json", data_files=DATASET)["train"]

def _to_str(x):                 # asegura texto plano
    if not isinstance(x, str):
        x = " ".join(map(str, x))  # lista -> string
    return x.strip()

def preprocess(rec):
    rec["prompt"]     = _to_str(rec["prompt"])
    rec["completion"] = _to_str(rec["completion"])
    return rec

clean = raw.map(preprocess, num_proc=4)
split = clean.train_test_split(test_size=0.1, seed=42)

def tokenize(example):
    # --- encode prompt (context) ---
    p_ids = tok(
        example["prompt"],
        add_special_tokens=False,
        truncation=True,
        max_length=SEQ_LEN - 1     # deja sitio para al menos 1 token del completion + eos
    )["input_ids"]

    # --- encode completion ---
    max_c_len = SEQ_LEN - len(p_ids) - 1
    c_ids = tok(
        example["completion"],
        add_special_tokens=False,
        truncation=True,
        max_length=max_c_len
    )["input_ids"]

    input_ids = p_ids + c_ids + [tok.eos_token_id]
    labels    = [-100]*len(p_ids) + c_ids + [tok.eos_token_id]   # máscara prompt

    return {"input_ids": input_ids, "labels": labels}

tokenized = split.map(tokenize, remove_columns=split["train"].column_names, num_proc=4)

collator = DataCollatorWithPadding(tok, pad_to_multiple_of=8, return_tensors="pt")

# ---------- training args ----------
args = TrainingArguments(
    output_dir                 = OUT_DIR,
    per_device_train_batch_size= BATCH,
    gradient_accumulation_steps= ACC_STEPS,
    learning_rate              = LR,
    num_train_epochs           = EPOCHS,
    warmup_steps               = 20,
    fp16                       = accel.fp16,       # se adapta a accelerate
    logging_steps              = 25,
    save_strategy              = "epoch",
    eval_strategy              = "epoch",
    save_total_limit           = 2,
    report_to                  = "none"
)

# ---------- trainer ----------
from transformers import Trainer
trainer = Trainer(
    model           = model,
    args            = args,
    train_dataset   = tokenized["train"],
    eval_dataset    = tokenized["test"],
    data_collator   = collator
)

# ---------- LOOP + merge ----------
if __name__ == "__main__":
    accel.print("🚀 Starting fine-tuning …")
    trainer.train()
    trainer.save_model(OUT_DIR)

    accel.print("🔀 Merging LoRA into base weights …")
    merged = PeftModel.from_pretrained(model, OUT_DIR).merge_and_unload()
    os.makedirs(MERGED_DIR, exist_ok=True)
    merged.save_pretrained(MERGED_DIR)
    tok.save_pretrained(MERGED_DIR)
    accel.print(f"✅ Listo: {MERGED_DIR}")
