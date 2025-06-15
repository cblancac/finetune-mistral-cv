#!/usr/bin/env python3
# training/finetune_mistral_cv.py
# Fine-tune estilo NER: loss sólo en completion

import os, torch, transformers, datasets
from datasets import load_dataset
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments
)
from peft import LoraConfig, PeftModel

print(f"🔧 torch {torch.__version__} | transformers {transformers.__version__}")

# ---------- hiperparámetros ----------
MODEL_ID   = "mistralai/Mistral-7B-Instruct-v0.2"
DATASET    = "./dataset/dataset.jsonl"
OUT_DIR    = "./checkpoints/mistral-cv-finetuned"
MERGED_DIR = "./checkpoints/mistral-cv-merged"
SEQ_LEN    = 2048
BATCH      = 2
ACC_STEPS  = 4
EPOCHS     = 3
LR         = 2e-4

# ---------- modelo 4-bit ----------
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
def norm(txt):                   # garantiza string plano
    return txt if isinstance(txt, str) else " ".join(map(str, txt))

raw = load_dataset("json", data_files=DATASET)["train"]
raw = raw.map(lambda r: {"prompt": norm(r["prompt"]),
                         "completion": norm(r["completion"])},
              num_proc=4)

data = raw.train_test_split(test_size=0.1, seed=42)

def tok_and_mask(r):
    p_ids = tok(r["prompt"], add_special_tokens=False,
                truncation=True, max_length=SEQ_LEN-1)["input_ids"]
    max_c = SEQ_LEN - len(p_ids) - 1
    c_ids = tok(r["completion"], add_special_tokens=False,
                truncation=True, max_length=max_c)["input_ids"]

    ids    = p_ids + c_ids + [tok.eos_token_id]
    labels = [-100]*len(p_ids) + c_ids + [tok.eos_token_id]
    return {"input_ids": ids, "labels": labels}

proc = data.map(tok_and_mask, remove_columns=data["train"].column_names,
                num_proc=4)

# ---------- collator que pad-ea input_ids y labels ----------
def collate(batch):
    ids   = [b["input_ids"] for b in batch]
    lbls  = [b["labels"]     for b in batch]
    enc   = tok.pad({"input_ids": ids}, return_tensors="pt",
                    padding=True)
    maxlen = enc["input_ids"].shape[1]
    # pad labels con -100
    padded_lbls = [torch.tensor(l + [-100]*(maxlen-len(l))) for l in lbls]
    enc["labels"] = torch.stack(padded_lbls)
    return enc

# ---------- training args ----------
args = TrainingArguments(
    output_dir                 = OUT_DIR,
    per_device_train_batch_size= BATCH,
    gradient_accumulation_steps= ACC_STEPS,
    learning_rate              = LR,
    num_train_epochs           = EPOCHS,
    warmup_steps               = 20,
    fp16                       = True,       # ya no usamos accel.fp16
    logging_steps              = 25,
    save_strategy              = "epoch",
    eval_strategy              = "epoch",
    save_total_limit           = 2,
    report_to                  = "none"
)

from transformers import Trainer
trainer = Trainer(
    model         = model,
    args          = args,
    train_dataset = proc["train"],
    eval_dataset  = proc["test"],
    data_collator = collate
)

# ---------- loop + merge ----------
if __name__ == "__main__":
    trainer.train()
    trainer.save_model(OUT_DIR)

    merged = PeftModel.from_pretrained(model, OUT_DIR).merge_and_unload()
    os.makedirs(MERGED_DIR, exist_ok=True)
    merged.save_pretrained(MERGED_DIR)
    tok.save_pretrained(MERGED_DIR)
    print(f"✅ Fine-tuning OK – modelo mergeado en {MERGED_DIR}")
