#!/usr/bin/env python3
# training/finetune_mistral_cv.py
# Fine-tuning estilo NER: sólo se optimiza la Completion.

import os, torch, transformers, datasets
from datasets import load_dataset
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, PeftModel

# ---- versiones -----------------------------------------------------------------
print(f"🔧 Versions → torch {torch.__version__} | transformers {transformers.__version__}")

# ---- hiperparámetros ------------------------------------------------------------
MODEL_ID   = "mistralai/Mistral-7B-Instruct-v0.2"
DATASET    = "./dataset/dataset.jsonl"
OUT_DIR    = "./checkpoints/mistral-cv-finetuned"
MERGED_DIR = "./checkpoints/mistral-cv-merged"
SEQ_LEN    = 2048
BATCH      = 2
ACC_STEPS  = 4
EPOCHS     = 3
LR         = 2e-4

# ---- modelo + tokenizador -------------------------------------------------------
cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
cfg.init_device = "cuda"
cfg.parallelization_style = "none"

tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tok.pad_token = tok.eos_token
tok.model_max_length = SEQ_LEN

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    config=cfg,
    device_map="auto",
    quantization_config=bnb_cfg,
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# ---- dataset --------------------------------------------------------------------
raw = load_dataset("json", data_files=DATASET)["train"]

def _ensure_string(rec):
    # Garantiza que prompt y completion son strings planos
    rec["prompt"]     = rec["prompt"] if isinstance(rec["prompt"], str) else " ".join(map(str, rec["prompt"]))
    rec["completion"] = rec["completion"] if isinstance(rec["completion"], str) else " ".join(map(str, rec["completion"]))
    return rec

clean = raw.map(_ensure_string, num_proc=4)
ds     = clean.train_test_split(test_size=0.1, seed=42)

# ---- collator (pérdida SOLO en completion) --------------------------------------
collator = DataCollatorForCompletionOnlyLM(
    tokenizer             = tok,
    response_template     = "completion",
    instruction_template  = "prompt",
    mlm=False
)

# ---- LoRA -----------------------------------------------------------------------
lora_cfg = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# ---- training args --------------------------------------------------------------
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
    eval_strategy              = "epoch",
    save_total_limit           = 2,
    remove_unused_columns      = False,
    report_to                  = "none"
)

# ---- trainer --------------------------------------------------------------------
trainer = SFTTrainer(
    model         = model,
    args          = train_args,
    train_dataset = ds["train"],
    eval_dataset  = ds["test"],
    data_collator = collator,
    peft_config   = lora_cfg
)

# ---- loop + merge ---------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    trainer.train()
    trainer.save_model(OUT_DIR)

    print("➡️  Merging LoRA → full checkpoints …")
    merged = PeftModel.from_pretrained(model, OUT_DIR).merge_and_unload()
    os.makedirs(MERGED_DIR, exist_ok=True)
    merged.save_pretrained(MERGED_DIR)
    tok.save_pretrained(MERGED_DIR)
    print(f"✅ Modelo mergeado guardado en {MERGED_DIR}")
