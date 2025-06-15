#!/usr/bin/env python3
# training/finetune_mistral_cv.py
# -------------------------------------------------------------------------
# Fine-tune de Mistral-7B-Instruct (int4 + LoRA) para extracción de CVs
# * Pérdida solo en completion
# * Dataset jsonl con campos: {"prompt": "...", "completion": "..."}
# -------------------------------------------------------------------------

import os, torch, transformers, torchvision, torchaudio
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    AutoConfig,
    BitsAndBytesConfig,
)
from trl import SFTTrainer                       # v0.18.x
from peft import LoraConfig, PeftModel

# --------------------------- 0 . Versión de libs --------------------------
print(
    f"🛠️  Versions → transformers {transformers.__version__} | "
    f"torch {torch.__version__} | torchvision {torchvision.__version__} | "
    f"torchaudio {torchaudio.__version__}"
)

# --------------------------- 1 . Parámetros -------------------------------
MODEL_ID       = "mistralai/Mistral-7B-Instruct-v0.2"
DATASET_PATH   = "./dataset/dataset.jsonl"
OUTPUT_DIR     = "./checkpoints/mistral-cv-finetuned"
MERGED_DIR     = "./checkpoints/mistral-cv-merged"
MAX_SEQ_LEN    = 2048
BATCH_SIZE     = 2
GRAD_ACCUM     = 4
EPOCHS         = 3
LR             = 2e-4

# --------------------------- 2 . Tokenizer & modelo -----------------------
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tok.pad_token = tok.eos_token               # asegura pad_token
tok.model_max_length = MAX_SEQ_LEN

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
cfg.init_device = "cuda"
cfg.parallelization_style = "none"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    config=cfg,
    device_map="auto",
    quantization_config=bnb_cfg,
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

# --------------------------- 3 . Dataset ----------------------------------
ds = load_dataset("json", data_files=DATASET_PATH)["train"]
ds = ds.train_test_split(test_size=0.1, seed=42)   # 90 / 10

# --------------------------- 4 . Collator personalizado -------------------
def build_sample(example):
    """Convierte dict→dict con ids + labels."""
    # IDs del prompt + <eos>
    prompt_ids = tok(
        example["prompt"],
        add_special_tokens=False,
    ).input_ids + [tok.eos_token_id]

    # IDs del completion + <eos>
    comp_ids = tok(
        example["completion"],
        add_special_tokens=False,
    ).input_ids + [tok.eos_token_id]

    # Corte si pasa el límite (recortamos del prompt)
    total_ids = prompt_ids + comp_ids
    if len(total_ids) > MAX_SEQ_LEN:
        overflow = len(total_ids) - MAX_SEQ_LEN
        prompt_ids = prompt_ids[overflow:]
        total_ids = prompt_ids + comp_ids

    labels = [-100] * len(prompt_ids) + comp_ids       # máscara en prompt
    example["input_ids"] = total_ids
    example["labels"]    = labels
    return example

ds_proc = ds.map(build_sample, remove_columns=ds["train"].column_names)

def collate(batch):
    """Pad a bloque máximo."""
    input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
    labels    = [torch.tensor(x["labels"],    dtype=torch.long) for x in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tok.pad_token_id,
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels,
        batch_first=True,
        padding_value=-100,
    )
    attention_mask = (input_ids != tok.pad_token_id).long()
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }

# --------------------------- 5 . LoRA -------------------------------------
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# --------------------------- 6 . TrainingArguments ------------------------
train_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    warmup_steps=20,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",          # “eval_strategy” es válido en 4.51.x
    save_total_limit=2,
    report_to="none",
    remove_unused_columns=False,
)

# --------------------------- 7 . SFTTrainer -------------------------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tok,
    train_dataset=ds_proc["train"],
    eval_dataset=ds_proc["test"],
    args=train_args,
    peft_config=lora_cfg,
    data_collator=collate,
    max_seq_length=MAX_SEQ_LEN,
    packing=False,
)

# --------------------------- 8 . Entrenamiento + merge --------------------
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.train()
    print(f"✅ Fine-tune completado. LoRA guardada en: {OUTPUT_DIR}")

    merged = PeftModel.from_pretrained(model, OUTPUT_DIR).merge_and_unload()
    os.makedirs(MERGED_DIR, exist_ok=True)
    merged.save_pretrained(MERGED_DIR)
    tok.save_pretrained(MERGED_DIR)
    print(f"✅ Modelo mergeado listo en: {MERGED_DIR}")
