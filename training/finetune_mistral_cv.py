#!/usr/bin/env python3
# training/finetune_mistral_cv.py
# Fine-tune Mistral-7B-Instruct-v0.2 in 4-bit + LoRA
# Loss is applied only on the assistant (structured-JSON) turn.

import os, torch, transformers, datasets
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)

print(f"🔧 torch {torch.__version__} | transformers {transformers.__version__}")

# ---------------------------------------------------------------------------
# HYPER-PARAMETERS
# ---------------------------------------------------------------------------
MODEL_ID   = "mistralai/Mistral-7B-Instruct-v0.2"
DATA_FILE  = "./dataset/dataset.jsonl"           # one example per line
OUT_DIR    = "./checkpoints/mistral-cv-finetuned"
MERGE_DIR  = "./checkpoints/mistral-cv-merged"
SEQ_LEN    = 2048
BATCH      = 2
ACC_STEPS  = 4
EPOCHS     = 3
LR         = 2e-4       # try 1e-4 if you notice over-fitting

# ---------------------------------------------------------------------------
# BASE MODEL (4-bit) + LoRA
# ---------------------------------------------------------------------------
cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tok.pad_token = tok.eos_token
tok.model_max_length = SEQ_LEN

base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    config=cfg,
    quantization_config=bnb,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

# prepare_model_for_kbit_training unlocks gradients & casts norms to fp32
base = prepare_model_for_kbit_training(base)

lora_cfg = LoraConfig(
    r                = 8,
    lora_alpha       = 16,
    lora_dropout     = 0.05,
    target_modules   = ["q_proj", "k_proj", "v_proj", "o_proj"],
    bias             = "none",
    task_type        = "CAUSAL_LM",
)

model = get_peft_model(base, lora_cfg)
model.print_trainable_parameters()        # sanity-check trainable %


# ---------------------------------------------------------------------------
# DATASET  →  chat-formatted messages
# ---------------------------------------------------------------------------
def norm(x):            # input may be list or str
    return x if isinstance(x, str) else " ".join(map(str, x))

raw = load_dataset("json", data_files=DATA_FILE)["train"]

SCHEMA = (
    '{'
      '"certifications":"",'
      '"contact_detail":{'
        '"age":"",'
        '"email":"",'
        '"home_city":"",'
        '"mobile":"",'
        '"name":""'
      '},'
      '"education":[],'
      '"gender":"",'
      '"industry":"",'
      '"skills":[],'
      '"software_tools":[],'
      '"work_abroad":"",'
      '"work_experience":[]'
    '}'
)

raw = raw.map(  # convert each row into a mini-dialogue
    lambda r: {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an API that extracts structured JSON from resumes.\n"
                    "Return *only* valid JSON matching exactly this schema:\n"
                    f"{SCHEMA}"
                ),
            },
            {"role": "user",      "content": norm(r["prompt"])},
            {"role": "assistant", "content": norm(r["completion"])},
        ]
    },
    num_proc=4,
)

splits = raw.train_test_split(test_size=0.1, seed=42)

# -------------------------- tokenisation & masking -------------------------
def tok_mask(example):
    """
    Build tokens:
      * System + user  -> tokens (we keep them for context, but ignore loss)
      * Assistant      -> tokens that will receive the loss
    We rely on the model's chat template to insert special tokens.
    """
    # 1️⃣  tokens for system+user with the assistant "cue" but *no* answer
    sys_user_ids = tok.apply_chat_template(
        example["messages"][:-1],            # drop assistant message
        tokenize=True,
        add_generation_prompt=True,          # leaves <assistant> tag & BOS
        truncation=True,
        max_length=SEQ_LEN - 1,              # reserve 1 for final </s>
    )

    # 2️⃣  tokens for the assistant answer (our ground-truth JSON frame)
    assistant_ids = tok(
        example["messages"][-1]["content"],
        add_special_tokens=False,
    )["input_ids"] + [tok.eos_token_id]

    # 3️⃣  fuse & build labels
    input_ids = sys_user_ids + assistant_ids
    labels    = [-100] * len(sys_user_ids) + assistant_ids

    # truncate if the combined length still exceeds SEQ_LEN
    input_ids = input_ids[:SEQ_LEN]
    labels    = labels[:SEQ_LEN]

    return {"input_ids": input_ids, "labels": labels}

proc = splits.map(
    tok_mask,
    remove_columns=splits["train"].column_names,
    num_proc=4,
)

# -------------------------- collator ---------------------------------------
def collate(batch):
    ids  = [b["input_ids"] for b in batch]
    lbls = [b["labels"]     for b in batch]

    enc = tok.pad({"input_ids": ids}, return_tensors="pt", padding=True)
    maxlen = enc["input_ids"].shape[1]

    lbl_pad = [
        torch.tensor(l + [-100] * (maxlen - len(l))) for l in lbls
    ]
    enc["labels"] = torch.stack(lbl_pad)
    return enc

# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------
args = TrainingArguments(
    output_dir                   = OUT_DIR,
    per_device_train_batch_size  = BATCH,
    gradient_accumulation_steps  = ACC_STEPS,
    learning_rate                = LR,
    num_train_epochs             = EPOCHS,
    warmup_steps                 = 20,
    fp16                         = True,
    logging_steps                = 25,
    save_strategy                = "epoch",
    evaluation_strategy          = "epoch",
    save_total_limit             = 2,
    report_to                    = "none",
)

trainer = Trainer(
    model           = model,
    args            = args,
    train_dataset   = proc["train"],
    eval_dataset    = proc["test"],
    data_collator   = collate,
)

# ---------------------------------------------------------------------------
# FIT  +  MERGE LoRA  →  standalone fp16 model
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    trainer.train()
    trainer.save_model(OUT_DIR)

    merged = PeftModel.from_pretrained(model, OUT_DIR).merge_and_unload()
    merged.save_pretrained(MERGE_DIR)
    tok.save_pretrained(MERGE_DIR)

    print(f"✅ Fine-tune complete — merged model saved to {MERGE_DIR}")
