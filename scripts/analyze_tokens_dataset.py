#!/usr/bin/env python3
# inspect_tokens.py
#
#   pip install transformers datasets numpy tqdm
#   python inspect_tokens.py ./dataset/dataset.jsonl --seq-len 2048
#
# Mide cuántos tokens ocupa cada ejemplo con el tokenizer de Mistral-7B-Instruct
# usando la misma plantilla de chat y el mismo SCHEMA que en el fine-tune.

import argparse, json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

assistant_lengths = []  # lista para longitudes assistant
assistant_indices = []  # índices para los top 5

# ---------------------------------------------------------------------------
# 1️⃣  Schema EXACTO utilizado en el fine-tune
# ---------------------------------------------------------------------------
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
}, separators=(",", ":"))                   # minificado, sin espacios

SYSTEM_PROMPT = (
    "You are an API that extracts structured JSON from resumes.\n"
    "Return *only* valid JSON matching exactly this schema:\n"
    f"{SCHEMA}"
)

# ---------------------------------------------------------------------------
# 2️⃣  CLI
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="Ruta a dataset.jsonl")
    ap.add_argument("--model",
                    default="mistralai/Mistral-7B-Instruct-v0.2",
                    help="ID o ruta del modelo/tokenizer base")
    ap.add_argument("--seq-len", type=int, default=2048,
                    help="Longitud objetivo (para ver cuántos se truncan)")
    ap.add_argument("--sample", type=int,
                    help="Analiza solo los primeros N ejemplos (debug)")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# 3️⃣  Medición
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    seq_len = args.seq_len

    ds = load_dataset("json", data_files=args.jsonl)["train"]
    if args.sample:
        ds = ds.select(range(args.sample))

    lengths = []
    truncated = 0

    for idx, rec in enumerate(tqdm(ds, desc="Tokenizando")):
        # Construimos exactamente la misma conversación que en el entrenamiento
        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": rec["prompt"]},
            {"role": "assistant", "content": rec["completion"]},
        ]

        ids = tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        length = len(ids) + 1          # +1 por el </s> final que añade el trainer
        if length > seq_len:
            truncated += 1
        lengths.append(length)

        # === Añadido: solo tokens assistant ===
        assistant_ids = tok(
            rec["completion"],
            add_special_tokens=False,
        )["input_ids"]
        assistant_lengths.append(len(assistant_ids))
        assistant_indices.append(idx)

    # -----------------------------------------------------------------------
    # 4️⃣  Estadísticos
    # -----------------------------------------------------------------------
    arr = np.array(lengths)
    total = len(arr)
    print("\n📊  Estadísticas de longitud en tokens")
    print(f"   Ejemplos analizados   : {total}")
    print(f"   Media / mediana       : {arr.mean():.0f} / {np.median(arr):.0f}")
    print(f"   p90 / p95 / p99       : "
          f"{np.percentile(arr, 90):.0f} / "
          f"{np.percentile(arr, 95):.0f} / "
          f"{np.percentile(arr, 99):.0f}")
    print(f"   Máximo                : {arr.max():.0f}")
    print(f"   > {seq_len} (truncados): "
          f"{truncated}  ({truncated/total*100:.2f} %)")

    # Sugerencia rápida
    if truncated == 0 or np.percentile(arr, 99) <= seq_len:
        print(f"\n✅  Con {seq_len} tokens prácticamente ningún ejemplo "
              "se trunca; puedes dejar SEQ_LEN como está.")
    else:
        # propone un nuevo tamaño = p99.5 redondeado + margen
        new_len = int(min(8192, np.percentile(arr, 99.5) + 32))
        print(f"\n⚠️  {truncated} ejemplos se truncarían a {seq_len} tokens. "
              f"Considera subir SEQ_LEN a ≈ {new_len} "
              "(o trocear CVs largos en inference).")
        
    # --- Nuevo análisis: top 5 tokens en completions ---
    assistant_lengths = np.array(assistant_lengths)
    top5_indices = assistant_lengths.argsort()[-5:][::-1]  # índices top 5 (mayor a menor)
    top5_values = assistant_lengths[top5_indices]

    print("\n🧮  Top 5 valores de tokens assistant (completion):")
    for rank, (idx, val) in enumerate(zip(top5_indices, top5_values), 1):
        print(f"{rank}. Ejemplo #{idx} -> {val} tokens")


if __name__ == "__main__":
    main()
