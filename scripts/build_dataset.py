#!/usr/bin/env python3
import json
import boto3
from botocore.exceptions import ClientError

# 1) CONFIGURA TU S3
BUCKET_NAME = "test-s3-putobject-presigned"
PREFIX_RAW  = "raw_text_without_annotations/"
PREFIX_LBL  = "labled/"
OUTPUT_FILE = "dataset.jsonl"

s3 = boto3.client("s3")

def list_keys(prefix):
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys

def get_s3_text(key):
    try:
        resp = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        return resp["Body"].read().decode("utf-8")
    except ClientError as e:
        print(f"⚠️ Error al leer {key}: {e}")
        return None

def main():
    # 2) Lista archivos
    raw_keys   = [k for k in list_keys(PREFIX_RAW) if k.endswith(".txt")]
    label_keys = [k for k in list_keys(PREFIX_LBL) if k.endswith("_labeled.json")]

    # 3) Construye un diccionario: base_name -> label_key
    label_map = {}
    for key in label_keys:
        # Extrae "filename" de "labeled/filename_labeled.json"
        basename = key.split("/")[-1].replace("_labeled.json", "")
        label_map[basename] = key
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for raw_key in raw_keys:
            # Extrae "filename" de "raw_text_without_annotations/filename.txt"
            basename = raw_key.split("/")[-1].replace(".txt", "")
            lbl_key = label_map.get(basename)
            if not lbl_key:
                print(f"✏️  No hay etiqueta para {basename}, lo salto.")
                continue
    
            # 4) Descarga contenido
            text = get_s3_text(raw_key)
            ann_json = get_s3_text(lbl_key)
            if text is None or ann_json is None:
                continue
    
            # 5) Parseo y serialización
            try:
                ann = json.loads(ann_json)
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON inválido en {lbl_key}: {e}")
                continue
    
            # ---------- NEW: serializa el dict completo como una única cadena JSON
            completion = json.dumps(          # ✨ reemplaza el join(…)
                ann,
                ensure_ascii=False,           # conserva acentos
                separators=(",", ":")         # minificado; quita espacios extra
                #  ↳ usa `indent=2` si prefieres pretty-print, pero sé consistente
            )
    
            # 6) Escribe una línea JSONL
            record = {
                "prompt": text.strip(),       # ✨ elimina \n inicial/final
                "completion": completion
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ dataset.jsonl generado: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
