#!/usr/bin/env bash
# Fetch the GPT-2 teacher and all benchmark data from mirrors that do not
# require huggingface.co (which is blocked in some environments).
set -euo pipefail

mkdir -p models/gpt2 data/wikitext-2 data/benchmarks

# --- GPT-2 124M teacher (legacy Hugging Face S3 bucket, HF format) ---
base="https://s3.amazonaws.com/models.huggingface.co/bert"
[ -f models/gpt2/pytorch_model.bin ] || curl -SL "$base/gpt2-pytorch_model.bin" -o models/gpt2/pytorch_model.bin
[ -f models/gpt2/config.json ]       || curl -SL "$base/gpt2-config.json"       -o models/gpt2/config.json
[ -f models/gpt2/vocab.json ]        || curl -SL "$base/gpt2-vocab.json"        -o models/gpt2/vocab.json
[ -f models/gpt2/merges.txt ]        || curl -SL "$base/gpt2-merges.txt"        -o models/gpt2/merges.txt

# --- WikiText-2 (word-level version vendored in pytorch/examples) ---
wt="https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2"
for split in train valid test; do
  [ -f data/wikitext-2/$split.txt ] || curl -SL "$wt/$split.txt" -o data/wikitext-2/$split.txt
done

# --- LAMBADA (OpenAI test split mirror) ---
[ -f data/benchmarks/lambada_test.jsonl ] || \
  curl -SL "https://raw.githubusercontent.com/cybertronai/bflm/master/lambada_test.jsonl" \
    -o data/benchmarks/lambada_test.jsonl

# --- HellaSwag validation ---
[ -f data/benchmarks/hellaswag_val.jsonl ] || \
  curl -SL "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl" \
    -o data/benchmarks/hellaswag_val.jsonl

# --- ARC-Easy (AI2 public S3) ---
if [ ! -f data/benchmarks/ARC-Easy-Dev.jsonl ]; then
  curl -SL "https://ai2-public-datasets.s3.amazonaws.com/arc/ARC-V1-Feb2018.zip" -o /tmp/arc.zip
  python3 - <<'EOF'
import zipfile
with zipfile.ZipFile("/tmp/arc.zip") as z:
    for name in z.namelist():
        if name.endswith("ARC-Easy-Dev.jsonl") and "__MACOSX" not in name:
            with z.open(name) as f, open("data/benchmarks/ARC-Easy-Dev.jsonl", "wb") as out:
                out.write(f.read())
EOF
  rm -f /tmp/arc.zip
fi

echo "All data fetched."
