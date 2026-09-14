import json
import re
from collections import Counter
from pathlib import Path

import torch


DATASET_PATH = Path("dataset/luau_v5.txt")
TOKENS_PATH = Path("dataset/tokens_v5.pt")
TOKENIZER_PATH = Path("model/tokenizer_v5.json")

SPECIAL_TOKENS = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>",
    "<CHAT>",
    "<CODE>",
    "<EXPLAIN>",
    "<FIX>",
    "<RESPONSE>",
    "<END>",
]

TOKEN_PATTERN = re.compile(
    r"""
    <PAD>|<UNK>|<BOS>|<EOS>|<CHAT>|<CODE>|<EXPLAIN>|<FIX>|<RESPONSE>|<END>
    |\r\n|\n|\r
    |\s+
    |[A-Za-z_][A-Za-z0-9_]*
    |[0-9]+(?:\.[0-9]+)?
    |"(?:\\.|[^"\\])*"
    |'(?:\\.|[^'\\])*'
    |==|~=|<=|>=|+=|-=|\*=|/=|::|->|\.\.
    |[^\s]
    """,
    re.VERBOSE,
)


def tokenize(text):
    return TOKEN_PATTERN.findall(text)


if not DATASET_PATH.exists():
    print("=" * 60)
    print("ERROR")
    print("=" * 60)
    print(f"Dataset not found: {DATASET_PATH}")
    print()
    print("먼저 실행하세요:")
    print("python src\\generate_dataset.py")
    raise SystemExit(1)


text = DATASET_PATH.read_text(encoding="utf-8")

print("=" * 60)
print("TinyLuauGPT v5 Dataset Preparation")
print("=" * 60)

print()
print(f"Dataset:    {DATASET_PATH}")
print(f"Characters: {len(text):,}")

print()
print("Tokenizing dataset...")

tokens = tokenize(text)

print(f"Raw tokens: {len(tokens):,}")

print()
print("Building vocabulary...")

counter = Counter(tokens)

vocab = {}

for token in SPECIAL_TOKENS:
    vocab[token] = len(vocab)

for token, count in counter.most_common():
    if token not in vocab:
        vocab[token] = len(vocab)

print(f"Vocabulary size: {len(vocab):,}")

print()
print("Encoding dataset...")

unk_id = vocab["<UNK>"]

ids = [
    vocab.get(token, unk_id)
    for token in tokens
]

tensor = torch.tensor(ids, dtype=torch.long)

print(f"Total tokens: {len(ids):,}")

TOKENS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

TOKENIZER_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

torch.save(tensor, TOKENS_PATH)

tokenizer_data = {
    "vocab": vocab,
    "special_tokens": SPECIAL_TOKENS,
}

TOKENIZER_PATH.write_text(
    json.dumps(
        tokenizer_data,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)

print()
print("=" * 60)
print("Saved")
print("=" * 60)

print(f"Tokens:")
print(f"  {TOKENS_PATH.resolve()}")

print()
print(f"Tokenizer:")
print(f"  {TOKENIZER_PATH.resolve()}")

print()
print("Special tokens:")

for token in SPECIAL_TOKENS:
    print(f"  {token:12s} -> {vocab[token]}")

print()
print("=" * 60)
print("Dataset Mode Statistics")
print("=" * 60)

for mode in ["CHAT", "CODE", "EXPLAIN", "FIX"]:
    count = text.count(f"<{mode}>")
    print(f"{mode:10s}: {count:,}")

print()
print("=" * 60)
print("TinyLuauGPT v5 Dataset preparation complete!")
print("=" * 60)
