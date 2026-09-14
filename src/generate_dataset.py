import json
import re
from collections import Counter
from pathlib import Path

import torch


# ============================================================
# TinyLuauGPT v5 Dataset Preparation
# ============================================================

DATASET_PATH = Path("dataset/luau_v5.txt")
TOKENS_PATH = Path("dataset/tokens_v5.pt")
TOKENIZER_PATH = Path("model/tokenizer_v5.json")


# ============================================================
# Special Tokens
# ============================================================

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


# ============================================================
# Tokenizer Pattern
# ============================================================

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


# ============================================================
# Dataset 확인
# ============================================================

if not DATASET_PATH.exists():
    print("=" * 60)
    print("ERROR")
    print("=" * 60)
    print(f"Dataset not found:")
    print(f"  {DATASET_PATH}")
    print()
    print("먼저 다음 명령을 실행하세요:")
    print()
    print("python src\\generate_dataset.py")
    print()
    raise SystemExit(1)


# ============================================================
# Dataset 읽기
# ============================================================

text = DATASET_PATH.read_text(encoding="utf-8")

print("=" * 60)
print("TinyLuauGPT v5 Dataset Preparation")
print("=" * 60)

print()
print(f"Dataset:     {DATASET_PATH}")
print(f"Characters:  {len(text):,}")


# ============================================================
# Tokenization
# ============================================================

print()
print("Tokenizing dataset...")

tokens = tokenize(text)

print(f"Raw tokens:  {len(tokens):,}")


# ============================================================
# Vocabulary
# ============================================================

print()
print("Building vocabulary...")

counter = Counter(tokens)

vocab = {}

# Special tokens를 가장 먼저 등록
for token in SPECIAL_TOKENS:
    vocab[token] = len(vocab)


# 일반 토큰 등록
for token, count in counter.most_common():
    if token not in vocab:
        vocab[token] = len(vocab)


print(f"Vocabulary size: {len(vocab):,}")


# ============================================================
# Encode
# ============================================================

print()
print("Encoding dataset...")

unk_id = vocab["<UNK>"]

ids = []

for token in tokens:
    token_id = vocab.get(token, unk_id)
    ids.append(token_id)


tensor = torch.tensor(ids, dtype=torch.long)

print(f"Total tokens: {len(ids):,}")


# ============================================================
# Directory 생성
# ============================================================

TOKENS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

TOKENIZER_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Tokens 저장
# ============================================================

torch.save(
    tensor,
    TOKENS_PATH
)


# ============================================================
# Tokenizer 저장
# ============================================================

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


# ============================================================
# 결과 출력
# ============================================================

print()
print("=" * 60)
print("Saved")
print("=" * 60)

print(f"Tokens:")
print(f"  {TOKENS_PATH.resolve()}")

print()

print(f"Tokenizer:")
print(f"  {TOKENIZER_PATH.resolve()}")


# ============================================================
# Special Token 출력
# ============================================================

print()
print("Special tokens:")

for token in SPECIAL_TOKENS:
    print(f"  {token:12s} -> {vocab[token]}")


# ============================================================
# Token 미리보기
# ============================================================

print()
print("=" * 60)
print("Token Preview")
print("=" * 60)

preview_count = min(30, len(tokens))

for i in range(preview_count):
    token = tokens[i]
    token_id = ids[i]

    print(
        f"{i:4d}: "
        f"{token!r:25s} "
        f"-> {token_id}"
    )


# ============================================================
# Dataset Mode 통계
# ============================================================

print()
print("=" * 60)
print("Dataset Mode Statistics")
print("=" * 60)

for mode in [
    "CHAT",
    "CODE",
    "EXPLAIN",
    "FIX",
]:
    count = text.count(f"<{mode}>")
    print(f"{mode:10s}: {count:,}")


# ============================================================
# 완료
# ============================================================

print()
print("=" * 60)
print("Dataset preparation complete!")
print("=" * 60)

print()
print("v4 files were NOT overwritten.")
print()
print("v5 files:")
print(f"  {TOKENS_PATH}")
print(f"  {TOKENIZER_PATH}")
print()