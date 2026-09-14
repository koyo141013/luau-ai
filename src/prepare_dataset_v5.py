import json
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


def tokenize(text):
    tokens = []
    i = 0

    special_tokens = sorted(
        SPECIAL_TOKENS,
        key=len,
        reverse=True
    )

    operators = [
        "==",
        "~=",
        "<=",
        ">=",
        "+=",
        "-=",
        "*=",
        "/=",
        "::",
        "->",
        "..",
    ]

    while i < len(text):

        # 특수 토큰
        found_special = False

        for special in special_tokens:
            if text.startswith(special, i):
                tokens.append(special)
                i += len(special)
                found_special = True
                break

        if found_special:
            continue

        # 공백
        if text[i].isspace():
            if text[i] == "\n":
                tokens.append("\n")
            i += 1
            continue

        # 문자열
        if text[i] == '"' or text[i] == "'":
            quote = text[i]
            start = i
            i += 1

            while i < len(text):
                if text[i] == "\\":
                    i += 2
                    continue

                if text[i] == quote:
                    i += 1
                    break

                i += 1

            tokens.append(text[start:i])
            continue

        # 숫자
        if text[i].isdigit():
            start = i

            while i < len(text):
                if text[i].isdigit() or text[i] == ".":
                    i += 1
                else:
                    break

            tokens.append(text[start:i])
            continue

        # 영어 identifier
        if text[i].isalpha() or text[i] == "_":
            start = i

            while i < len(text):
                ch = text[i]

                if ch.isalnum() or ch == "_":
                    i += 1
                else:
                    break

            tokens.append(text[start:i])
            continue

        # 여러 글자 연산자
        found_operator = False

        for op in operators:
            if text.startswith(op, i):
                tokens.append(op)
                i += len(op)
                found_operator = True
                break

        if found_operator:
            continue

        # 한 글자
        tokens.append(text[i])
        i += 1

    return tokens


if not DATASET_PATH.exists():
    print("=" * 60)
    print("ERROR")
    print("=" * 60)
    print(f"Dataset not found:")
    print(DATASET_PATH)
    print()
    print("dataset/luau_v5.txt 파일이 있는지 확인하세요.")
    raise SystemExit(1)


print("=" * 60)
print("TinyLuauGPT v5 Dataset Preparation")
print("=" * 60)

text = DATASET_PATH.read_text(
    encoding="utf-8"
)

print()
print(f"Dataset:    {DATASET_PATH}")
print(f"Characters: {len(text):,}")

print()
print("Tokenizing dataset...")

tokens = tokenize(text)

print(f"Total tokens: {len(tokens):,}")

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

tensor = torch.tensor(
    ids,
    dtype=torch.long
)

TOKENS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

TOKENIZER_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

torch.save(
    tensor,
    TOKENS_PATH
)

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

print()
print("Tokens:")
print(TOKENS_PATH.resolve())

print()
print("Tokenizer:")
print(TOKENIZER_PATH.resolve())

print()
print("Special tokens:")

for token in SPECIAL_TOKENS:
    print(
        f"  {token:12s} -> {vocab[token]}"
    )

print()
print("=" * 60)
print("Dataset Mode Statistics")
print("=" * 60)

for mode in ["CHAT", "CODE", "EXPLAIN", "FIX"]:
    count = text.count(f"<{mode}>")
    print(
        f"{mode:10s}: {count:,}"
    )

print()
print("=" * 60)
print("TinyLuauGPT v5 preparation complete!")
print("=" * 60)
