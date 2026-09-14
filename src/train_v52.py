import math
import time
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from model_v52 import TinyLuauGPTv52


# ============================================================
# Luau AI v5.2
# User-facing version: Beta 0.5
# ============================================================

VERSION = "5.2-beta.0.5"

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "dataset" / "tokens_v52.pt"
TOKENIZER_PATH = ROOT / "model" / "tokenizer_v52.json"

BEST_PATH = ROOT / "model" / "tiny_luau_gpt_v52_best.pt"
FINAL_PATH = ROOT / "model" / "tiny_luau_gpt_v52.pt"


# -----------------------------
# Training settings
# -----------------------------

STEPS = 5000
BATCH_SIZE = 16
BLOCK_SIZE = 256

LR_MAX = 3e-4
LR_MIN = 3e-5

WEIGHT_DECAY = 0.1

EVAL_INTERVAL = 100
EVAL_BATCHES = 20

TRAIN_RATIO = 0.90

torch.manual_seed(42)


# -----------------------------
# Device
# -----------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 64)
print("Luau AI v5.2 Training")
print("User-facing version: Beta 0.5")
print("=" * 64)

print(f"Device: {device}")

if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")


# -----------------------------
# Load dataset
# -----------------------------

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )

tokens = torch.load(DATASET_PATH, map_location="cpu")

if tokens.dtype != torch.long:
    tokens = tokens.long()

total_tokens = len(tokens)

train_size = int(total_tokens * TRAIN_RATIO)

train_tokens = tokens[:train_size]
val_tokens = tokens[train_size:]

print()
print(f"Total tokens:      {total_tokens:,}")
print(f"Training tokens:   {len(train_tokens):,}")
print(f"Validation tokens: {len(val_tokens):,}")


# -----------------------------
# Load tokenizer
# -----------------------------

if not TOKENIZER_PATH.exists():
    raise FileNotFoundError(
        f"Tokenizer not found:\n{TOKENIZER_PATH}"
    )

with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
    tokenizer = json.load(f)

vocab = tokenizer["vocab"]
vocab_size = len(vocab)

print(f"Vocabulary size:   {vocab_size}")


# -----------------------------
# Model
# -----------------------------

model = TinyLuauGPTv52(
    vocab_size=vocab_size,
    block_size=BLOCK_SIZE,
    n_embd=320,
    n_head=8,
    n_layer=8,
    dropout=0.1,
).to(device)

print(f"Parameters:        {model.num_parameters():,}")


# -----------------------------
# Optimizer
# -----------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR_MAX,
    betas=(0.9, 0.95),
    weight_decay=WEIGHT_DECAY,
)


# -----------------------------
# Batch function
# -----------------------------

def get_batch(data):
    max_start = len(data) - BLOCK_SIZE - 1

    ix = torch.randint(
        0,
        max_start,
        (BATCH_SIZE,),
    )

    x = torch.stack([
        data[i:i + BLOCK_SIZE]
        for i in ix
    ])

    y = torch.stack([
        data[i + 1:i + BLOCK_SIZE + 1]
        for i in ix
    ])

    return x.to(device), y.to(device)


# -----------------------------
# Evaluation
# -----------------------------

@torch.no_grad()
def estimate_loss():

    model.eval()

    results = {}

    for name, data in [
        ("train", train_tokens),
        ("val", val_tokens),
    ]:

        losses = []

        for _ in range(EVAL_BATCHES):

            x, y = get_batch(data)

            logits, loss = model(x, y)

            losses.append(loss.item())

        results[name] = sum(losses) / len(losses)

    model.train()

    return results


# -----------------------------
# Learning rate schedule
# -----------------------------

def get_lr(step):

    if step >= STEPS:
        return LR_MIN

    ratio = step / STEPS

    cosine = 0.5 * (
        1.0 + math.cos(math.pi * ratio)
    )

    return LR_MIN + (
        LR_MAX - LR_MIN
    ) * cosine


# -----------------------------
# Training
# -----------------------------

print()
print("-" * 64)
print("Starting training...")
print("-" * 64)

start_time = time.time()

best_val = float("inf")
best_step = 0

model.train()

for step in range(1, STEPS + 1):

    lr = get_lr(step)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    x, y = get_batch(train_tokens)

    optimizer.zero_grad(set_to_none=True)

    logits, loss = model(x, y)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0,
    )

    optimizer.step()


    # -------------------------
    # Evaluation
    # -------------------------

    if step == 1 or step % EVAL_INTERVAL == 0:

        losses = estimate_loss()

        train_loss = losses["train"]
        val_loss = losses["val"]

        print(
            f"Step {step:4d}/{STEPS} | "
            f"Train {train_loss:.4f} | "
            f"Val {val_loss:.4f} | "
            f"LR {lr:.6f}"
        )

        # Save best model
        if val_loss < best_val:

            best_val = val_loss
            best_step = step

            torch.save(
                {
                    "version": VERSION,
                    "model_state_dict": model.state_dict(),
                    "vocab_size": vocab_size,
                    "block_size": BLOCK_SIZE,
                    "n_embd": 320,
                    "n_head": 8,
                    "n_layer": 8,
                    "dropout": 0.1,
                    "best_val_loss": best_val,
                    "best_step": best_step,
                },
                BEST_PATH,
            )


# -----------------------------
# Save final model
# -----------------------------

torch.save(
    {
        "version": VERSION,
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "block_size": BLOCK_SIZE,
        "n_embd": 320,
        "n_head": 8,
        "n_layer": 8,
        "dropout": 0.1,
        "best_val_loss": best_val,
        "best_step": best_step,
    },
    FINAL_PATH,
)


# -----------------------------
# Finish
# -----------------------------

elapsed = time.time() - start_time

print()
print("=" * 64)
print("Training complete!")
print("=" * 64)

print(f"Best validation loss: {best_val:.4f}")
print(f"Best step:            {best_step}")
print(f"Training time:        {elapsed / 60:.1f} minutes")

print()
print("Best model:")
print(BEST_PATH)

print()
print("Final model:")
print(FINAL_PATH)

print()
print("User-facing version: Beta 0.5")