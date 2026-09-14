import json
import math
import time
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_

from model_v5 import TinyLuauGPTv5


# ============================================================
# Paths
# ============================================================

TOKENS_PATH = Path("dataset/tokens_v5.pt")
TOKENIZER_PATH = Path("model/tokenizer_v5.json")

BEST_MODEL_PATH = Path(
    "model/tiny_luau_gpt_v5_best.pt"
)

FINAL_MODEL_PATH = Path(
    "model/tiny_luau_gpt_v5.pt"
)


# ============================================================
# Training settings
# ============================================================

BLOCK_SIZE = 256

N_EMBD = 320
N_HEAD = 8
N_LAYER = 8
DROPOUT = 0.1

BATCH_SIZE = 16

MAX_STEPS = 5000

LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 3e-5

WEIGHT_DECAY = 0.1

BETAS = (0.9, 0.95)

GRAD_CLIP = 1.0

EVAL_INTERVAL = 100
EVAL_BATCHES = 20

TRAIN_RATIO = 0.90

SEED = 42


# ============================================================
# Device
# ============================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    device = torch.device("cuda")

    torch.cuda.manual_seed_all(SEED)

else:
    device = torch.device("cpu")


print("=" * 70)
print("TinyLuauGPT v5 Training")
print("=" * 70)

print()
print(f"Device: {device}")

if device.type == "cuda":
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    print(
        f"CUDA: {torch.version.cuda}"
    )


# ============================================================
# Load tokenizer
# ============================================================

if not TOKENIZER_PATH.exists():
    raise FileNotFoundError(
        f"Tokenizer not found: {TOKENIZER_PATH}"
    )

with open(
    TOKENIZER_PATH,
    "r",
    encoding="utf-8",
) as f:
    tokenizer_data = json.load(f)


vocab = tokenizer_data["vocab"]

vocab_size = len(vocab)

print()
print(f"Vocabulary size: {vocab_size:,}")

special_tokens = tokenizer_data.get(
    "special_tokens",
    [],
)

print()
print("Special tokens:")

for token in special_tokens:
    print(
        f"  {token:12s} -> {vocab[token]}"
    )


# ============================================================
# Load dataset
# ============================================================

if not TOKENS_PATH.exists():
    raise FileNotFoundError(
        f"Token dataset not found: {TOKENS_PATH}"
    )


tokens = torch.load(
    TOKENS_PATH,
    map_location="cpu",
)


if tokens.dtype != torch.long:
    tokens = tokens.long()


total_tokens = len(tokens)

split_index = int(
    total_tokens * TRAIN_RATIO
)

train_data = tokens[:split_index]
val_data = tokens[split_index:]


print()
print(f"Total tokens:       {total_tokens:,}")
print(f"Training tokens:    {len(train_data):,}")
print(f"Validation tokens:  {len(val_data):,}")


# ============================================================
# Model
# ============================================================

model = TinyLuauGPTv5(
    vocab_size=vocab_size,
    block_size=BLOCK_SIZE,
    n_embd=N_EMBD,
    n_head=N_HEAD,
    n_layer=N_LAYER,
    dropout=DROPOUT,
).to(device)


print()
print(
    f"Parameters: "
    f"{model.num_parameters():,}"
)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    betas=BETAS,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# Learning rate schedule
# ============================================================

def get_learning_rate(step):

    if step >= MAX_STEPS:
        return MIN_LEARNING_RATE

    progress = step / MAX_STEPS

    cosine = 0.5 * (
        1.0
        + math.cos(
            math.pi * progress
        )
    )

    return (
        MIN_LEARNING_RATE
        + (
            LEARNING_RATE
            - MIN_LEARNING_RATE
        )
        * cosine
    )


# ============================================================
# Batch
# ============================================================

def get_batch(data):

    max_start = len(data) - BLOCK_SIZE - 1

    ix = torch.randint(
        0,
        max_start,
        (BATCH_SIZE,),
    )

    x = torch.stack(
        [
            data[i:i + BLOCK_SIZE]
            for i in ix.tolist()
        ]
    )

    y = torch.stack(
        [
            data[i + 1:i + BLOCK_SIZE + 1]
            for i in ix.tolist()
        ]
    )

    return (
        x.to(device),
        y.to(device),
    )


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def estimate_loss():

    model.eval()

    train_losses = []
    val_losses = []

    for _ in range(EVAL_BATCHES):

        x, y = get_batch(train_data)

        _, loss = model(
            x,
            y,
        )

        train_losses.append(
            loss.item()
        )

        x, y = get_batch(val_data)

        _, loss = model(
            x,
            y,
        )

        val_losses.append(
            loss.item()
        )

    model.train()

    train_loss = sum(train_losses) / len(
        train_losses
    )

    val_loss = sum(val_losses) / len(
        val_losses
    )

    return train_loss, val_loss


# ============================================================
# Training
# ============================================================

best_val_loss = float("inf")
best_step = 0

start_time = time.time()

print()
print("=" * 70)
print("Starting training...")
print("=" * 70)

print()
print(f"Steps:       {MAX_STEPS:,}")
print(f"Batch size:  {BATCH_SIZE}")
print(f"Block size:  {BLOCK_SIZE}")
print(f"LR:          {LEARNING_RATE}")
print(f"Min LR:      {MIN_LEARNING_RATE}")

print()


for step in range(1, MAX_STEPS + 1):

    # Learning rate
    lr = get_learning_rate(step)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # Batch
    x, y = get_batch(train_data)

    # Forward
    logits, loss = model(
        x,
        y,
    )

    # Backward
    optimizer.zero_grad(
        set_to_none=True
    )

    loss.backward()

    # Gradient clipping
    clip_grad_norm_(
        model.parameters(),
        GRAD_CLIP,
    )

    # Update
    optimizer.step()

    # Evaluation
    if (
        step == 1
        or step % EVAL_INTERVAL == 0
        or step == MAX_STEPS
    ):

        train_loss, val_loss = estimate_loss()

        print(
            f"Step {step:5d}/{MAX_STEPS} "
            f"| Train {train_loss:.4f} "
            f"| Val {val_loss:.4f} "
            f"| LR {lr:.2e}"
        )

        # Save best
        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_step = step

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "vocab_size": vocab_size,
                "block_size": BLOCK_SIZE,
                "n_embd": N_EMBD,
                "n_head": N_HEAD,
                "n_layer": N_LAYER,
                "dropout": DROPOUT,
                "step": step,
                "val_loss": val_loss,
            }

            BEST_MODEL_PATH.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            torch.save(
                checkpoint,
                BEST_MODEL_PATH,
            )


# ============================================================
# Save final model
# ============================================================

elapsed = time.time() - start_time

final_checkpoint = {
    "model_state_dict": model.state_dict(),
    "vocab_size": vocab_size,
    "block_size": BLOCK_SIZE,
    "n_embd": N_EMBD,
    "n_head": N_HEAD,
    "n_layer": N_LAYER,
    "dropout": DROPOUT,
    "step": MAX_STEPS,
    "val_loss": val_loss,
}

FINAL_MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

torch.save(
    final_checkpoint,
    FINAL_MODEL_PATH,
)


# ============================================================
# Summary
# ============================================================

print()
print("=" * 70)
print("Training complete!")
print("=" * 70)

print()
print(
    f"Best validation loss: "
    f"{best_val_loss:.4f}"
)

print(
    f"Best step:            "
    f"{best_step:,}"
)

print(
    f"Training time:        "
    f"{elapsed / 60:.1f} minutes"
)

print()
print("Best model:")
print(
    BEST_MODEL_PATH.resolve()
)

print()
print("Final model:")
print(
    FINAL_MODEL_PATH.resolve()
)

print()
print("=" * 70)
print("TinyLuauGPT v5 training finished!")
print("=" * 70)
