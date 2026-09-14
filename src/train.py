from pathlib import Path
import math
import time

import torch
from torch.utils.data import Dataset, DataLoader

from model import TinyLuauGPT
from tokenizer import LuauTokenizer


# =========================================================
# Config
# =========================================================

TOKENS_PATH = Path("dataset/tokens.pt")
TOKENIZER_PATH = Path("model/tokenizer.json")

BEST_MODEL_PATH = Path("model/tiny_luau_gpt_v4_best.pt")
FINAL_MODEL_PATH = Path("model/tiny_luau_gpt_v4.pt")

BLOCK_SIZE = 256

N_EMBD = 320
N_HEAD = 8
N_LAYER = 8
DROPOUT = 0.1

BATCH_SIZE = 16

STEPS = 5000

LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 3e-5

WEIGHT_DECAY = 0.1

VAL_RATIO = 0.10

EVAL_INTERVAL = 100
EVAL_STEPS = 20

GRAD_CLIP = 1.0

SEED = 42


# =========================================================
# Device
# =========================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# =========================================================
# Dataset
# =========================================================

class TokenDataset(Dataset):
    def __init__(
        self,
        tokens,
        block_size,
    ):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, index):
        x = self.tokens[
            index:index + self.block_size
        ]

        y = self.tokens[
            index + 1:index + self.block_size + 1
        ]

        return x, y


# =========================================================
# Learning rate schedule
# =========================================================

def get_learning_rate(step):
    if step >= STEPS:
        return MIN_LEARNING_RATE

    progress = step / STEPS

    cosine = (
        0.5
        * (
            1.0
            + math.cos(
                math.pi * progress
            )
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


# =========================================================
# Evaluation
# =========================================================

@torch.no_grad()
def estimate_loss(
    model,
    train_loader,
    val_loader,
):
    model.eval()

    results = {}

    for name, loader in [
        ("train", train_loader),
        ("val", val_loader),
    ]:

        losses = []

        iterator = iter(loader)

        for _ in range(EVAL_STEPS):

            try:
                x, y = next(iterator)

            except StopIteration:
                iterator = iter(loader)
                x, y = next(iterator)

            x = x.to(device)
            y = y.to(device)

            _, loss = model(
                x,
                y,
            )

            losses.append(
                loss.item()
            )

        results[name] = (
            sum(losses)
            / len(losses)
        )

    model.train()

    return results


# =========================================================
# Main
# =========================================================

def main():

    print(
        "=== TinyLuauGPT v4 Training ==="
    )
    print()

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print()

    # -----------------------------------------------------
    # Load tokenizer
    # -----------------------------------------------------

    print(
        "Loading tokenizer..."
    )

    tokenizer = LuauTokenizer.load(
        TOKENIZER_PATH
    )

    vocab_size = tokenizer.vocab_size

    print(
        f"Vocabulary size: {vocab_size}"
    )

    # -----------------------------------------------------
    # Load tokens
    # -----------------------------------------------------

    print(
        "Loading dataset..."
    )

    tokens = torch.load(
        TOKENS_PATH,
        weights_only=True,
    )

    tokens = tokens.long()

    print(
        f"Total tokens: {len(tokens):,}"
    )

    # -----------------------------------------------------
    # Train / validation split
    # -----------------------------------------------------

    split_index = int(
        len(tokens)
        * (1.0 - VAL_RATIO)
    )

    train_tokens = tokens[
        :split_index
    ]

    val_tokens = tokens[
        split_index:
    ]

    print(
        f"Training tokens: "
        f"{len(train_tokens):,}"
    )

    print(
        f"Validation tokens: "
        f"{len(val_tokens):,}"
    )

    # -----------------------------------------------------
    # Datasets
    # -----------------------------------------------------

    train_dataset = TokenDataset(
        train_tokens,
        BLOCK_SIZE,
    )

    val_dataset = TokenDataset(
        val_tokens,
        BLOCK_SIZE,
    )

    # -----------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    print()
    print(
        "Creating model..."
    )

    model = TinyLuauGPT(
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        dropout=DROPOUT,
    )

    model = model.to(device)

    parameter_count = (
        model.num_parameters()
    )

    print(
        f"Parameters: "
        f"{parameter_count:,}"
    )

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    best_val_loss = float("inf")

    start_time = time.time()

    print()
    print(
        f"Training steps: {STEPS:,}"
    )
    print()

    train_iterator = iter(
        train_loader
    )

    for step in range(1, STEPS + 1):

        # -------------------------------------------------
        # Get batch
        # -------------------------------------------------

        try:
            x, y = next(
                train_iterator
            )

        except StopIteration:
            train_iterator = iter(
                train_loader
            )

            x, y = next(
                train_iterator
            )

        x = x.to(
            device,
            non_blocking=True,
        )

        y = y.to(
            device,
            non_blocking=True,
        )

        # -------------------------------------------------
        # Learning rate
        # -------------------------------------------------

        lr = get_learning_rate(
            step - 1
        )

        for group in optimizer.param_groups:
            group["lr"] = lr

        # -------------------------------------------------
        # Forward
        # -------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        _, loss = model(
            x,
            y,
        )

        # -------------------------------------------------
        # Backward
        # -------------------------------------------------

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        optimizer.step()

        # -------------------------------------------------
        # Evaluation
        # -------------------------------------------------

        if (
            step == 1
            or step % EVAL_INTERVAL == 0
            or step == STEPS
        ):

            losses = estimate_loss(
                model,
                train_loader,
                val_loader,
            )

            train_loss = losses[
                "train"
            ]

            val_loss = losses[
                "val"
            ]

            elapsed = (
                time.time()
                - start_time
            )

            print(
                f"Step {step:4d}/{STEPS} "
                f"| Train: {train_loss:.4f} "
                f"| Val: {val_loss:.4f} "
                f"| LR: {lr:.2e} "
                f"| {elapsed:.0f}s"
            )

            # -------------------------------------------------
            # Save best model
            # -------------------------------------------------

            if val_loss < best_val_loss:

                best_val_loss = val_loss

                checkpoint = {
                    "model_state_dict":
                        model.state_dict(),

                    "vocab_size":
                        vocab_size,

                    "block_size":
                        BLOCK_SIZE,

                    "n_embd":
                        N_EMBD,

                    "n_head":
                        N_HEAD,

                    "n_layer":
                        N_LAYER,

                    "dropout":
                        DROPOUT,

                    "step":
                        step,

                    "val_loss":
                        val_loss,
                }

                torch.save(
                    checkpoint,
                    BEST_MODEL_PATH,
                )

                print(
                    f"  ★ Best model saved "
                    f"(Val: {val_loss:.4f})"
                )

    # -----------------------------------------------------
    # Save final model
    # -----------------------------------------------------

    final_checkpoint = {
        "model_state_dict":
            model.state_dict(),

        "vocab_size":
            vocab_size,

        "block_size":
            BLOCK_SIZE,

        "n_embd":
            N_EMBD,

        "n_head":
            N_HEAD,

        "n_layer":
            N_LAYER,

        "dropout":
            DROPOUT,

        "step":
            STEPS,

        "val_loss":
            best_val_loss,
    }

    torch.save(
        final_checkpoint,
        FINAL_MODEL_PATH,
    )

    # -----------------------------------------------------
    # Done
    # -----------------------------------------------------

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print(
        "========================================"
    )

    print(
        "Training complete!"
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.4f}"
    )

    print(
        f"Training time: "
        f"{elapsed / 60:.1f} minutes"
    )

    print()
    print(
        f"Best model:"
    )

    print(
        f"  {BEST_MODEL_PATH}"
    )

    print(
        f"Final model:"
    )

    print(
        f"  {FINAL_MODEL_PATH}"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()