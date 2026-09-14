from pathlib import Path
import json
import math
import time

import torch
import torch.nn as nn

from model_v51 import TinyLuauGPTv51


# ============================================================
# Paths
# ============================================================

TOKENS_PATH = Path("dataset/tokens_v51.pt")
TOKENIZER_PATH = Path("model/tokenizer_v51.json")

BEST_MODEL_PATH = Path("model/tiny_luau_gpt_v51_best.pt")
FINAL_MODEL_PATH = Path("model/tiny_luau_gpt_v51.pt")


# ============================================================
# Training settings
# ============================================================

BLOCK_SIZE = 256
BATCH_SIZE = 16

N_EMBD = 320
N_HEAD = 8
N_LAYER = 8
DROPOUT = 0.1

MAX_STEPS = 5000

LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 3e-5

WEIGHT_DECAY = 0.1

EVAL_INTERVAL = 100
EVAL_BATCHES = 20

TRAIN_RATIO = 0.90

GRAD_CLIP = 1.0


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Learning rate schedule
# ============================================================

def get_lr(step):

    if step >= MAX_STEPS:
        return MIN_LEARNING_RATE

    progress = step / MAX_STEPS

    cosine = 0.5 * (
        1.0 + math.cos(
            math.pi * progress
        )
    )

    return (
        MIN_LEARNING_RATE
        +
        (
            LEARNING_RATE
            -
            MIN_LEARNING_RATE
        )
        * cosine
    )


# ============================================================
# Random batch
# ============================================================

def get_batch(tokens, batch_size, block_size):

    max_start = len(tokens) - block_size - 1

    starts = torch.randint(
        0,
        max_start,
        (batch_size,)
    )

    x = torch.stack(
        [
            tokens[
                start:
                start + block_size
            ]
            for start in starts
        ]
    )

    y = torch.stack(
        [
            tokens[
                start + 1:
                start + block_size + 1
            ]
            for start in starts
        ]
    )

    return (
        x.to(device),
        y.to(device)
    )


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def estimate_loss(model, train_tokens, val_tokens):

    model.eval()

    results = {}

    for name, tokens in [
        ("train", train_tokens),
        ("val", val_tokens),
    ]:

        losses = []

        for _ in range(EVAL_BATCHES):

            x, y = get_batch(
                tokens,
                BATCH_SIZE,
                BLOCK_SIZE
            )

            _, loss = model(
                x,
                y
            )

            losses.append(
                loss.item()
            )

        results[name] = (
            sum(losses)
            /
            len(losses)
        )

    model.train()

    return results


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 64)
    print("TinyLuauGPT v5.1 Training")
    print("=" * 64)


    # --------------------------------------------------------
    # Device info
    # --------------------------------------------------------

    print()

    print(
        "Device:",
        device
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "CUDA:",
            torch.version.cuda
        )

    else:

        print(
            "WARNING: CUDA unavailable!"
        )


    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    with open(
        TOKENIZER_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        tokenizer = json.load(f)


    vocab = tokenizer["vocab"]

    vocab_size = len(vocab)


    print()

    print(
        "Vocabulary size:",
        vocab_size
    )


    print(
        "Special tokens:"
    )

    for token in tokenizer["special_tokens"]:

        print(
            f"  {token} -> {vocab[token]}"
        )


    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    tokens = torch.load(
        TOKENS_PATH,
        map_location="cpu"
    )


    if not torch.is_tensor(tokens):

        tokens = torch.tensor(
            tokens,
            dtype=torch.long
        )


    tokens = tokens.long()


    print()

    print(
        "Total tokens:",
        f"{len(tokens):,}"
    )


    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    split = int(
        len(tokens)
        *
        TRAIN_RATIO
    )


    train_tokens = tokens[:split]

    val_tokens = tokens[split:]


    print(
        "Training tokens:",
        f"{len(train_tokens):,}"
    )

    print(
        "Validation tokens:",
        f"{len(val_tokens):,}"
    )


    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = TinyLuauGPTv51(
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        dropout=DROPOUT,
    )


    model = model.to(device)


    parameters = model.num_parameters()


    print()

    print(
        "Parameters:",
        f"{parameters:,}"
    )

    print(
        "Steps:",
        MAX_STEPS
    )

    print(
        "Batch size:",
        BATCH_SIZE
    )

    print(
        "Block size:",
        BLOCK_SIZE
    )

    print(
        "Learning rate:",
        LEARNING_RATE,
        "->",
        MIN_LEARNING_RATE
    )


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY,
    )


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_loss = float("inf")
    best_step = 0

    start_time = time.time()


    print()
    print("=" * 64)
    print("Training started!")
    print("=" * 64)
    print()


    for step in range(
        1,
        MAX_STEPS + 1
    ):

        # ----------------------------------------------------
        # Learning rate
        # ----------------------------------------------------

        lr = get_lr(
            step - 1
        )

        for param_group in optimizer.param_groups:

            param_group["lr"] = lr


        # ----------------------------------------------------
        # Batch
        # ----------------------------------------------------

        x, y = get_batch(
            train_tokens,
            BATCH_SIZE,
            BLOCK_SIZE
        )


        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        logits, loss = model(
            x,
            y
        )


        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()


        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP
        )


        optimizer.step()


        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        if (
            step == 1
            or step % EVAL_INTERVAL == 0
        ):

            losses = estimate_loss(
                model,
                train_tokens,
                val_tokens
            )


            train_loss = losses["train"]
            val_loss = losses["val"]


            print(
                f"Step {step:4d}/{MAX_STEPS} "
                f"| Train {train_loss:.4f} "
                f"| Val {val_loss:.4f} "
                f"| LR {lr:.2e}"
            )


            # ------------------------------------------------
            # Save best
            # ------------------------------------------------

            if val_loss < best_val_loss:

                best_val_loss = val_loss
                best_step = step


                torch.save(
                    {
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
                    },
                    BEST_MODEL_PATH
                )


    # ========================================================
    # Save final model
    # ========================================================

    torch.save(
        {
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
                MAX_STEPS,

            "val_loss":
                val_loss,
        },
        FINAL_MODEL_PATH
    )


    # ========================================================
    # Done
    # ========================================================

    elapsed = (
        time.time()
        -
        start_time
    )


    print()
    print("=" * 64)
    print("Training complete!")
    print("=" * 64)

    print()

    print(
        "Best validation loss:",
        f"{best_val_loss:.4f}"
    )

    print(
        "Best step:",
        best_step
    )

    print(
        "Training time:",
        f"{elapsed / 60:.1f} minutes"
    )

    print()

    print(
        "Best model:"
    )

    print(
        BEST_MODEL_PATH
    )

    print()

    print(
        "Final model:"
    )

    print(
        FINAL_MODEL_PATH
    )

    print()
    print("=" * 64)


if __name__ == "__main__":
    main()