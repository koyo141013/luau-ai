from pathlib import Path
import json

import torch
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from src.model_v52 import TinyLuauGPTv52


# ============================================================
# Luau AI v5.2
# User-facing version: Beta 0.5
# ============================================================

VERSION = "Beta 0.5"

ROOT = Path(__file__).resolve().parent

WEB_DIR = ROOT / "web"

TOKENIZER_PATH = ROOT / "model" / "tokenizer_v52.json"
MODEL_PATH = ROOT / "model" / "tiny_luau_gpt_v52_best.pt"


# ============================================================
# Flask
# ============================================================

app = Flask(
    __name__,
    static_folder=str(WEB_DIR)
)

CORS(
    app,
    resources={
        r"/api/*": {
            "origins": "https://koyo141013.github.io"
        }
    }
)


# ============================================================
# Device
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 64)
print("Luau AI v5.2 Web Server")
print("User-facing version: Beta 0.5")
print("=" * 64)

print(f"Device: {device}")

if device == "cuda":
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# Tokenizer
# ============================================================

with open(
    TOKENIZER_PATH,
    "r",
    encoding="utf-8"
) as f:

    tokenizer_data = json.load(f)


vocab = tokenizer_data["vocab"]

stoi = vocab

itos = {
    int(v): k
    for k, v in vocab.items()
}

special_tokens = tokenizer_data.get(
    "special_tokens",
    []
)

print(
    f"Vocabulary size: {len(vocab)}"
)


def encode(text):

    tokens = []

    i = 0

    special_sorted = sorted(
        special_tokens,
        key=len,
        reverse=True,
    )

    while i < len(text):

        matched = False

        for token in special_sorted:

            if text.startswith(
                token,
                i
            ):

                tokens.append(
                    stoi[token]
                )

                i += len(token)

                matched = True

                break

        if matched:
            continue

        if text[i] == " ":

            token = "<SPACE>"

        elif text[i] == "\t":

            token = "<TAB>"

        elif text[i] == "\n":

            token = "<NEWLINE>"

        else:

            token = text[i]

        tokens.append(
            stoi.get(
                token,
                stoi.get(
                    "<UNK>",
                    1
                ),
            )
        )

        i += 1

    return tokens


def decode(ids):

    result = []

    for idx in ids:

        token = itos.get(
            int(idx),
            ""
        )

        if token == "<SPACE>":

            result.append(" ")

        elif token == "<TAB>":

            result.append("\t")

        elif token == "<NEWLINE>":

            result.append("\n")

        elif (
            token.startswith("<")
            and token.endswith(">")
        ):

            continue

        else:

            result.append(token)

    return "".join(result)


# ============================================================
# Model
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False,
)


model = TinyLuauGPTv52(
    vocab_size=checkpoint["vocab_size"],
    block_size=checkpoint["block_size"],
    n_embd=checkpoint["n_embd"],
    n_head=checkpoint["n_head"],
    n_layer=checkpoint["n_layer"],
    dropout=checkpoint["dropout"],
).to(device)


model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


print(
    f"Model: {MODEL_PATH.name}"
)

print(
    f"Parameters: {model.num_parameters():,}"
)


if "best_val_loss" in checkpoint:

    print(
        f"Best Val Loss: "
        f"{checkpoint['best_val_loss']:.4f}"
    )


print()


# ============================================================
# Generation
# ============================================================

MODE_SETTINGS = {

    "CHAT": {
        "temperature": 0.85,
        "max_new_tokens": 120,
    },

    "CODE": {
        "temperature": 0.55,
        "max_new_tokens": 260,
    },

    "EXPLAIN": {
        "temperature": 0.70,
        "max_new_tokens": 220,
    },

    "FIX": {
        "temperature": 0.55,
        "max_new_tokens": 260,
    },
}


@torch.no_grad()
def generate_response(
    mode,
    prompt
):

    settings = MODE_SETTINGS.get(
        mode,
        MODE_SETTINGS["CHAT"],
    )

    input_text = (
        f"<{mode}>\n"
        f"{prompt}\n"
        f"<RESPONSE>\n"
    )

    ids = encode(
        input_text
    )

    x = torch.tensor(
        [ids],
        dtype=torch.long,
        device=device,
    )

    output = model.generate(
        x,
        max_new_tokens=settings[
            "max_new_tokens"
        ],
        temperature=settings[
            "temperature"
        ],
        top_k=40,
        stop_token_id=stoi.get(
            "<END>"
        ),
    )

    output_ids = output[0].tolist()

    response_ids = (
        output_ids[len(ids):]
    )

    response = decode(
        response_ids
    )

    return response.strip()


# ============================================================
# API
# ============================================================

@app.get("/")
def index():

    return send_from_directory(
        WEB_DIR,
        "index.html",
    )


@app.get("/<path:path>")
def static_files(path):

    return send_from_directory(
        WEB_DIR,
        path,
    )


@app.get("/api/status")
def status():

    return jsonify({

        "version": VERSION,

        "model": "Luau AI v5.2",

        "device": device,

        "gpu": (
            torch.cuda.get_device_name(0)
            if device == "cuda"
            else None
        ),

        "parameters":
            model.num_parameters(),

        "best_val_loss":
            checkpoint.get(
                "best_val_loss"
            ),

    })


@app.post("/api/generate")
def generate():

    data = request.get_json(
        silent=True
    ) or {}


    prompt = str(
        data.get(
            "prompt",
            ""
        )
    ).strip()


    mode = str(
        data.get(
            "mode",
            "CHAT"
        )
    ).upper()


    if not prompt:

        return jsonify({
            "error":
                "Prompt is empty."
        }), 400


    if mode not in MODE_SETTINGS:

        mode = "CHAT"


    try:

        response = generate_response(
            mode,
            prompt,
        )

        return jsonify({

            "response":
                response,

            "mode":
                mode,

            "version":
                VERSION,

        })


    except Exception as e:

        print(
            "Generation error:",
            e
        )

        return jsonify({

            "error":
                str(e)

        }), 500


# ============================================================
# Start
# ============================================================

if __name__ == "__main__":

    print("=" * 64)

    print(
        "Luau AI v5.2 Web Server Ready"
    )

    print("=" * 64)

    print()

    print("Open:")

    print(
        "http://127.0.0.1:8000/"
    )

    print()

    app.run(
        host="127.0.0.1",
        port=8000,
        debug=False,
    )