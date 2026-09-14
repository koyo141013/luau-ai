import json
from pathlib import Path

import torch

from model_v52 import TinyLuauGPTv52


# ============================================================
# Luau AI v5.2
# User-facing version: Beta 0.5
# ============================================================

VERSION = "5.2-beta.0.5"

ROOT = Path(__file__).resolve().parent.parent

TOKENIZER_PATH = ROOT / "model" / "tokenizer_v52.json"
BEST_MODEL_PATH = ROOT / "model" / "tiny_luau_gpt_v52_best.pt"
FINAL_MODEL_PATH = ROOT / "model" / "tiny_luau_gpt_v52.pt"


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 64)
print("Luau AI v5.2")
print("User-facing version: Beta 0.5")
print("=" * 64)

print(f"Device: {device}")

if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ------------------------------------------------------------
# Load tokenizer
# ------------------------------------------------------------

with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
    tokenizer_data = json.load(f)

vocab = tokenizer_data["vocab"]

# JSON은 token -> id 형태
stoi = vocab
itos = {int(v): k for k, v in vocab.items()}

special_tokens = tokenizer_data.get("special_tokens", [])

print(f"Vocabulary size: {len(vocab)}")


def encode(text):
    """
    v5.2 tokenizer와 동일한 방식으로
    텍스트를 토큰 ID로 변환한다.
    """

    tokens = []

    i = 0

    # 긴 특수 토큰부터 확인
    special_sorted = sorted(
        special_tokens,
        key=len,
        reverse=True
    )

    while i < len(text):

        matched = False

        # Special token
        for token in special_sorted:

            if text.startswith(token, i):

                tokens.append(stoi[token])
                i += len(token)

                matched = True
                break

        if matched:
            continue

        # Whitespace
        if text[i] == " ":
            token = "<SPACE>"

        elif text[i] == "\t":
            token = "<TAB>"

        elif text[i] == "\n":
            token = "<NEWLINE>"

        else:
            token = text[i]

        if token in stoi:
            tokens.append(stoi[token])
        else:
            tokens.append(stoi.get("<UNK>", 1))

        i += 1

    return tokens


def decode(ids):
    """
    토큰 ID -> 실제 텍스트
    """

    result = []

    for idx in ids:

        token = itos.get(int(idx), "")

        if token == "<SPACE>":
            result.append(" ")

        elif token == "<TAB>":
            result.append("\t")

        elif token == "<NEWLINE>":
            result.append("\n")

        elif token.startswith("<") and token.endswith(">"):
            # 모델 제어 토큰 제거
            continue

        else:
            result.append(token)

    return "".join(result)


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

if BEST_MODEL_PATH.exists():
    MODEL_PATH = BEST_MODEL_PATH
else:
    MODEL_PATH = FINAL_MODEL_PATH

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
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

print()
print(f"Model: {MODEL_PATH.name}")
print(
    f"Parameters: "
    f"{model.num_parameters():,}"
)

if "best_val_loss" in checkpoint:
    print(
        f"Best Val Loss: "
        f"{checkpoint['best_val_loss']:.4f}"
    )

print()


# ------------------------------------------------------------
# Mode settings
# ------------------------------------------------------------

MODE_SETTINGS = {

    "1": {
        "name": "CHAT",
        "temperature": 0.85,
        "max_new_tokens": 120,
    },

    "2": {
        "name": "CODE",
        "temperature": 0.55,
        "max_new_tokens": 260,
    },

    "3": {
        "name": "EXPLAIN",
        "temperature": 0.70,
        "max_new_tokens": 220,
    },

    "4": {
        "name": "FIX",
        "temperature": 0.55,
        "max_new_tokens": 260,
    },
}


# ------------------------------------------------------------
# Generation
# ------------------------------------------------------------

@torch.no_grad()
def generate_response(
    mode,
    prompt,
    temperature,
    max_new_tokens,
    top_k=40,
):

    mode_token = f"<{mode}>"

    response_token = "<RESPONSE>"
    end_token = "<END>"

    input_text = (
        mode_token
        + "\n"
        + prompt
        + "\n"
        + response_token
        + "\n"
    )

    ids = encode(input_text)

    x = torch.tensor(
        [ids],
        dtype=torch.long,
        device=device,
    )

    generated = model.generate(
        x,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        stop_token_id=stoi.get(end_token),
    )

    new_ids = generated[0].tolist()

    # 입력 부분 제거
    response_ids = new_ids[len(ids):]

    return decode(response_ids).strip()


# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------

while True:

    print("=" * 64)
    print("Select mode")
    print("=" * 64)

    print("1. CHAT")
    print("2. CODE")
    print("3. EXPLAIN")
    print("4. FIX")
    print("Q. EXIT")

    mode_input = input("\nMode [1-4] > ").strip()

    if mode_input.lower() == "q":
        print("Goodbye!")
        break

    if mode_input not in MODE_SETTINGS:
        print("1~4 중에서 선택해줘!")
        continue

    settings = MODE_SETTINGS[mode_input]

    mode = settings["name"]

    print()
    print(f"Mode: {mode}")

    prompt = input("Prompt > ").strip()

    if not prompt:
        print("Prompt가 비어 있어!")
        continue

    print()
    print(f"AI ({mode}) >")
    print("-" * 64)

    try:

        response = generate_response(
            mode=mode,
            prompt=prompt,
            temperature=settings["temperature"],
            max_new_tokens=settings["max_new_tokens"],
        )

        if not response:
            response = "(응답을 생성하지 못했어.)"

        print(response)

    except Exception as e:

        print()
        print("Generation error:")
        print(e)

    print()