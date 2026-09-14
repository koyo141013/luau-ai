import json
from pathlib import Path

import torch

from model_v5 import TinyLuauGPTv5


# ============================================================
# TinyLuauGPT v5 Generator
# ============================================================

TOKENIZER_PATH = Path("model/tokenizer_v5.json")
BEST_MODEL_PATH = Path("model/tiny_luau_gpt_v5_best.pt")
FINAL_MODEL_PATH = Path("model/tiny_luau_gpt_v5.pt")

BLOCK_SIZE = 256
N_EMBD = 320
N_HEAD = 8
N_LAYER = 8
DROPOUT = 0.1


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("TinyLuauGPT v5 Generator")
print("=" * 70)
print()

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)


# ============================================================
# Load tokenizer
# ============================================================

with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
    tokenizer_data = json.load(f)

vocab = tokenizer_data["vocab"]
special_tokens = tokenizer_data["special_tokens"]


# vocab: token -> id
token_to_id = {}

for token, token_id in vocab.items():
    token_to_id[str(token)] = int(token_id)


# special_tokens is a list in v5
for token in special_tokens:
    if token not in token_to_id:
        print(
            "WARNING: special token missing:",
            token
        )


# id -> token
id_to_token = {
    token_id: token
    for token, token_id in token_to_id.items()
}

vocab_size = max(id_to_token.keys()) + 1


print()
print("Vocabulary size:", vocab_size)


# ============================================================
# Special token IDs
# ============================================================

PAD = token_to_id["<PAD>"]
UNK = token_to_id["<UNK>"]
BOS = token_to_id["<BOS>"]
EOS = token_to_id["<EOS>"]

CHAT = token_to_id["<CHAT>"]
CODE = token_to_id["<CODE>"]
EXPLAIN = token_to_id["<EXPLAIN>"]
FIX = token_to_id["<FIX>"]
RESPONSE = token_to_id["<RESPONSE>"]
END = token_to_id["<END>"]


print()
print("Special tokens:")

for name in [
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
]:
    print(
        f"  {name:<12} -> {token_to_id[name]}"
    )


# ============================================================
# Load model
# ============================================================

if BEST_MODEL_PATH.exists():
    model_path = BEST_MODEL_PATH
else:
    model_path = FINAL_MODEL_PATH


print()
print("Loading model:")
print(model_path)


model = TinyLuauGPTv5(
    vocab_size=vocab_size,
    block_size=BLOCK_SIZE,
    n_embd=N_EMBD,
    n_head=N_HEAD,
    n_layer=N_LAYER,
    dropout=DROPOUT,
)


checkpoint = torch.load(
    model_path,
    map_location=device,
    weights_only=False,
)


# Support different checkpoint formats
if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    elif "state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["state_dict"]
        )

    else:
        model.load_state_dict(checkpoint)

else:

    model.load_state_dict(checkpoint)


model.to(device)
model.eval()


print()
print("Parameters:", model.num_parameters())
print("Model loaded successfully!")


# ============================================================
# Encode
# ============================================================

def encode_text(text):

    tokens = []
    i = 0

    ordered_special = sorted(
        special_tokens,
        key=len,
        reverse=True
    )

    while i < len(text):

        # ----------------------------------------
        # Special tokens
        # ----------------------------------------

        found_special = False

        for special in ordered_special:

            if text.startswith(special, i):

                tokens.append(
                    token_to_id[special]
                )

                i += len(special)
                found_special = True

                break

        if found_special:
            continue


        ch = text[i]


        # ----------------------------------------
        # Newline
        # ----------------------------------------

        if ch == "\n":

            if ch in token_to_id:

                tokens.append(
                    token_to_id[ch]
                )

            else:

                tokens.append(UNK)

            i += 1
            continue


        # ----------------------------------------
        # Whitespace
        # ----------------------------------------

        if ch.isspace():

            i += 1
            continue


        # ----------------------------------------
        # Word / number / identifier
        # ----------------------------------------

        if ch.isalnum() or ch == "_":

            j = i + 1

            while j < len(text):

                c = text[j]

                if c.isalnum() or c == "_":
                    j += 1
                else:
                    break

            word = text[i:j]


            if word in token_to_id:

                tokens.append(
                    token_to_id[word]
                )

            else:

                # Character fallback
                for c in word:

                    if c in token_to_id:

                        tokens.append(
                            token_to_id[c]
                        )

                    else:

                        tokens.append(UNK)


            i = j
            continue


        # ----------------------------------------
        # Symbols
        # ----------------------------------------

        if ch in token_to_id:

            tokens.append(
                token_to_id[ch]
            )

        else:

            tokens.append(UNK)

        i += 1


    return tokens


# ============================================================
# Decode
# ============================================================

def decode_tokens(ids):

    result = []

    ignored_tokens = {
        "<PAD>",
        "<BOS>",
        "<EOS>",
        "<CHAT>",
        "<CODE>",
        "<EXPLAIN>",
        "<FIX>",
        "<RESPONSE>",
    }


    for token_id in ids:

        token = id_to_token.get(
            int(token_id),
            ""
        )


        if token == "<END>":
            break


        if token in ignored_tokens:
            continue


        result.append(token)


    return "".join(result)


# ============================================================
# Generate
# ============================================================

@torch.no_grad()
def generate(
    prompt,
    mode,
    temperature,
    top_k,
    max_new_tokens
):

    formatted = (
        f"{mode}\n"
        f"{prompt}\n"
        f"<RESPONSE>\n"
    )


    input_ids = encode_text(
        formatted
    )


    if not input_ids:
        return ""


    # Keep inside model context
    input_ids = input_ids[
        -(BLOCK_SIZE - 1):
    ]


    x = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=device
    )


    prompt_length = x.shape[1]


    # ========================================================
    # Token generation
    # ========================================================

    for _ in range(max_new_tokens):

        x_cond = x[:, -BLOCK_SIZE:]


        # IMPORTANT:
        # model_v5.forward() returns:
        #
        #     logits, loss
        #
        # so we must unpack the tuple.

        output = model(x_cond)


        if isinstance(output, tuple):

            logits = output[0]

        else:

            logits = output


        logits = logits[:, -1, :]


        # ----------------------------------------
        # Temperature
        # ----------------------------------------

        logits = logits / max(
            temperature,
            0.01
        )


        # ----------------------------------------
        # Top-K sampling
        # ----------------------------------------

        if top_k > 0:

            k = min(
                top_k,
                logits.size(-1)
            )


            values, indices = torch.topk(
                logits,
                k
            )


            filtered_logits = torch.full_like(
                logits,
                float("-inf")
            )


            filtered_logits.scatter_(
                1,
                indices,
                values
            )


            logits = filtered_logits


        # ----------------------------------------
        # Probability
        # ----------------------------------------

        probabilities = torch.softmax(
            logits,
            dim=-1
        )


        # ----------------------------------------
        # Sample
        # ----------------------------------------

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )


        x = torch.cat(
            [x, next_token],
            dim=1
        )


        # ----------------------------------------
        # END token
        # ----------------------------------------

        if next_token.item() == END:
            break


    # Only decode newly generated tokens
    generated_ids = x[
        0,
        prompt_length:
    ].tolist()


    return decode_tokens(
        generated_ids
    )


# ============================================================
# Main
# ============================================================

def main():

    while True:

        print()
        print("=" * 40)
        print("1. 💬 CHAT")
        print("2. 💻 CODE")
        print("3. 📖 EXPLAIN")
        print("4. 🔧 FIX")
        print("5. ❌ EXIT")
        print("=" * 40)


        choice = input(
            "Mode [1-5] > "
        ).strip()


        # ----------------------------------------
        # Exit
        # ----------------------------------------

        if choice == "5":

            print()
            print("Bye! 👋")

            break


        # ----------------------------------------
        # Invalid menu
        # ----------------------------------------

        if choice not in {
            "1",
            "2",
            "3",
            "4"
        }:

            print()
            print("1~5 중에서 선택해줘!")

            continue


        # ----------------------------------------
        # Prompt
        # ----------------------------------------

        prompt = input(
            "\nPrompt > "
        ).strip()


        if not prompt:

            print()
            print("프롬프트를 입력해줘!")

            continue


        # ----------------------------------------
        # Mode settings
        # ----------------------------------------

        if choice == "1":

            mode = "<CHAT>"
            temperature = 0.85
            max_tokens = 120


        elif choice == "2":

            mode = "<CODE>"
            temperature = 0.55
            max_tokens = 240


        elif choice == "3":

            mode = "<EXPLAIN>"
            temperature = 0.70
            max_tokens = 200


        else:

            mode = "<FIX>"
            temperature = 0.55
            max_tokens = 240


        # ----------------------------------------
        # Generate
        # ----------------------------------------

        print()
        print(f"AI ({mode}) >")
        print()


        response = generate(
            prompt=prompt,
            mode=mode,
            temperature=temperature,
            top_k=40,
            max_new_tokens=max_tokens
        )


        # ----------------------------------------
        # Result
        # ----------------------------------------

        if not response.strip():

            print(
                "(응답을 생성하지 못했어.)"
            )

        else:

            print(response)


        print()


# ============================================================
# Start
# ============================================================

if __name__ == "__main__":
    main()