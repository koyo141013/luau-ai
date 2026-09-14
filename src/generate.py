from pathlib import Path

import torch

from model import TinyLuauGPT
from tokenizer import LuauTokenizer


# =========================================================
# Config
# =========================================================

TOKENIZER_PATH = Path("model/tokenizer.json")

BEST_MODEL_PATH = Path(
    "model/tiny_luau_gpt_v4_best.pt"
)

FINAL_MODEL_PATH = Path(
    "model/tiny_luau_gpt_v4.pt"
)

BLOCK_SIZE = 256

DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


# =========================================================
# Load tokenizer
# =========================================================

print("=== TinyLuauGPT v4 Generator ===")
print()

print(f"Device: {DEVICE}")

if DEVICE.type == "cuda":
    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

print()

print("Loading tokenizer...")

tokenizer = LuauTokenizer.load(
    TOKENIZER_PATH
)

print(
    f"Vocabulary size: "
    f"{tokenizer.vocab_size}"
)


# =========================================================
# Load checkpoint
# =========================================================

print()
print("Loading model...")

if BEST_MODEL_PATH.exists():
    checkpoint_path = BEST_MODEL_PATH
    print(
        f"Using BEST checkpoint:"
    )
else:
    checkpoint_path = FINAL_MODEL_PATH
    print(
        f"Best checkpoint not found."
    )
    print(
        f"Using FINAL checkpoint:"
    )

print(
    f"  {checkpoint_path}"
)

checkpoint = torch.load(
    checkpoint_path,
    map_location=DEVICE,
    weights_only=True,
)


# =========================================================
# Create model
# =========================================================

model = TinyLuauGPT(
    vocab_size=checkpoint[
        "vocab_size"
    ],

    block_size=checkpoint[
        "block_size"
    ],

    n_embd=checkpoint[
        "n_embd"
    ],

    n_head=checkpoint[
        "n_head"
    ],

    n_layer=checkpoint[
        "n_layer"
    ],

    dropout=checkpoint[
        "dropout"
    ],
)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

model = model.to(DEVICE)

model.eval()


print()
print("Model loaded!")

print(
    f"Parameters: "
    f"{model.num_parameters():,}"
)

print(
    f"Best validation loss: "
    f"{checkpoint.get('val_loss', 'unknown')}"
)


# =========================================================
# Token IDs
# =========================================================

CHAT_ID = tokenizer.token_to_id[
    "<CHAT>"
]

CODE_ID = tokenizer.token_to_id[
    "<CODE>"
]

EXPLAIN_ID = tokenizer.token_to_id[
    "<EXPLAIN>"
]

FIX_ID = tokenizer.token_to_id[
    "<FIX>"
]

RESPONSE_ID = tokenizer.token_to_id[
    "<RESPONSE>"
]

END_ID = tokenizer.token_to_id[
    "<END>"
]


# =========================================================
# Generation
# =========================================================

def generate_response(
    mode,
    prompt,
    temperature=0.7,
    top_k=40,
    max_new_tokens=180,
):

    mode_id = tokenizer.token_to_id[
        mode
    ]

    # -----------------------------------------------------
    # Format input
    # -----------------------------------------------------

    text = (
        f"{mode}\n"
        f"{prompt}\n"
        f"<RESPONSE>\n"
    )

    input_ids = tokenizer.encode(
        text
    )

    input_tensor = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=DEVICE,
    )

    # -----------------------------------------------------
    # Generate
    # -----------------------------------------------------

    output = model.generate(
        input_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        stop_token_id=END_ID,
    )

    # -----------------------------------------------------
    # Decode
    # -----------------------------------------------------

    generated_ids = output[
        0
    ].tolist()

    generated_text = tokenizer.decode(
        generated_ids
    )

    # -----------------------------------------------------
    # Extract response
    # -----------------------------------------------------

    marker = "<RESPONSE>"

    if marker in generated_text:

        response = generated_text.split(
            marker,
            1
        )[1]

    else:
        response = generated_text

    # Stop at END
    if "<END>" in response:
        response = response.split(
            "<END>",
            1
        )[0]

    # Remove mode markers if generated
    for token in [
        "<CHAT>",
        "<CODE>",
        "<EXPLAIN>",
        "<FIX>",
        "<RESPONSE>",
    ]:
        response = response.replace(
            token,
            ""
        )

    return response.strip()


# =========================================================
# Mode selection
# =========================================================

def select_mode():

    print()
    print(
        "========================================"
    )

    print("1. 💬 CHAT")
    print("2. 💻 CODE")
    print("3. 📖 EXPLAIN")
    print("4. 🔧 FIX")

    print(
        "========================================"
    )

    while True:

        choice = input(
            "Mode [1-4] > "
        ).strip()

        modes = {
            "1": "<CHAT>",
            "2": "<CODE>",
            "3": "<EXPLAIN>",
            "4": "<FIX>",
        }

        if choice in modes:
            return modes[choice]

        print(
            "1~4 중에서 선택해줘!"
        )


# =========================================================
# Main loop
# =========================================================

print()
print(
    "========================================"
)

print("Ready! 🚀")

print()
print(
    "사용법:"
)

print(
    "1 = 일반 대화"
)

print(
    "2 = Luau 코드 생성"
)

print(
    "3 = 코드 설명"
)

print(
    "4 = 코드 수정"
)

print()
print(
    "종료하려면 /exit"
)

print(
    "========================================"
)


while True:

    try:

        mode = select_mode()

        print()

        prompt = input(
            "Prompt > "
        )

        if prompt.strip() == "/exit":
            print(
                "종료합니다."
            )
            break

        if not prompt.strip():
            continue

        print()

        print(
            f"AI ({mode}) >"
        )

        # -------------------------------------------------
        # Different generation settings
        # -------------------------------------------------

        if mode == "<CHAT>":

            temperature = 0.85
            max_tokens = 100

        elif mode == "<CODE>":

            temperature = 0.55
            max_tokens = 220

        elif mode == "<EXPLAIN>":

            temperature = 0.70
            max_tokens = 180

        else:

            temperature = 0.55
            max_tokens = 220

        response = generate_response(
            mode=mode,
            prompt=prompt,
            temperature=temperature,
            top_k=40,
            max_new_tokens=max_tokens,
        )

        print()
        print(response)
        print()

    except KeyboardInterrupt:

        print()
        print(
            "종료합니다."
        )
        break

    except Exception as error:

        print()
        print(
            "오류가 발생했습니다:"
        )

        print(error)
        print()