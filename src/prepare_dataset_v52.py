from pathlib import Path
from collections import Counter
import json
import re


# ============================================================
# TinyLuauGPT v5.2 Dataset Preparation
# User-facing version: Beta 0.5
# ============================================================

DATASET_PATH = Path("dataset/luau_v52.txt")

TOKENS_PATH = Path("dataset/tokens_v52.pt")

TOKENIZER_PATH = Path("model/tokenizer_v52.json")


# ============================================================
# Special tokens
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


# Whitespace tokens
SPACE_TOKEN = "<SPACE>"
TAB_TOKEN = "<TAB>"
NEWLINE_TOKEN = "<NEWLINE>"


# ============================================================
# Tokenization
# ============================================================

def tokenize_text(text):

    tokens = []

    i = 0

    while i < len(text):

        # ----------------------------------------------------
        # Newline
        # ----------------------------------------------------

        if text[i] == "\n":

            tokens.append(
                NEWLINE_TOKEN
            )

            i += 1

            continue


        # ----------------------------------------------------
        # Space
        # ----------------------------------------------------

        if text[i] == " ":

            tokens.append(
                SPACE_TOKEN
            )

            i += 1

            continue


        # ----------------------------------------------------
        # Tab
        # ----------------------------------------------------

        if text[i] == "\t":

            tokens.append(
                TAB_TOKEN
            )

            i += 1

            continue


        # ----------------------------------------------------
        # Special tokens
        # ----------------------------------------------------

        matched = False

        for special in SPECIAL_TOKENS:

            if text.startswith(
                special,
                i
            ):

                tokens.append(
                    special
                )

                i += len(special)

                matched = True

                break


        if matched:

            continue


        # ----------------------------------------------------
        # Identifier / Korean / Unicode
        # ----------------------------------------------------

        char = text[i]

        if (
            char.isalpha()
            or char == "_"
            or ord(char) >= 128
        ):

            start = i

            while i < len(text):

                c = text[i]

                if (
                    c.isalnum()
                    or c == "_"
                    or ord(c) >= 128
                ):

                    i += 1

                else:

                    break


            token = text[
                start:i
            ]


            tokens.append(
                token
            )

            continue


        # ----------------------------------------------------
        # Number
        # ----------------------------------------------------

        if char.isdigit():

            start = i

            while i < len(text):

                c = text[i]

                if (
                    c.isdigit()
                    or c == "."
                ):

                    i += 1

                else:

                    break


            tokens.append(
                text[start:i]
            )

            continue


        # ----------------------------------------------------
        # Quoted string
        # ----------------------------------------------------

        if char in (
            '"',
            "'",
            "`"
        ):

            quote = char

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


            tokens.append(
                text[start:i]
            )

            continue


        # ----------------------------------------------------
        # Operators
        # ----------------------------------------------------

        operators = [
            "==",
            "~=",
            "<=",
            ">=",
            "..",
            "...",
            "+=",
            "-=",
            "*=",
            "/=",
            "::",
            "->",
        ]


        found_operator = None


        for operator in operators:

            if text.startswith(
                operator,
                i
            ):

                found_operator = operator

                break


        if found_operator:

            tokens.append(
                found_operator
            )

            i += len(
                found_operator
            )

            continue


        # ----------------------------------------------------
        # Single character
        # ----------------------------------------------------

        tokens.append(
            char
        )

        i += 1


    return tokens


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 64)
    print("TinyLuauGPT v5.2 Dataset Preparation")
    print("User-facing version: Beta 0.5")
    print("=" * 64)


    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n"
            f"{DATASET_PATH}"
        )


    # --------------------------------------------------------
    # Read dataset
    # --------------------------------------------------------

    text = DATASET_PATH.read_text(
        encoding="utf-8"
    )


    print()
    print(
        "Dataset:",
        DATASET_PATH
    )

    print(
        "Characters:",
        f"{len(text):,}"
    )


    # --------------------------------------------------------
    # Tokenize
    # --------------------------------------------------------

    print()
    print(
        "Tokenizing dataset..."
    )


    tokens = tokenize_text(
        text
    )


    print(
        "Total tokens:",
        f"{len(tokens):,}"
    )


    # --------------------------------------------------------
    # Vocabulary
    # --------------------------------------------------------

    print()
    print(
        "Building vocabulary..."
    )


    counter = Counter(
        tokens
    )


    vocab = {}

    # Special tokens first
    for token in SPECIAL_TOKENS:

        if token not in vocab:

            vocab[token] = len(vocab)


    # Whitespace tokens
    for token in [
        SPACE_TOKEN,
        TAB_TOKEN,
        NEWLINE_TOKEN,
    ]:

        if token not in vocab:

            vocab[token] = len(vocab)


    # Remaining tokens by frequency
    remaining = [
        token
        for token in counter
        if token not in vocab
    ]


    remaining.sort(
        key=lambda x: (
            -counter[x],
            x
        )
    )


    for token in remaining:

        vocab[token] = len(vocab)


    print(
        "Vocabulary size:",
        len(vocab)
    )


    # --------------------------------------------------------
    # Encode
    # --------------------------------------------------------

    print()
    print(
        "Encoding dataset..."
    )


    unk_id = vocab[
        "<UNK>"
    ]


    encoded = [
        vocab.get(
            token,
            unk_id
        )
        for token in tokens
    ]


    # --------------------------------------------------------
    # Save tokens
    # --------------------------------------------------------

    import torch


    tensor = torch.tensor(
        encoded,
        dtype=torch.long
    )


    TOKENS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    torch.save(
        tensor,
        TOKENS_PATH
    )


    # --------------------------------------------------------
    # Save tokenizer
    # --------------------------------------------------------

    tokenizer_data = {

        "version":
            "5.2-beta.0.5",

        "vocab":
            vocab,

        "special_tokens":
            SPECIAL_TOKENS,

        "whitespace_tokens": {

            "space":
                SPACE_TOKEN,

            "tab":
                TAB_TOKEN,

            "newline":
                NEWLINE_TOKEN,
        },

    }


    TOKENIZER_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        TOKENIZER_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            tokenizer_data,
            f,
            ensure_ascii=False,
            indent=2
        )


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print()
    print(
        "Saved"
    )

    print(
        "Tokens:"
    )

    print(
        TOKENS_PATH.resolve()
    )

    print()

    print(
        "Tokenizer:"
    )

    print(
        TOKENIZER_PATH.resolve()
    )


    # --------------------------------------------------------
    # Special tokens
    # --------------------------------------------------------

    print()
    print(
        "Special tokens:"
    )

    for token in SPECIAL_TOKENS:

        print(
            f"  {token:<12} -> "
            f"{vocab[token]}"
        )


    print()
    print(
        "Whitespace tokens:"
    )

    print(
        f"  {SPACE_TOKEN:<12} -> "
        f"{vocab[SPACE_TOKEN]}"
    )

    print(
        f"  {TAB_TOKEN:<12} -> "
        f"{vocab[TAB_TOKEN]}"
    )

    print(
        f"  {NEWLINE_TOKEN:<12} -> "
        f"{vocab[NEWLINE_TOKEN]}"
    )


    # --------------------------------------------------------
    # Mode statistics
    # --------------------------------------------------------

    mode_counts = {
        "CHAT": 0,
        "CODE": 0,
        "EXPLAIN": 0,
        "FIX": 0,
    }


    for mode in mode_counts:

        mode_counts[mode] = text.count(
            f"<{mode}>\n"
        )


    print()
    print(
        "Dataset Mode Statistics:"
    )

    for mode, count in mode_counts.items():

        print(
            f"  {mode:<8}",
            f"{count:,}"
        )


    # --------------------------------------------------------
    # Tokenizer test
    # --------------------------------------------------------

    test_text = '''local Players = game:GetService("Players")

Players.PlayerAdded:Connect(function(player)
    print(player.Name)
end)'''


    print()
    print(
        "Tokenizer test:"
    )


    test_tokens = tokenize_text(
        test_text
    )


    decoded = []


    reverse_vocab = {
        value: key
        for key, value in vocab.items()
    }


    for token in test_tokens:

        token_id = vocab.get(
            token,
            unk_id
        )

        decoded_token = reverse_vocab.get(
            token_id,
            "<UNK>"
        )


        if decoded_token == SPACE_TOKEN:

            decoded.append(" ")

        elif decoded_token == TAB_TOKEN:

            decoded.append("\t")

        elif decoded_token == NEWLINE_TOKEN:

            decoded.append("\n")

        else:

            decoded.append(
                decoded_token
            )


    decoded_text = "".join(
        decoded
    )


    print()
    print("Original:")
    print(test_text)

    print()
    print("Decoded:")
    print(decoded_text)


    if decoded_text == test_text:

        print()
        print(
            "Tokenizer test: PASS"
        )

    else:

        print()
        print(
            "Tokenizer test: FAILED"
        )


    print()
    print("=" * 64)
    print(
        "TinyLuauGPT v5.2 preparation complete!"
    )
    print(
        "User-facing version: Beta 0.5"
    )
    print("=" * 64)


if __name__ == "__main__":
    main()