import json
from pathlib import Path

import torch


DATASET_PATH = Path("dataset/luau_v5.txt")
TOKENS_PATH = Path("dataset/tokens_v51.pt")
TOKENIZER_PATH = Path("model/tokenizer_v51.json")


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
    """
    v5.1 tokenizer

    핵심:
    - 공백 보존
    - 탭 보존
    - 줄바꿈 보존
    - 특수 토큰 보존
    - Luau 기호 보존
    """

    tokens = []

    i = 0

    while i < len(text):

        # -------------------------
        # Special tokens
        # -------------------------

        found = False

        for special in SPECIAL_TOKENS:

            if text.startswith(special, i):

                tokens.append(special)

                i += len(special)

                found = True

                break

        if found:
            continue


        ch = text[i]


        # -------------------------
        # 공백
        # -------------------------

        if ch == " ":
            tokens.append("<SPACE>")
            i += 1
            continue


        # -------------------------
        # 탭
        # -------------------------

        if ch == "\t":
            tokens.append("<TAB>")
            i += 1
            continue


        # -------------------------
        # 줄바꿈
        # -------------------------

        if ch == "\n":
            tokens.append("<NEWLINE>")
            i += 1
            continue


        # -------------------------
        # \r
        # -------------------------

        if ch == "\r":
            i += 1
            continue


        # -------------------------
        # 영문 / 숫자 / 한글
        # -------------------------

        if (
            ch.isalnum()
            or ord(ch) >= 128
            or ch == "_"
        ):

            j = i + 1

            while j < len(text):

                c = text[j]

                if (
                    c.isalnum()
                    or ord(c) >= 128
                    or c == "_"
                ):
                    j += 1
                else:
                    break


            tokens.append(
                text[i:j]
            )

            i = j

            continue


        # -------------------------
        # 기타 기호
        # -------------------------

        tokens.append(ch)

        i += 1


    return tokens


def main():

    print("=" * 60)
    print("TinyLuauGPT v5.1 Dataset Preparation")
    print("=" * 60)

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )


    text = DATASET_PATH.read_text(
        encoding="utf-8"
    )


    print(
        f"Dataset:    {DATASET_PATH}"
    )

    print(
        f"Characters: {len(text):,}"
    )

    print()
    print("Tokenizing dataset...")


    raw_tokens = tokenize(text)


    print(
        f"Total tokens: {len(raw_tokens):,}"
    )


    # -------------------------
    # Vocabulary
    # -------------------------

    print()
    print("Building vocabulary...")


    vocab = {}

    for token in SPECIAL_TOKENS:

        if token not in vocab:

            vocab[token] = len(vocab)


    # 공백 관련 토큰
    for token in [
        "<SPACE>",
        "<TAB>",
        "<NEWLINE>",
    ]:

        if token not in vocab:

            vocab[token] = len(vocab)


    for token in raw_tokens:

        if token not in vocab:

            vocab[token] = len(vocab)


    print(
        f"Vocabulary size: {len(vocab)}"
    )


    # -------------------------
    # Encode
    # -------------------------

    print()
    print("Encoding dataset...")


    ids = [
        vocab[token]
        for token in raw_tokens
    ]


    tokens_tensor = torch.tensor(
        ids,
        dtype=torch.long
    )


    # -------------------------
    # Save tokens
    # -------------------------

    TOKENS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    MODEL_DIR = TOKENIZER_PATH.parent

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    torch.save(
        tokens_tensor,
        TOKENS_PATH
    )


    # -------------------------
    # Save tokenizer
    # -------------------------

    tokenizer_data = {

        "vocab": vocab,

        "special_tokens": SPECIAL_TOKENS,

        "format_version": "v5.1",

        "whitespace_tokens": {
            "<SPACE>": " ",
            "<TAB>": "\t",
            "<NEWLINE>": "\n"
        }

    }


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


    print()
    print("Saved")

    print(
        f"Tokens:\n{TOKENS_PATH.resolve()}"
    )

    print(
        f"Tokenizer:\n{TOKENIZER_PATH.resolve()}"
    )


    print()
    print("Special tokens:")

    for token in SPECIAL_TOKENS:

        print(
            f"  {token:<12} -> {vocab[token]}"
        )


    print()
    print("Whitespace tokens:")

    print(
        f"  <SPACE>      -> {vocab['<SPACE>']}"
    )

    print(
        f"  <TAB>        -> {vocab['<TAB>']}"
    )

    print(
        f"  <NEWLINE>    -> {vocab['<NEWLINE>']}"
    )


    # -------------------------
    # Test
    # -------------------------

    print()
    print("Tokenizer test:")


    test_text = (
        "local Players = game:GetService(\"Players\")\n"
        "\n"
        "Players.PlayerAdded:Connect(function(player)\n"
        "    print(player.Name)\n"
        "end)"
    )


    test_tokens = tokenize(test_text)


    reverse_vocab = {
        idx: token
        for token, idx in vocab.items()
    }


    decoded = ""

    for token in test_tokens:

        if token == "<SPACE>":
            decoded += " "

        elif token == "<TAB>":
            decoded += "\t"

        elif token == "<NEWLINE>":
            decoded += "\n"

        else:
            decoded += token


    print()
    print("Original:")
    print(test_text)

    print()
    print("Decoded:")
    print(decoded)


    if decoded == test_text:

        print()
        print("Tokenizer test: PASS")

    else:

        print()
        print("Tokenizer test: FAIL")


    print()
    print("=" * 60)
    print("TinyLuauGPT v5.1 preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()