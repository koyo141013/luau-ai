import json
import re
from pathlib import Path


class LuauTokenizer:
    # v4 special tokens
    SPECIAL_TOKENS = [
        "<PAD>",
        "<UNK>",
        "<BOS>",
        "<EOS>",

        # Conversation / task modes
        "<CHAT>",
        "<CODE>",
        "<EXPLAIN>",
        "<FIX>",

        # Response / boundary
        "<RESPONSE>",
        "<END>",
    ]

    # Tokenization order matters:
    # special tokens must be detected before normal symbols.
    TOKEN_PATTERN = re.compile(
        r"<CHAT>|<CODE>|<EXPLAIN>|<FIX>|<RESPONSE>|<END>"
        r"|<PAD>|<UNK>|<BOS>|<EOS>"
        r"|\s+"
        r"|[A-Za-z_][A-Za-z0-9_]*"
        r"|\d+(?:\.\d+)?"
        r"|==|~=|<=|>=|::|->"
        r'|"(?:\\.|[^"\\])*"'
        r"|'(?:\\.|[^'\\])*'"
        r"|[^\sA-Za-z0-9_]"
    )

    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}

        for token in self.SPECIAL_TOKENS:
            self._add_token(token)

    def _add_token(self, token):
        if token not in self.token_to_id:
            idx = len(self.token_to_id)

            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

    @property
    def vocab_size(self):
        return len(self.token_to_id)

    def tokenize(self, text):
        return self.TOKEN_PATTERN.findall(text)

    def build_vocab(self, texts):
        for text in texts:
            for token in self.tokenize(text):
                self._add_token(token)

    def encode(self, text, add_bos=False, add_eos=False):
        tokens = self.tokenize(text)

        ids = []

        if add_bos:
            ids.append(self.token_to_id["<BOS>"])

        for token in tokens:
            token_id = self.token_to_id.get(
                token,
                self.token_to_id["<UNK>"]
            )

            ids.append(token_id)

        if add_eos:
            ids.append(self.token_to_id["<EOS>"])

        return ids

    def decode(self, ids):
        tokens = []

        for idx in ids:
            token = self.id_to_token.get(
                int(idx),
                "<UNK>"
            )

            tokens.append(token)

        text = ""

        for token in tokens:
            if token == "<WS>":
                text += " "
            else:
                text += token

        return text

    def save(self, path):
        path = Path(path)

        data = {
            "token_to_id": self.token_to_id,
            "id_to_token": {
                str(k): v
                for k, v in self.id_to_token.items()
            }
        }

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2
            )

    @classmethod
    def load(cls, path):
        path = Path(path)

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:
            data = json.load(f)

        tokenizer = cls()

        tokenizer.token_to_id = {
            k: int(v)
            for k, v in data["token_to_id"].items()
        }

        tokenizer.id_to_token = {
            int(k): v
            for k, v in data["id_to_token"].items()
        }

        return tokenizer


if __name__ == "__main__":
    # Simple tokenizer test
    tokenizer = LuauTokenizer()

    test_text = """<CHAT>
안녕?
<RESPONSE>
안녕하세요!
<END>

<CODE>
플레이어 이름을 출력해줘.
<RESPONSE>
local Players = game:GetService("Players")
print("Hello")
<END>"""

    print("=== TinyLuauGPT v4 Tokenizer Test ===")
    print()
    print("Original:")
    print(test_text)
    print()

    tokens = tokenizer.tokenize(test_text)

    print("Tokens:")
    print(tokens)
    print()

    tokenizer.build_vocab([test_text])

    print("Vocabulary size:")
    print(tokenizer.vocab_size)
    print()

    encoded = tokenizer.encode(test_text)

    print("Encoded:")
    print(encoded)
    print()

    decoded = tokenizer.decode(encoded)

    print("Decoded:")
    print(decoded)
    print()

    print("Special tokens:")

    for token in tokenizer.SPECIAL_TOKENS:
        print(
            f"  {token:10} -> "
            f"{tokenizer.token_to_id[token]}"
        )

    print()
    print("Tokenizer test complete!")