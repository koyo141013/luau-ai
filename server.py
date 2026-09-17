from pathlib import Path
import json
import time

import torch
from flask import Flask, jsonify, request, send_from_directory


# ============================================================
# 경로 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
WEB_DIR = BASE_DIR / "web"


# ============================================================
# Flask
# ============================================================

app = Flask(
    __name__,
    static_folder=str(WEB_DIR),
)


# ============================================================
# CORS
# ============================================================

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = (
        "https://koyo141013.github.io"
    )

    response.headers["Access-Control-Allow-Methods"] = (
        "GET, POST, OPTIONS"
    )

    response.headers["Access-Control-Allow-Headers"] = (
        "Content-Type"
    )

    return response


@app.route("/api/<path:path>", methods=["OPTIONS"])
def handle_options(path):
    return "", 204


# ============================================================
# 디바이스
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 모델 / 토크나이저
# ============================================================

from src.model_v52 import TinyLuauGPTv52


TOKENIZER_PATH = MODEL_DIR / "tokenizer_v52.json"
CHECKPOINT_PATH = MODEL_DIR / "tiny_luau_gpt_v52_best.pt"


# ------------------------------------------------------------
# Tokenizer 로드
# ------------------------------------------------------------

with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
    tokenizer_data = json.load(f)


stoi = tokenizer_data["stoi"]
itos = tokenizer_data["itos"]


# JSON에서 숫자가 문자열로 저장될 가능성 대응
stoi = {
    str(k): int(v)
    for k, v in stoi.items()
}

itos = {
    str(k): str(v)
    for k, v in itos.items()
}


# ============================================================
# Tokenizer 함수
# ============================================================

def encode_text(text):
    """
    문자열 -> 토큰 ID
    """

    tokens = []

    for ch in text:
        key = ch

        if key in stoi:
            tokens.append(stoi[key])
        else:
            # 알 수 없는 문자는 공백으로 대체
            if " " in stoi:
                tokens.append(stoi[" "])

    return tokens


def decode_tokens(tokens):
    """
    토큰 ID -> 문자열
    """

    result = []

    for token_id in tokens:
        key = str(int(token_id))

        if key in itos:
            result.append(itos[key])

    return "".join(result)


# ============================================================
# Checkpoint 로드
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
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


# ============================================================
# 정보
# ============================================================

PARAMETERS = sum(
    p.numel()
    for p in model.parameters()
)

BEST_VAL_LOSS = checkpoint.get(
    "best_val_loss",
    None,
)


print("=" * 60)
print("Luau AI v5.2 Web Server")
print("User-facing version: Beta 0.5")
print("=" * 60)
print(f"Device: {device}")
print(f"Vocabulary size: {len(stoi)}")
print(f"Model: {CHECKPOINT_PATH.name}")
print(f"Parameters: {PARAMETERS:,}")

if BEST_VAL_LOSS is not None:
    print(
        f"Best Val Loss: {float(BEST_VAL_LOSS):.4f}"
    )

print("=" * 60)


# ============================================================
# 생성 설정
# ============================================================

MODE_SETTINGS = {
    "CHAT": {
        "temperature": 0.85,

        # Render Free CPU 테스트용
        "max_new_tokens": 10,
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


# ============================================================
# 모델 생성 함수
# ============================================================

@torch.no_grad()
def generate_response(prompt, mode="CHAT"):
    """
    사용자 프롬프트를 받아 모델 응답 생성
    """

    mode = str(mode).upper()

    if mode not in MODE_SETTINGS:
        mode = "CHAT"

    settings = MODE_SETTINGS[mode]

    # 모드 토큰
    formatted_prompt = (
        f"<{mode}>\n"
        f"{prompt}\n"
        f"<RESPONSE>\n"
    )

    encoded = encode_text(formatted_prompt)

    if not encoded:
        encoded = [
            stoi.get(" ", 0)
        ]

    # block size보다 길면 마지막 부분만 사용
    encoded = encoded[
        -checkpoint["block_size"] :
    ]

    x = torch.tensor(
        [encoded],
        dtype=torch.long,
        device=device,
    )

    output = model.generate(
        x,
        max_new_tokens=settings["max_new_tokens"],
        temperature=settings["temperature"],
        top_k=40,
        stop_token_id=stoi.get("<END>"),
    )

    generated_tokens = output[0].tolist()

    # 입력 부분 제거
    generated_tokens = generated_tokens[
        len(encoded):
    ]

    text = decode_tokens(
        generated_tokens
    )

    # 혹시 END 토큰 문자열이 출력되는 경우 제거
    if "<END>" in text:
        text = text.split(
            "<END>",
            1
        )[0]

    return text.strip()


# ============================================================
# 홈페이지
# ============================================================

@app.route("/")
def index():
    return send_from_directory(
        WEB_DIR,
        "index.html",
    )


# ============================================================
# 정적 파일
# ============================================================

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(
        WEB_DIR,
        filename,
    )


# ============================================================
# 상태 확인 API
# ============================================================

@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "version": "Beta 0.5",
        "model": "Luau AI v5.2",
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "parameters": PARAMETERS,
        "best_val_loss": BEST_VAL_LOSS,
    })


# ============================================================
# 🔥 Render 성능 측정용 Benchmark
# ============================================================

@app.route("/api/benchmark", methods=["POST"])
def benchmark():

    # 테스트 입력 100토큰
    x = torch.randint(
        0,
        len(stoi),
        (1, 100),
        dtype=torch.long,
        device=device,
    )

    # CPU/GPU 동기화
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():
        model.generate(
            x,
            max_new_tokens=1,
            temperature=0.7,
            top_k=40,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = (
        time.perf_counter() - start
    )

    return jsonify({
        "seconds": round(elapsed, 3),
        "device": str(device),
        "parameters": PARAMETERS,
        "input_tokens": 100,
        "generated_tokens": 1,
    })


# ============================================================
# 생성 API
# ============================================================

@app.route(
    "/api/generate",
    methods=["POST"],
)
def generate():

    try:

        data = request.get_json(
            silent=True
        ) or {}

        prompt = str(
            data.get("prompt", "")
        ).strip()

        mode = str(
            data.get("mode", "CHAT")
        ).upper()

        if not prompt:
            return jsonify({
                "error": "prompt is required"
            }), 400

        if mode not in MODE_SETTINGS:
            mode = "CHAT"

        start = time.perf_counter()

        response = generate_response(
            prompt,
            mode,
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        return jsonify({
            "response": response,
            "mode": mode,
            "seconds": round(
                elapsed,
                3,
            ),
            "model": "Luau AI v5.2",
            "version": "Beta 0.5",
        })

    except Exception as e:

        print(
            "Generation error:",
            repr(e),
        )

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# 서버 실행
# ============================================================

if __name__ == "__main__":

    print(
        "Starting Luau AI server..."
    )

    app.run(
        host="0.0.0.0",
        port=10000,
        debug=False,
    )