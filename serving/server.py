"""Lightweight OpenAI-compatible server for Qwen2.5-0.5B-Instruct.

Why not vLLM: vLLM 0.28 needs FlashInfer which JIT-compiles CUDA kernels at
runtime, requiring nvcc + ninja (not installed). This server uses plain
transformers - no nvcc, no ninja, works immediately.

Endpoints (OpenAI-compatible):
    GET  /v1/models
    POST /v1/chat/completions
    POST /v1/completions

Usage:
    pip install fastapi uvicorn transformers torch
    MODEL=Qwen/Qwen2.5-0.5B-Instruct PORT=8000 python serving/server.py
"""
import os
import threading

import torch
from fastapi import FastAPI, Request
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
PORT = int(os.environ.get("PORT", "8000"))
DEFAULT_MAX_TOKENS = int(os.environ.get("DEFAULT_MAX_TOKENS", "1536"))

app = FastAPI()

print(f"loading {MODEL} ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

# GPU generation is not reentrant: 40 concurrent requests must serialize.
_lock = threading.Lock()


def _generate(messages, max_tokens):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt").to(model.device)
    with _lock:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=0,
                pad_token_id=tok.eos_token_id,
            )
    gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return gen


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL, "object": "model"}]}


@app.post("/v1/chat/completions")
def chat_completions(req: Request):
    body = req.json()
    messages = body.get("messages", [])
    max_tokens = int(body.get("max_tokens", DEFAULT_MAX_TOKENS))
    gen = _generate(messages, max_tokens)
    return {
        "id": "cmpl-0",
        "object": "chat.completion",
        "model": MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": gen}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.post("/v1/completions")
def completions(req: Request):
    body = req.json()
    prompt = body.get("prompt", "")
    max_tokens = int(body.get("max_tokens", DEFAULT_MAX_TOKENS))
    messages = [{"role": "user", "content": prompt}]
    gen = _generate(messages, max_tokens)
    return {
        "id": "cmpl-0",
        "object": "text_completion",
        "model": MODEL,
        "choices": [{"index": 0, "text": gen, "finish_reason": "stop"}],
    }


if __name__ == "__main__":
    import uvicorn
    print(f"listening on http://0.0.0.0:{PORT}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
