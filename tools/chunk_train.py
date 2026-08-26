"""Chunk long coding answers into pieces that fit < max_seq_length, no data loss.

The original claude_opus_743 produces answers up to ~25k tokens. Instead of
truncating (losing the tail), split each long answer into multiple shorter
samples, each reusing the same prompt + a contiguous piece of the answer.
Every resulting sample is < max_seq_length tokens (estimated conservatively).

Usage:
    python tools/chunk_train.py --input data/train/coder_train_all.jsonl \
        --output data/train/coder_train_all_chunked.jsonl --max-tokens 4000
"""
import argparse
import json
import os
import re

# conservative chars-per-token for code/English (safe under-estimate of tokens)
CHARS_PER_TOKEN = 3.5


def estimate_tokens(text):
    return len(text) / CHARS_PER_TOKEN


def split_answer(answer, prompt_len_tokens, max_tokens):
    """Split answer into pieces so prompt+piece fits within max_tokens."""
    budget_chars = int((max_tokens - prompt_len_tokens) * CHARS_PER_TOKEN)
    if budget_chars <= 100:
        return []
    # split at newline boundaries (code blocks) when possible
    # find a safe split point around budget_chars
    chunks = []
    remaining = answer
    while len(remaining) > budget_chars:
        window = remaining[:budget_chars]
        # split at last newline or space within window
        nl = window.rfind("\n")
        sp = window.rfind(" ")
        boundary = max(nl, sp)
        if boundary < budget_chars * 0.5:
            boundary = budget_chars  # hard cut (no good boundary)
        piece = remaining[:boundary].rstrip()
        if piece:
            chunks.append(piece)
        remaining = remaining[boundary:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-tokens", type=int, default=4000)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8")]
    out = []
    n_chunked = 0
    for r in rows:
        user = r["messages"][0]["content"]
        asst = r["messages"][1]["content"]
        prompt_tok = estimate_tokens(user)
        if estimate_tokens(user) + estimate_tokens(asst) <= args.max_tokens:
            out.append(r)
            continue
        pieces = split_answer(asst, prompt_tok, args.max_tokens)
        if len(pieces) <= 1:
            out.append(r)
            continue
        n_chunked += 1
        for piece in pieces:
            out.append({"messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": piece},
            ]})
    with open(args.output, "w", encoding="utf-8", newline="\n") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # verify no sample exceeds max_tokens (conservative)
    over = sum(1 for r in out if (estimate_tokens(r["messages"][0]["content"]) +
                                  estimate_tokens(r["messages"][1]["content"])) > args.max_tokens)
    print(f"input: {len(rows)} | output: {len(out)} | chunked: {n_chunked} | "
          f"over_max_tokens: {over}")
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
