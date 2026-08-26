"""Pull curated Python code-instruct samples from Magicoder-OSS-Instruct-75K.

Licence MIT. We select Python, short (< 4000 tok), and EXCLUDE samples whose
target function collides with eval/coding.jsonl entries (fizzbuzz, two_sum,
is_palindrome, ...) to avoid train/eval contamination.

Input : data/raw/magicoder_oss.jsonl  (75k rows, downloaded)
Output: data/train/magicoder_python_5000.jsonl
"""
import json
import os
import random
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "magicoder_oss.jsonl")
EVAL = os.path.join(ROOT, "data", "eval", "coding.jsonl")
OUT = os.path.join(ROOT, "data", "train", "magicoder_python_5000.jsonl")

SEED = 20260903
CHARS = 3.5
MAX = 4000
N_SAMPLE = 5000
MIN_SOL = 40
MAX_PROB = 3000


def load_eval_funcs():
    funcs, keywords = set(), set()
    for l in open(EVAL, encoding="utf-8"):
        entry = json.loads(l)["entry"]
        funcs.add(entry)
        for w in re.split(r"[_]+", entry):
            if len(w) >= 3:
                keywords.add(w.lower())
    return funcs, keywords


def main():
    funcs, keywords = load_eval_funcs()
    sel = []
    with open(RAW, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("lang") != "python":
                continue
            p, s = r.get("problem", ""), r.get("solution", "")
            if len(s) < MIN_SOL or len(p) > MAX_PROB:
                continue
            if (len(p) + len(s)) / CHARS > MAX:
                continue
            names = re.findall(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", s)
            entry = names[0].lower() if names else ""
            if not entry:
                continue
            if entry in funcs:
                continue
            if any(kw in entry for kw in keywords if len(kw) >= 3):
                continue
            sel.append(r)
    print(f"python eligible (no eval collision): {len(sel)}")

    rng = random.Random(SEED)
    rng.shuffle(sel)
    chosen = sel[:N_SAMPLE]

    out = []
    seen = set()
    for r in chosen:
        p = r["problem"].strip()
        s = r["solution"].strip()
        # solution may already include ```python fences; keep as-is inside answer
        key = " ".join(p.lower().split())
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "messages": [
                {"role": "user", "content": p},
                {"role": "assistant", "content": s},
            ]
        })
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} -> {OUT}")


if __name__ == "__main__":
    main()
