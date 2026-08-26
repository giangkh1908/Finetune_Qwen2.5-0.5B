"""Shorten long training answers + add a fresh math dataset (NO eval leak).

1) claude_opus: cut each assistant answer so it fits <= max_seq_length.
   ~4 chars/token for code/English; cut at line/space boundary.

2) math generation: uses a DIFFERENT seed (20260827) than eval (20260826) AND
   uses templates that do NOT overlap with data/eval/math.jsonl templates.
   Post-check removes any prompt that still appears in the eval set.

Outputs:
    data/train/claude_opus_743_short.jsonl
    data/train/math_train_400.jsonl
"""
import json
import os
import random
import re
from math import comb, gcd

SEED = 20260827  # != eval seed
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "data", "train")
EVAL_DIR = os.path.join(ROOT, "data", "eval")

CHARS_PER_TOKEN = 4.0
DEFAULT_MAX_TOKENS = 4096


def split_at_block(text, max_chars):
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    nl = cut.rfind("\n")
    sp = cut.rfind(" ")
    boundary = max(nl, sp)
    if boundary > max_chars * 0.6:
        return cut[:boundary].rstrip()
    return cut.rstrip()


def load_eval_prompts():
    path = os.path.join(EVAL_DIR, "math.jsonl")
    if not os.path.exists(path):
        return set(), set()
    prompts, templates = set(), set()
    for line in open(path, encoding="utf-8"):
        obj = json.loads(line)
        p = obj.get("prompt", "")
        prompts.add(" ".join(p.lower().split()))
        templates.add(" ".join(re.sub(r"\d+", "N", p.lower()).split()))
    return prompts, templates


# ------------------------------------------------------------- MATH (leak-free)
def build_math_train(rng, n=400, eval_prompts=None, eval_templates=None):
    """Generate n math QA using templates NOT present in eval."""
    eval_prompts = eval_prompts or set()
    eval_templates = eval_templates or set()

    def question(answer, templ):
        return templ

    items = []
    seen = set()
    # Each template takes answer + a renderer. We avoid any template whose
    # normalized form matches an eval template.
    # distinct templates
    def add(q, a):
        if q is None:
            return
        norm_q = " ".join(q.lower().split())
        norm_t = " ".join(re.sub(r"\d+", "N", q.lower()).split())
        if norm_q in eval_prompts or norm_t in eval_templates:
            return  # skip leak
        if norm_q in seen:
            return
        seen.add(norm_q)
        items.append({"messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": f"The answer is {a}."},
        ]})

    while len(items) < n:
        kind = rng.randint(0, 7)
        if kind == 0:  # combine two orders of magnitude (unique)
            a = rng.randint(11, 99)
            b = rng.randint(2, 9)
            ans = a * b + (a - b)
            q = f"Calculate {a} times {b} plus the difference of {a} and {b}. Answer with a single number."
            add(q, ans)
        elif kind == 1:  # divide then add
            a = rng.randint(2, 9) * rng.randint(2, 9)
            b = rng.randint(1, a - 1)
            while a % b != 0:
                b = rng.randint(1, a - 1)
            c = rng.randint(1, 11)
            ans = a // b + c
            q = f"Divide {a} by {b} and then add {c}. What is the result? Answer with a single number."
            add(q, ans)
        elif kind == 2:  # age-ish algebra
            x = rng.randint(2, 8)
            mult = rng.randint(2, 5)
            const = rng.randint(1, 9)
            ans = x
            # "x is const less than mult times x"? build: mult*x - x = y -> x = y/(mult-1)
            y = (mult - 1) * x
            q = f"A number multiplied by {mult} minus itself equals {y}. What is the number? Answer with a single number."
            add(q, ans)
        elif kind == 3:  # remaining after percentage (different wording)
            base = rng.choice([50, 80, 120, 150, 200, 250, 300])
            pct = rng.choice([10, 20, 30, 40, 60, 70])
            ans = base * (100 - pct) // 100
            q = f"After applying a {pct}% reduction to {base}, what remains? Answer with a single number."
            add(q, ans)
        elif kind == 4:  # sum of consecutive
            start = rng.randint(1, 6)
            cnt = rng.choice([3, 4, 5])
            ans = cnt * start + cnt * (cnt - 1) // 2
            q = f"What is the sum of the {cnt} consecutive integers starting from {start}? Answer with a single number."
            add(q, ans)
        elif kind == 5:  # triangle perimeter
            s = rng.randint(3, 12)
            a = rng.randint(2, s - 1)
            b = rng.randint(1, s - a)
            c = s - a - b
            if c < 1:
                continue
            ans = a + b + c
            q = f"A triangle has side lengths {a}, {b}, and {c}. What is its perimeter? Answer with a single number."
            add(q, ans)
        elif kind == 6:  # total from ratio
            total = rng.choice([12, 16, 20, 24, 30, 36, 40])
            part = rng.choice([2, 3, 4])
            if total % part != 0:
                continue
            ans = total // part
            q = f"A sum of {total} is split into {part} equal parts. How large is each part? Answer with a single number."
            add(q, ans)
        else:  # lcm by listing
            a = rng.randint(3, 12)
            b = rng.randint(3, 12)
            ans = a * b // gcd(a, b)
            q = f"What is the smallest positive number that is a multiple of both {a} and {b}? Answer with a single number."
            add(q, ans)
    return items


# ----------------------------------------------------------------------------
def shorten_dataset():
    src = os.path.join(TRAIN_DIR, "claude_opus_743_clean.jsonl")
    dst = os.path.join(TRAIN_DIR, "claude_opus_743_short.jsonl")
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    out, n_cut = [], 0
    max_chars = int(DEFAULT_MAX_TOKENS * CHARS_PER_TOKEN)
    for r in rows:
        user = r["messages"][0]["content"]
        asst = r["messages"][1]["content"]
        if len(asst) > max_chars:
            asst = split_at_block(asst, max_chars)
            n_cut += 1
        out.append({"messages": [{"role": "user", "content": user},
                                 {"role": "assistant", "content": asst}]})
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"shortened: {len(out)} rows (cut {n_cut} long answers) -> {dst}")


def build_math():
    eval_prompts, eval_templates = load_eval_prompts()
    rng = random.Random(SEED)
    items = build_math_train(rng, n=2000, eval_prompts=eval_prompts, eval_templates=eval_templates)
    items = items[:400]
    dst = os.path.join(TRAIN_DIR, "math_train_400.jsonl")
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        for r in items:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"math: {len(items)} rows -> {dst}")


def main():
    shorten_dataset()
    build_math()


if __name__ == "__main__":
    main()
