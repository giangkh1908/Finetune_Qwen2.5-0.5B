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


def build_vi():
    """Generate ~250 Vietnamese QA (math + knowledge), no eval leak."""
    rng = random.Random(20260829)  # distinct seed
    items, seen = [], set()

    def add(q, a):
        key = " ".join(q.lower().split())
        if key in seen:
            return
        seen.add(key)
        items.append({"messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]})

    for _ in range(60):
        a, b, c = rng.randint(5, 25), rng.randint(2, 12), rng.randint(2, 9)
        op1, op2 = rng.choice(["+", "-"]), rng.choice(["+", "*"])
        expr = f"{a} {op1} {b} {op2} {c}"
        add(f"Thực hiện phép tính: {expr}. Kết quả là bao nhiêu? Trả lời bằng một số.",
            f"Kết quả là {eval(expr)}.")
    for _ in range(40):
        p, n = rng.choice([5000, 8000, 12000, 15000]), rng.randint(2, 6)
        add(f"Một cuốn sách giá {p} đồng. Nếu mua {n} cuốn cùng loại thì tổng tiền là bao nhiêu? Trả lời bằng một số.",
            f"Tổng tiền là {p * n} đồng.")
    for _ in range(30):
        b, q = rng.randint(2, 12), rng.randint(3, 9)
        add(f"Chia {b * q} thành {b} phần bằng nhau. Mỗi phần là bao nhiêu? Trả lời bằng một số.",
            f"Mỗi phần là {q}.")
    for _ in range(30):
        a, b = rng.randint(2, 19), rng.randint(2, 19)
        op = rng.choice(["+", "-", "*"])
        if op == "-" and a < b:
            a, b = b, a
        ans = {"+": a + b, "-": a - b, "*": a * b}[op]
        add(f"Tính kết quả của phép tính: {a} {op} {b}. Trả lời bằng một số.", f"Kết quả là {ans}.")
    for _ in range(25):
        start, step = rng.randint(1, 9), rng.randint(2, 6)
        terms = [start + i * step for i in range(4)]
        add(f"Cho dãy số: {', '.join(map(str, terms))}, ... Số hạng tiếp theo là bao nhiêu? Trả lời bằng một số.",
            f"Số hạng tiếp theo là {terms[-1] + step}.")
    for _ in range(20):
        p, d = rng.choice([120, 200, 300, 400, 500]), rng.choice([10, 20, 25, 50])
        add(f"Một chiếc áo giá {p} đồng được giảm giá {d}%. Hỏi giá bán sau khi giảm là bao nhiêu? Trả lời bằng một số.",
            f"Giá bán sau khi giảm là {p - p * d // 100} đồng.")
    for _ in range(30):
        a, b = rng.randint(3, 12), rng.randint(3, 12)
        if rng.random() < 0.5:
            add(f"Một hình chữ nhật có chiều dài {a} cm và chiều rộng {b} cm. Chu vi là bao nhiêu? Trả lời bằng một số.",
                f"Kết quả là {2 * (a + b)}.")
        else:
            add(f"Một hình chữ nhật có chiều dài {a} cm và chiều rộng {b} cm. Diện tích là bao nhiêu? Trả lời bằng một số.",
                f"Kết quả là {a * b}.")
    for _ in range(15):
        now, add_year = rng.randint(5, 12), rng.randint(1, 5)
        add(f"Bé hiện nay {now} tuổi. Hỏi sau {add_year} năm nữa bé bao nhiêu tuổi? Trả lời bằng một số.",
            f"Sau {add_year} năm nữa bé {now + add_year} tuổi.")

    know = [
        ("Đồng bằng sông Cửu Long còn gọi là gì?", "Đồng bằng Nam Bộ / miền Tây"),
        ("Thành phố nào được gọi là 'thành phố ngàn hoa'?", "Đà Lạt"),
        ("Nước ta giáp với những nước nào trên đất liền?", "Lào, Campuchia, Trung Quốc"),
        ("Tết Nguyên Đán là ngày lễ lớn nhất ở đâu?", "Việt Nam"),
        ("Phở nổi tiếng của miền nào?", "miền Bắc"),
        ("Sài Gòn tên chính thức hiện nay là gì?", "Thành phố Hồ Chí Minh"),
        ("Bác Hồ sinh năm nào?", "1890"),
        ("Núi cao nhất Việt Nam?", "Fansipan"),
        ("Vịnh Hạ Long thuộc tỉnh nào?", "tỉnh Quảng Ninh"),
        ("Chùa Một Cột ở thành phố nào?", "Hà Nội"),
        ("Một tuần có mấy ngày?", "7 ngày"),
        ("Một năm có mấy tháng?", "12 tháng"),
        ("Thủ đô của nước Nhật?", "Tokyo"),
        ("Động vật lớn nhất trên cạn?", "voi"),
        ("Đại dương lớn nhất thế giới?", "Thái Bình Dương"),
        ("Nước nào có diện tích lớn nhất thế giới?", "Nga"),
        ("H₂O là công thức hóa học của nước. Đúng hay sai?", "đúng"),
        ("Ánh sáng truyền nhanh hơn âm thanh. Đúng hay sai?", "đúng"),
        ("Mặt Trời mọc ở phía Tây. Đúng hay sai?", "sai, mặt trời mọc ở phía Đông"),
        ("Quốc kỳ Việt Nam có màu gì?", "màu đỏ với ngôi sao vàng"),
        ("Đơn vị tiền tệ của Việt Nam?", "đồng Việt Nam"),
        ("Việt Nam có bao nhiêu dân tộc anh em (chính thức)?", "54 dân tộc"),
    ]
    rng.shuffle(know)
    for q, a in know:
        add(q, a)

    dst = os.path.join(TRAIN_DIR, "vi_250.jsonl")
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        for r in items:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"vi: {len(items)} rows -> {dst}")


def build_merged():
    """Merge coding(short) + math + vi into the single train file."""
    main = [json.loads(l) for l in open(os.path.join(TRAIN_DIR, "claude_opus_math_1143.jsonl"), encoding="utf-8")]
    vi = [json.loads(l) for l in open(os.path.join(TRAIN_DIR, "vi_250.jsonl"), encoding="utf-8")]
    m_q = set(" ".join(r["messages"][0]["content"].lower().split()) for r in main)
    vi_new = [r for r in vi if " ".join(r["messages"][0]["content"].lower().split()) not in m_q]
    combined = main + vi_new
    dst = os.path.join(TRAIN_DIR, "claude_opus_train_all.jsonl")
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        for r in combined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"merged: {len(combined)} rows -> {dst}")


def main():
    shorten_dataset()
    build_math()
    build_vi()
    build_merged()


if __name__ == "__main__":
    main()
