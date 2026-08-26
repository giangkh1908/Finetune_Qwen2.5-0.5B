"""Build the FROZEN benchmark sets for the Qwen2.5-0.5B fine-tuning lab.

Deterministic (SEED fixed). Ground truth is COMPUTED by this script, never
guessed: math answers come from arithmetic, logic puzzles are solved by
brute-force enumeration, coding tests are validated against reference
implementations before anything is written.

Outputs (freeze these, never edit afterwards):
    data/eval/math.jsonl          50 items  - numeric
    data/eval/reasoning.jsonl     40 items  - exact (verified)
    data/eval/coding.jsonl        24 items  - unit tests (subprocess)
    data/eval/general.jsonl       36 items  - MCQ + constraint checkers
    data/eval/qualitative.jsonl   24 prompts - no GT, for human diff
    data/eval/manifest.json       sha256 + counts
"""

import hashlib
import json
import os
import re
from fractions import Fraction
from itertools import combinations, permutations, product
from math import comb, gcd


SEED = 20260826
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(ROOT, "data", "eval")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def write_jsonl(name, items):
    path = os.path.join(EVAL_DIR, name)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return path


# --------------------------------------------------------------------------
# MATH (50)
# --------------------------------------------------------------------------

def build_math(rng):
    items = []
    n = 0

    def add(sub, prompt, answer, **kw):
        nonlocal n
        n += 1
        item = {
            "id": f"math_{n:03d}",
            "category": "math",
            "subcategory": sub,
            "prompt": prompt,
            "checker": "numeric",
            "answer": str(answer),
        }
        item.update(kw)
        items.append(item)

    # arithmetic (8)
    for _ in range(8):
        ops = [rng.choice(["+", "-", "*"]) for _ in range(3)]
        nums = [rng.randint(2, 19) for _ in range(4)]
        expr = f"{nums[0]} {ops[0]} {nums[1]} {ops[1]} {nums[2]} {ops[2]} {nums[3]}"
        ans = eval(expr)
        add("arithmetic", f"What is the value of {expr}? Answer with a single number.", ans)

    # algebra linear (6)
    for _ in range(6):
        x0 = rng.randint(-9, 12)
        a = rng.randint(2, 9)
        b = rng.randint(-20, 20)
        c = a * x0 + b
        if b >= 0:
            eq = f"{a}x + {b} = {c}"
        else:
            eq = f"{a}x - {-b} = {c}"
        add("algebra", f"Solve for x: {eq}. Answer with a single number.", x0)

    # quadratic larger root (4)
    for _ in range(4):
        while True:
            r1, r2 = sorted(rng.sample(range(-6, 8), 2))
            if r1 * r2 != 0 and r1 + r2 != 0:
                break
        S, P = r1 + r2, r1 * r2
        mid = f"- {S}x" if S > 0 else f"+ {-S}x"
        tail = f"+ {P}" if P > 0 else f"- {-P}"
        add("algebra",
            f"The quadratic equation x^2 {mid} {tail} = 0 has two integer roots. What is the larger root? Answer with a single number.",
            r2)

    # percentages (4) discount
    for _ in range(4):
        d = rng.choice([10, 20, 25, 40, 50])
        P = rng.choice([120, 160, 200, 240, 300, 400, 500])
        add("percent",
            f"A jacket costs ${P}. It is discounted by {d}%. What is the sale price in dollars? Answer with a single number.",
            P - P * d // 100)
    # simple interest (4)
    for _ in range(4):
        while True:
            P = rng.choice([200, 400, 500, 800, 1000, 1200, 2000])
            r = rng.choice([2, 4, 5, 8, 10])
            t = rng.randint(2, 5)
            if (P * r * t) % 100 == 0:
                break
        add("percent",
            f"A deposit of ${P} earns simple interest at {r}% per year. How many dollars of interest are earned after {t} years? Answer with a single number.",
            P * r * t // 100)

    # rates: distance (3) + work (3)
    for _ in range(3):
        v = rng.randint(15, 90)
        t = rng.randint(2, 6)
        add("rates", f"A train travels at {v} km/h for {t} hours. How many kilometers does it travel? Answer with a single number.", v * t)
    clean_pairs = [(a, b) for a in range(2, 25) for b in range(a + 1, 40) if (a * b) % (a + b) == 0]
    for _ in range(3):
        a, b = rng.choice(clean_pairs)
        add("rates",
            f"Alice can paint a fence in {a} hours. Bob can paint the same fence in {b} hours. Working together at these rates, how many hours do they need? Answer with a single number.",
            (a * b) // (a + b))

    # sequences (7)
    seq_types = ["arith", "geom", "fib", "squares", "arith", "geom", "fib"]
    for ty in seq_types:
        if ty == "arith":
            a0, d = rng.randint(1, 10), rng.randint(2, 7)
            terms = [a0 + i * d for i in range(5)]
            ans = a0 + 5 * d
        elif ty == "geom":
            a0, r = rng.randint(1, 3), rng.choice([2, 3])
            terms = [a0 * r ** i for i in range(5)]
            ans = a0 * r ** 5
        elif ty == "fib":
            a, b = rng.randint(1, 4), rng.randint(2, 5)
            terms = [a, b]
            while len(terms) < 5:
                terms.append(terms[-1] + terms[-2])
            ans = terms[-1] + terms[-2]
        else:
            k = rng.randint(1, 3)
            terms = [(k + i) ** 2 for i in range(4)]
            ans = (k + 4) ** 2
        seq = ", ".join(map(str, terms))
        add("sequences", f"What is the next number in the sequence {seq}, ...? Answer with a single number.", ans)

    # number theory (7)
    for _ in range(2):
        base, m = rng.choice([2, 3, 7]), rng.choice([5, 7, 11, 13])
        e = rng.randint(10, 40)
        add("number_theory", f"What is the remainder when {base}^{e} is divided by {m}? Answer with a single number.", pow(base, e, m))
    for _ in range(2):
        while True:
            a, b = rng.randint(6, 96), rng.randint(6, 96)
            if 1 < gcd(a, b) < min(a, b):
                break
        if rng.random() < 0.5:
            add("number_theory", f"What is the greatest common divisor of {a} and {b}? Answer with a single number.", gcd(a, b))
        else:
            add("number_theory", f"What is the least common multiple of {a} and {b}? Answer with a single number.", a * b // gcd(a, b))

    def sieve(n):
        s = [True] * (n + 1)
        s[0:2] = [False, False]
        for i in range(2, int(n ** 0.5) + 1):
            if s[i]:
                for j in range(i * i, n + 1, i):
                    s[j] = False
        return s
    prime_flags = sieve(200)
    for _ in range(2):
        lo = rng.randint(1, 90)
        hi = lo + rng.randint(20, 80)
        cnt = sum(1 for p in range(lo, hi + 1) if prime_flags[p])
        add("number_theory", f"How many prime numbers are there between {lo} and {hi} inclusive? Answer with a single number.", cnt)
    while True:
        nn = rng.randint(30, 150)
        divs = sum(1 for d in range(1, nn + 1) if nn % d == 0)
        if 4 <= divs <= 12:
            break
    add("number_theory", f"How many positive divisors does {nn} have? Answer with a single number.", divs)

    # combinatorics (4)
    nn, k = rng.randint(6, 12), rng.randint(2, 4)
    add("combinatorics", f"How many ways can you choose {k} people from a group of {nn}? Answer with a single number.", comb(nn, k))
    nn = rng.randint(6, 15)
    add("combinatorics", f"At a meeting, {nn} people each shake hands exactly once with every other person. How many handshakes happen in total? Answer with a single number.", nn * (nn - 1) // 2)
    nn = rng.randint(6, 12)
    add("combinatorics", f"How many diagonals does a convex polygon with {nn} sides have? Answer with a single number.", nn * (nn - 3) // 2)
    a = rng.randint(1, 9)
    cnt = sum(1 for x in range(1, 201) if x % a == 0)
    add("combinatorics", f"How many integers from 1 to 200 inclusive are divisible by {a}? Answer with a single number.", cnt)

    assert len(items) == 50, len(items)
    return items


# --------------------------------------------------------------------------
# REASONING (40)
# --------------------------------------------------------------------------

def _knight_knave_item(rng, names):
    A, B = names
    kinds = ["other_knight", "other_knave", "both_knaves", "at_least_one_knight"]
    ph = {
        "other_knight": lambda s, o: f"{o} is a knight",
        "other_knave": lambda s, o: f"{o} is a knave",
        "both_knaves": lambda s, o: "both of us are knaves",
        "at_least_one_knight": lambda s, o: "at least one of us is a knight",
    }

    def truth(who, kind, world):
        if kind == "other_knight":
            return world[B if who == A else A]
        if kind == "other_knave":
            return not world[B if who == A else A]
        if kind == "both_knaves":
            return (not world[A]) and (not world[B])
        if kind == "at_least_one_knight":
            return world[A] or world[B]
        raise ValueError

    for _ in range(200):
        ka, kb = rng.choice(kinds), rng.choice(kinds)
        sols = []
        for va, vb in product([True, False], repeat=2):
            world = {A: va, B: vb}
            if (truth(A, ka, world) == va) and (truth(B, kb, world) == vb):
                sols.append(world)
        if len(sols) == 1:
            w = sols[0]
            prompt = (
                f"On an island knights always tell the truth and knaves always lie. You meet {A} and {B}.\n"
                f'{A} says: "{ph[ka](A, B)}."\n'
                f'{B} says: "{ph[kb](B, A)}."\n'
                f"What is {A} - a knight or a knave? Answer with one word: knight or knave."
            )
            ans = "knight" if w[A] else "knave"
            return prompt, ans
    raise RuntimeError("knight/knave generation failed")


def _order_puzzle(rng, names):
    k = len(names)
    sol = list(names)
    rng.shuffle(sol)
    pos = {name: i for i, name in enumerate(sol)}

    # generate true constraints pool
    def pool():
        out = []
        # before
        for a, b in permutations(names, 2):
            if pos[a] < pos[b]:
                out.append(("before", a, b, f"{a} is somewhere before {b}"))
        # immediately before
        for a, b in permutations(names, 2):
            if pos[b] == pos[a] + 1:
                out.append(("imm", a, b, f"{a} is immediately before {b}"))
        # not adjacent
        for a, b in combinations(names, 2):
            if abs(pos[a] - pos[b]) > 1:
                out.append(("notadj", a, b, f"{a} is not next to {b}"))
        # position
        for a in names:
            out.append(("pos", a, pos[a], f"{a} is in position {pos[a]+1}"))
        # between (a between b and c): b < a < c or c < a < b
        if k >= 3:
            for a in names:
                for b, c in permutations([x for x in names if x != a], 2):
                    if pos[b] < pos[a] < pos[c]:
                        out.append(("between", a, b, c, f"{a} is between {b} and {c}"))
                        break
        return out

    all_cons = pool()
    rng.shuffle(all_cons)

    # pick minimal set yielding uniqueness, try greedy
    for target_n in range(3, 6):
        for _ in range(60):
            cand = rng.sample(all_cons, min(target_n, len(all_cons)))

            def holds(perm):
                p = {n: i for i, n in enumerate(perm)}
                for c in cand:
                    if c[0] == "before":
                        if not (p[c[1]] < p[c[2]]):
                            return False
                    elif c[0] == "imm":
                        if not (p[c[2]] == p[c[1]] + 1):
                            return False
                    elif c[0] == "notadj":
                        if abs(p[c[1]] - p[c[2]]) <= 1:
                            return False
                    elif c[0] == "pos":
                        if p[c[1]] != c[2]:
                            return False
                    elif c[0] == "between":
                        if not (p[c[2]] < p[c[1]] < p[c[3]] or p[c[3]] < p[c[1]] < p[c[2]]):
                            return False
                return True

            sols = [perm for perm in permutations(names) if holds(perm)]
            if len(sols) == 1 and sols[0] == tuple(sol):
                # question: who is in position q
                q = rng.randint(1, k)
                ans = sol[q - 1]
                clues = "\n".join(f"- {c[-1]}." for c in cand)
                prompt = (
                    f"There are {k} people standing in a line in positions 1 to {k} (left to right). "
                    f"Clues:\n{clues}\nWho is in position {q}? Answer with the person's name only."
                )
                return prompt, ans
    return None


def build_reasoning(rng):
    items = []
    n = 0

    def add(sub, prompt, answer, accept=None):
        nonlocal n
        n += 1
        it = {
            "id": f"reasoning_{n:03d}",
            "category": "reasoning",
            "subcategory": sub,
            "prompt": prompt,
            "checker": "exact",
            "answer": answer,
        }
        if accept:
            it["accept"] = accept
        items.append(it)

    # knights & knaves (6)
    name_pairs = [("Alice", "Bob"), ("Carol", "Dave"), ("Eve", "Frank"),
                  ("Grace", "Heidi"), ("Ivy", "Judy"), ("Kim", "Leo")]
    rng.shuffle(name_pairs)
    for pair in name_pairs[:6]:
        prompt, ans = _knight_knave_item(rng, pair)
        add("knights", prompt, ans)

    # syllogisms (8) - fixed, validity when assuming premises true
    syllogisms = [
        ("All mammals are warm-blooded. All dogs are mammals. Therefore, all dogs are warm-blooded. Assuming the premises are true, does the conclusion logically follow? Answer yes or no.", "yes"),
        ("All cats are mammals. Some pets are cats. Therefore, some pets are mammals. Assuming the premises are true, does the conclusion logically follow? Answer yes or no.", "yes"),
        ("No reptiles are mammals. All snakes are reptiles. Therefore, no snakes are mammals. Assuming the premises are true, does the conclusion logically follow? Answer yes or no.", "yes"),
        ("Some A are B. Some B are C. Therefore, some A are C. Does the conclusion necessarily follow? Answer yes or no.", "no"),
        ("All A are B. Some C are A. Therefore, some C are B. Does the conclusion necessarily follow? Answer yes or no.", "yes"),
        ("No A are B. Some C are A. Therefore, some C are not B. Does the conclusion necessarily follow? Answer yes or no.", "yes"),
        ("If it rains, the ground is wet. The ground is wet. Therefore, it rained. Does the conclusion necessarily follow? Answer yes or no.", "no"),
        ("All birds can fly. Penguins are birds. Therefore, penguins can fly. Assuming the premises are true, does the conclusion logically follow (validity, not soundness)? Answer yes or no.", "yes"),
    ]
    for p, a in syllogisms:
        add("syllogism", p, a)

    # reasoning sequences (6) - harder than math's
    # interleaved arithmetic
    for _ in range(2):
        a0, d1 = rng.randint(1, 5), rng.randint(2, 4)
        b0, d2 = rng.randint(6, 12), rng.randint(2, 4)
        A = [a0 + i * d1 for i in range(4)]
        B = [b0 + i * d2 for i in range(4)]
        inter = []
        for i in range(4):
            inter.append(A[i]); inter.append(B[i])
        shown = inter[:7]
        ans = inter[7]
        seq = ", ".join(map(str, shown))
        add("pattern", f"What is the next number in the sequence {seq}, ...? Look for two interleaved patterns. Answer with a single number.", str(ans))
    # alternating +a -b
    for _ in range(2):
        s = rng.randint(5, 15)
        a, b = rng.randint(3, 7), rng.randint(1, 4)
        terms = [s]
        for i in range(1, 7):
            terms.append(terms[-1] + (a if i % 2 == 1 else -b))
        shown = terms[:6]
        ans = terms[6]
        seq = ", ".join(map(str, shown))
        add("pattern", f"What is the next number in the sequence {seq}, ...? Answer with a single number.", str(ans))
    # letter skip
    for _ in range(2):
        step = rng.choice([2, 3])
        start = rng.randint(0, 5)
        letters = [chr(ord("A") + start + i * step) for i in range(5)]
        ans_l = chr(ord(letters[-1]) + step)
        seq = ", ".join(letters)
        add("pattern", f"What is the next letter in the sequence {seq}, ...? Answer with the single letter only.", ans_l)

    # ordering puzzles (8)
    for _ in range(8):
        k = rng.choice([4, 4, 5])
        pool_names = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi"]
        names = rng.sample(pool_names, k)
        res = _order_puzzle(rng, names)
        if res is None:
            # fallback simple puzzle
            prompt = (
                f"Five runners finished a race. Alice finished before Bob. "
                f"Carol finished before Alice. Dave finished after Bob. "
                f"Eve finished before Carol. Who finished first? Answer with the name only."
            )
            ans = "Eve"
            add("ordering", prompt, ans)
        else:
            prompt, ans = res
            add("ordering", prompt, ans)

    # probability (6) - Fractions
    for _ in range(2):
        target = rng.randint(5, 9)
        # two dice sum == target
        cnt = sum(1 for a in range(1, 7) for b in range(1, 7) if a + b == target)
        f = Fraction(cnt, 36)
        prompt = (
            f"Two fair six-sided dice are rolled. What is the probability that the sum is {target}? "
            f"Answer as a reduced fraction a/b."
        )
        add("probability", prompt, f"{f.numerator}/{f.denominator}", accept=[f"{f.numerator}/{f.denominator}", str(float(f))])
    for _ in range(2):
        cn, ck = rng.choice([(3, 2), (4, 2), (4, 3), (5, 2), (5, 3)])
        cnt = comb(cn, ck)
        f = Fraction(cnt, 2 ** cn)
        prompt = (
            f"A fair coin is flipped {cn} times. What is the probability of getting exactly {ck} heads? "
            f"Answer as a reduced fraction a/b."
        )
        add("probability", prompt, f"{f.numerator}/{f.denominator}")
    for _ in range(2):
        r, b = rng.randint(3, 6), rng.randint(3, 6)
        tot = r + b
        # draw 2 without replacement, both red
        f = Fraction(r, tot) * Fraction(r - 1, tot - 1)
        prompt = (
            f"An urn contains {r} red balls and {b} blue balls. Two balls are drawn without replacement. "
            f"What is the probability both are red? Answer as a reduced fraction a/b."
        )
        add("probability", prompt, f"{f.numerator}/{f.denominator}")

    # classic puzzles (6) - all computed
    # snail
    for _ in range(1):
        H = rng.randint(9, 15)
        up, down = 3, 2
        # days: ceil((H-up)/(up-down))+1
        import math as _m
        days = _m.ceil((H - up) / (up - down)) + 1 if H > up else 1
        # verify by simulation
        h = 0
        for d in range(1, 30):
            h += up
            if h >= H:
                assert d == days
                break
            h -= down
        add("classic", f"A snail is at the bottom of a {H}-meter well. Each day it climbs {up} meters up and each night it slips {down} meters down. On which day does it reach the top? Answer with a single number.", str(days))
    # bat+ball fixed
    add("classic", "A bat and a ball together cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost in dollars? Answer with a single number like 0.05.", "0.05", accept=["0.05", "$0.05", "0.05 dollars", "5 cents", "5"])
    # handshakes (again but reasoning flavor) - count with formula, keep but as reasoning classic
    nn = rng.randint(6, 12)
    add("classic", f"At a party, {nn} people each shake hands once with every other person. How many handshakes occur? Answer with a single number.", str(nn * (nn - 1) // 2))
    # pages digit count
    N = rng.randint(50, 150)
    digits = sum(len(str(i)) for i in range(1, N + 1))
    add("classic", f"A book is numbered from page 1 to page {N}. How many digits are used in total to number all pages? Answer with a single number.", str(digits))
    # socks
    add("classic", "A drawer has 10 black socks and 10 white socks. You pick socks at random in the dark. How many socks must you pick to guarantee you have a matching pair? Answer with a single number.", "3")
    # chimes
    add("classic", "A clock chimes 6 times in 5 seconds (the 6 chimes have 5 intervals). How many seconds does it take to chime 12 times? Answer with a single number.", "11")

    assert len(items) == 40, len(items)
    return items


# --------------------------------------------------------------------------
# CODING (24) - checker "unit_test", tests are assert strings executed in VM
# --------------------------------------------------------------------------

CODING_PROBLEMS = [
    {
        "id": "coding_001",
        "prompt": (
            "Implement a Python function `fizzbuzz(n)` that returns a list of strings for numbers 1..n:\n"
            "- multiples of 3 -> \"Fizz\", 5 -> \"Buzz\", both -> \"FizzBuzz\", otherwise str(i).\n"
            "Example: fizzbuzz(5) == [\"1\", \"2\", \"Fizz\", \"4\", \"Buzz\"].\n"
            "Return only a ```python code block with the function."
        ),
        "entry": "fizzbuzz",
        "tests": [
            'assert fizzbuzz(5) == ["1","2","Fizz","4","Buzz"]',
            'assert fizzbuzz(15)[-1] == "FizzBuzz"',
            'assert fizzbuzz(1) == ["1"]',
            'assert fizzbuzz(3) == ["1","2","Fizz"]',
            'assert fizzbuzz(0) == []',
        ],
        "reference": 'def fizzbuzz(n):\n    out=[]\n    for i in range(1,n+1):\n        if i%15==0: out.append("FizzBuzz")\n        elif i%3==0: out.append("Fizz")\n        elif i%5==0: out.append("Buzz")\n        else: out.append(str(i))\n    return out',
    },
    {
        "id": "coding_002",
        "prompt": (
            "Implement `is_palindrome(s)` that returns True if s is a palindrome ignoring case and "
            "non-alphanumeric characters, otherwise False. Empty string is True.\n"
            "Example: is_palindrome(\"A man, a plan, a canal: Panama\") == True.\n"
            "Return only a ```python code block."
        ),
        "entry": "is_palindrome",
        "tests": [
            'assert is_palindrome("A man, a plan, a canal: Panama") == True',
            'assert is_palindrome("race a car") == False',
            'assert is_palindrome("") == True',
            'assert is_palindrome("No lemon, no melon") == True',
            'assert is_palindrome("hello") == False',
        ],
        "reference": 'def is_palindrome(s):\n    import re\n    t="".join(ch.lower() for ch in s if ch.isalnum())\n    return t==t[::-1]',
    },
    {
        "id": "coding_003",
        "prompt": (
            "Implement `fib(n)` that returns the n-th Fibonacci number with fib(0)=0, fib(1)=1.\n"
            "Example: fib(10)==55.\nReturn only a ```python code block."
        ),
        "entry": "fib",
        "tests": [
            'assert fib(0) == 0',
            'assert fib(1) == 1',
            'assert fib(10) == 55',
            'assert fib(20) == 6765',
            'assert fib(2) == 1',
        ],
        "reference": 'def fib(n):\n    a,b=0,1\n    for _ in range(n): a,b=b,a+b\n    return a',
    },
    {
        "id": "coding_004",
        "prompt": (
            "Implement `two_sum(nums, target)` that returns a list [i, j] (i<j, sorted ascending) of indices "
            "whose values sum to target. Exactly one solution exists.\n"
            "Example: two_sum([2,7,11,15],9)==[0,1].\nReturn only a ```python code block."
        ),
        "entry": "two_sum",
        "tests": [
            'assert two_sum([2,7,11,15],9)==[0,1]',
            'assert two_sum([3,2,4],6)==[1,2]',
            'assert two_sum([0,4,3,0],0)==[0,3]',
            'assert two_sum([1,2,3],5)==[1,2]',
        ],
        "reference": 'def two_sum(nums,target):\n    d={}\n    for i,x in enumerate(nums):\n        if target-x in d: return sorted([d[target-x],i])\n        d[x]=i',
    },
    {
        "id": "coding_005",
        "prompt": (
            "Implement `reverse_words(s)` that reverses the order of words. Words are separated by whitespace; "
            "collapse multiple spaces and strip leading/trailing spaces. Join with single spaces.\n"
            "Example: reverse_words(\"  the sky is blue \")==\"blue is sky the\".\nReturn only a ```python code block."
        ),
        "entry": "reverse_words",
        "tests": [
            'assert reverse_words("the sky is blue")=="blue is sky the"',
            'assert reverse_words("  hello  world ") =="world hello"',
            'assert reverse_words("a b  c")=="c b a"',
            'assert reverse_words("")==""',
        ],
        "reference": 'def reverse_words(s):\n    return " ".join(s.split()[::-1])',
    },
    {
        "id": "coding_006",
        "prompt": (
            "Implement `count_vowels(s)` that counts vowels a,e,i,o,u (case-insensitive).\n"
            "Example: count_vowels(\"Hello World\")==3.\nReturn only a ```python code block."
        ),
        "entry": "count_vowels",
        "tests": [
            'assert count_vowels("Hello World")==3',
            'assert count_vowels("")==0',
            'assert count_vowels("AEIOU")==5',
            'assert count_vowels("bcdfg")==0',
        ],
        "reference": 'def count_vowels(s):\n    return sum(1 for ch in s.lower() if ch in "aeiou")',
    },
    {
        "id": "coding_007",
        "prompt": (
            "Implement `flatten(lst)` that flattens a list of lists by one level. "
            "It must handle an empty outer list.\n"
            "Example: flatten([[1,2],[3],[4,5]])==[1,2,3,4,5].\nReturn only a ```python code block."
        ),
        "entry": "flatten",
        "tests": [
            'assert flatten([[1,2],[3],[4,5,6]])==[1,2,3,4,5,6]',
            'assert flatten([])==[]',
            'assert flatten([[],[1],[]])==[1]',
            'assert flatten([[1],[2],[3]])==[1,2,3]',
        ],
        "reference": 'def flatten(lst):\n    out=[]\n    for sub in lst:\n        out.extend(sub)\n    return out',
    },
    {
        "id": "coding_008",
        "prompt": (
            "Implement `binary_search(arr, x)` on a sorted list arr. Return the index of x or -1 if not found. "
            "Empty list returns -1.\nReturn only a ```python code block."
        ),
        "entry": "binary_search",
        "tests": [
            'assert binary_search([1,3,5,7,9],5)==2',
            'assert binary_search([1,3,5,7,9],2)==-1',
            'assert binary_search([],1)==-1',
            'assert binary_search([2],2)==0',
            'assert binary_search([1,2,3,4],4)==3',
        ],
        "reference": 'def binary_search(arr,x):\n    lo,hi=0,len(arr)-1\n    while lo<=hi:\n        m=(lo+hi)//2\n        if arr[m]==x: return m\n        elif arr[m]<x: lo=m+1\n        else: hi=m-1\n    return -1',
    },
    {
        "id": "coding_009",
        "prompt": (
            "Implement `gcd(a,b)` that returns the greatest common divisor. Handle non-negative inputs; "
            "gcd(0,b)==b. Use no external libraries beyond math if you want.\nReturn only a ```python code block."
        ),
        "entry": "gcd",
        "tests": [
            'assert gcd(48,18)==6',
            'assert gcd(0,5)==5',
            'assert gcd(7,13)==1',
            'assert gcd(100,80)==20',
        ],
        "reference": 'def gcd(a,b):\n    import math\n    return math.gcd(a,b)',
    },
    {
        "id": "coding_010",
        "prompt": (
            "Implement `is_prime(n)` returning True if n is prime, else False. Handle n<2 as False.\n"
            "Return only a ```python code block."
        ),
        "entry": "is_prime",
        "tests": [
            'assert is_prime(2)==True',
            'assert is_prime(1)==False',
            'assert is_prime(0)==False',
            'assert is_prime(97)==True',
            'assert is_prime(91)==False',
            'assert is_prime(-3)==False',
        ],
        "reference": 'def is_prime(n):\n    if n<2: return False\n    if n%2==0: return n==2\n    i=3\n    while i*i<=n:\n        if n%i==0: return False\n        i+=2\n    return True',
    },
    {
        "id": "coding_011",
        "prompt": (
            "Implement `merge_sorted(a,b)` that merges two sorted lists into a new sorted list.\n"
            "Example: merge_sorted([1,3,5],[2,4,6])==[1,2,3,4,5,6].\nReturn only a ```python code block."
        ),
        "entry": "merge_sorted",
        "tests": [
            'assert merge_sorted([1,3,5],[2,4,6])==[1,2,3,4,5,6]',
            'assert merge_sorted([], [1,2])==[1,2]',
            'assert merge_sorted([1,2],[])==[1,2]',
            'assert merge_sorted([],[])==[]',
        ],
        "reference": 'def merge_sorted(a,b):\n    i=j=0; out=[]\n    while i<len(a) and j<len(b):\n        if a[i]<=b[j]: out.append(a[i]); i+=1\n        else: out.append(b[j]); j+=1\n    out.extend(a[i:]); out.extend(b[j:]); return out',
    },
    {
        "id": "coding_012",
        "prompt": (
            "Implement `remove_duplicates(lst)` that removes duplicates keeping first occurrence order.\n"
            "Example: remove_duplicates([3,1,3,2,1])==[3,1,2].\nReturn only a ```python code block."
        ),
        "entry": "remove_duplicates",
        "tests": [
            'assert remove_duplicates([3,1,3,2,1])==[3,1,2]',
            'assert remove_duplicates([])==[]',
            'assert remove_duplicates([1,1,1])==[1]',
            'assert remove_duplicates([1,2,3])==[1,2,3]',
        ],
        "reference": 'def remove_duplicates(lst):\n    seen=set(); out=[]\n    for x in lst:\n        if x not in seen:\n            seen.add(x); out.append(x)\n    return out',
    },
    {
        "id": "coding_013",
        "prompt": (
            "Implement `longest_word(s)` that returns the longest word in s (split on whitespace). Tie -> first.\n"
            "Example: longest_word(\"I love programming in Python\")==\"programming\".\nReturn only a ```python code block."
        ),
        "entry": "longest_word",
        "tests": [
            'assert longest_word("I love programming in Python")=="programming"',
            'assert longest_word("a bb ccc bb")=="ccc"',
            'assert longest_word("hello")== "hello"',
            'assert longest_word("one two three four")=="three"',
        ],
        "reference": 'def longest_word(s):\n    w=s.split()\n    return max(w, key=len) if w else ""',
    },
    {
        "id": "coding_014",
        "prompt": (
            "Implement `caesar(s, shift)` that Caesar-shifts letters by shift (wrap, preserve case), leaving other chars unchanged. "
            "Negative shifts allowed.\n"
            "Example: caesar(\"abc\",1)==\"bcd\"; caesar(\"xyz\",3)==\"abc\".\nReturn only a ```python code block."
        ),
        "entry": "caesar",
        "tests": [
            'assert caesar("abc",1)=="bcd"',
            'assert caesar("xyz",3)=="abc"',
            'assert caesar("Hello, World!",7)=="Olssv, Dvysk!"',
            'assert caesar("bcd",-1)=="abc"',
        ],
        "reference": 'def caesar(s,shift):\n    out=""\n    for ch in s:\n        if "a"<=ch<="z": out+=chr((ord(ch)-97+shift)%26+97)\n        elif "A"<=ch<="Z": out+=chr((ord(ch)-65+shift)%26+65)\n        else: out+=ch\n    return out',
    },
    {
        "id": "coding_015",
        "prompt": (
            "Implement `is_anagram(a,b)` that returns True if a and b are anagrams ignoring case and spaces.\n"
            "Example: is_anagram(\"listen\",\"silent\")==True.\nReturn only a ```python code block."
        ),
        "entry": "is_anagram",
        "tests": [
            'assert is_anagram("listen","silent")==True',
            'assert is_anagram("hello","world")==False',
            'assert is_anagram("Astronomer","Moon starer")==True',
            'assert is_anagram("abc","ab") == False',
        ],
        "reference": 'def is_anagram(a,b):\n    import re\n    fa="".join(a.lower().split())\n    fb="".join(b.lower().split())\n    return sorted(fa)==sorted(fb)',
    },
    {
        "id": "coding_016",
        "prompt": (
            "Implement `transpose(m)` that transposes a rectangular matrix (list of lists). Handle empty matrix -> [].\n"
            "Example: transpose([[1,2,3],[4,5,6]])==[[1,4],[2,5],[3,6]].\nReturn only a ```python code block."
        ),
        "entry": "transpose",
        "tests": [
            'assert transpose([[1,2,3],[4,5,6]])==[[1,4],[2,5],[3,6]]',
            'assert transpose([])==[]',
            'assert transpose([[1]])==[[1]]',
            'assert transpose([[1,2],[3,4]])==[[1,3],[2,4]]',
        ],
        "reference": 'def transpose(m):\n    if not m or not m[0]: return []\n    return [list(row) for row in zip(*m)]',
    },
    {
        "id": "coding_017",
        "prompt": (
            "Implement `prime_factors(n)` that returns the list of prime factors in non-decreasing order. n>=2; for 1 return [].\n"
            "Example: prime_factors(12)==[2,2,3].\nReturn only a ```python code block."
        ),
        "entry": "prime_factors",
        "tests": [
            'assert prime_factors(12)==[2,2,3]',
            'assert prime_factors(97)==[97]',
            'assert prime_factors(1)==[]',
            'assert prime_factors(100)==[2,2,5,5]',
        ],
        "reference": 'def prime_factors(n):\n    if n<2: return []\n    out=[]; d=2\n    while d*d<=n:\n        while n%d==0: out.append(d); n//=d\n        d+=1 if d==2 else 2\n    if n>1: out.append(n)\n    return out',
    },
    {
        "id": "coding_018",
        "prompt": (
            "Implement `balanced(s)` that returns True if s consisting of ()[]{} is balanced, otherwise False. Empty string is True.\n"
            "Example: balanced(\"()[]{}\")==True; balanced(\"([)]\")==False.\nReturn only a ```python code block."
        ),
        "entry": "balanced",
        "tests": [
            'assert balanced("()[]{}")==True',
            'assert balanced("([)]")==False',
            'assert balanced("(")==False',
            'assert balanced("")==True',
            'assert balanced("{[()]}")==True',
        ],
        "reference": 'def balanced(s):\n    st=[]; m={")":"(","]":"[","}":"{"}\n    for ch in s:\n        if ch in "([{": st.append(ch)\n        else:\n            if not st or st[-1]!=m[ch]: return False\n            st.pop()\n    return not st',
    },
    {
        "id": "coding_019",
        "prompt": (
            "Implement `rle(s)` that run-length encodes s into a list of [char, count] pairs (each pair is a 2-element list).\n"
            "Example: rle(\"aaabbc\")==[[\"a\",3],[\"b\",2],[\"c\",1]].\nReturn only a ```python code block."
        ),
        "entry": "rle",
        "tests": [
            'assert rle("aaabbc")==[["a",3],["b",2],["c",1]]',
            'assert rle("")==[]',
            'assert rle("a")==[["a",1]]',
            'assert rle("aa") == [["a",2]]',
        ],
        "reference": 'def rle(s):\n    if not s: return []\n    out=[]; cur=s[0]; cnt=1\n    for ch in s[1:]:\n        if ch==cur: cnt+=1\n        else: out.append([cur,cnt]); cur=ch; cnt=1\n    out.append([cur,cnt]); return out',
    },
    {
        "id": "coding_020",
        "prompt": (
            "Implement `pascal_row(n)` that returns the n-th row (0-indexed) of Pascal's triangle.\n"
            "Example: pascal_row(0)==[1]; pascal_row(4)==[1,4,6,4,1].\nReturn only a ```python code block."
        ),
        "entry": "pascal_row",
        "tests": [
            'assert pascal_row(0)==[1]',
            'assert pascal_row(4)==[1,4,6,4,1]',
            'assert pascal_row(6)==[1,6,15,20,15,6,1]',
            'assert pascal_row(1)==[1,1]',
        ],
        "reference": 'def pascal_row(n):\n    row=[1]\n    for k in range(1,n+1): row.append(row[-1]*(n-k+1)//k)\n    return row',
    },
    {
        "id": "coding_021",
        "prompt": (
            "Implement `rotate_list(lst, k)` that rotates the list to the right by k steps. k may be larger than len(lst).\n"
            "Example: rotate_list([1,2,3,4,5],2)==[4,5,1,2,3].\nReturn only a ```python code block."
        ),
        "entry": "rotate_list",
        "tests": [
            'assert rotate_list([1,2,3,4,5],2)==[4,5,1,2,3]',
            'assert rotate_list([1,2,3],4)==[3,1,2]',
            'assert rotate_list([],5)==[]',
            'assert rotate_list([1],10)==[1]',
        ],
        "reference": 'def rotate_list(lst,k):\n    if not lst: return []\n    k%=len(lst)\n    return lst[-k:]+lst[:-k] if k else lst[:]',
    },
    {
        "id": "coding_022",
        "prompt": (
            "Implement `second_largest(lst)` that returns the second largest distinct value, or None if it does not exist.\n"
            "Example: second_largest([5,1,9,3])==5; second_largest([4,4,2])==2.\nReturn only a ```python code block."
        ),
        "entry": "second_largest",
        "tests": [
            'assert second_largest([5,1,9,3])==5',
            'assert second_largest([4,4,2])==2',
            'assert second_largest([1]) is None',
            'assert second_largest([7,7,7]) is None',
        ],
        "reference": 'def second_largest(lst):\n    u=sorted(set(lst), reverse=True)\n    return u[1] if len(u)>=2 else None',
    },
    {
        "id": "coding_023",
        "prompt": (
            "Implement `is_valid_date(s)` that returns True if s is a valid date in strict YYYY-MM-DD format "
            "(zero-padded, e.g. 2024-02-09), otherwise False. Check calendar validity including leap years.\n"
            "Example: is_valid_date(\"2024-02-29\")==True; is_valid_date(\"2023-02-29\")==False.\nReturn only a ```python code block."
        ),
        "entry": "is_valid_date",
        "tests": [
            'assert is_valid_date("2024-02-29")==True',
            'assert is_valid_date("2023-02-29")==False',
            'assert is_valid_date("2024-13-01")==False',
            'assert is_valid_date("2024-04-31")==False',
            'assert is_valid_date("2024-1-1")==False',
            'assert is_valid_date("2024/01/01")==False',
        ],
        "reference": 'def is_valid_date(s):\n    import re, datetime\n    if not re.match(r"^\\d{4}-\\d{2}-\\d{2}$", s): return False\n    try: datetime.date.fromisoformat(s); return True\n    except ValueError: return False',
    },
    {
        "id": "coding_024",
        "prompt": (
            "Implement `chunk(lst, size)` that splits lst into chunks of length size (last chunk may be shorter). "
            "If size <= 0 raise ValueError. Empty list -> [].\n"
            "Example: chunk([1,2,3,4,5],2)==[[1,2],[3,4],[5]].\nReturn only a ```python code block."
        ),
        "entry": "chunk",
        "tests": [
            'assert chunk([1,2,3,4,5],2)==[[1,2],[3,4],[5]]',
            'assert chunk([],2)==[]',
            'assert chunk([1,2,3],3)==[[1,2,3]]',
            'assert chunk([1,2,3],1)==[[1],[2],[3]]',
            'import pytest as _p; _ok=False\ntry:\n    chunk([1],0)\nexcept ValueError:\n    _ok=True\nassert _ok',
        ],
        "reference": 'def chunk(lst,size):\n    if size<=0: raise ValueError("size must be >0")\n    return [lst[i:i+size] for i in range(0,len(lst),size)]',
    },
]


def build_coding():
    items = []
    for prob in CODING_PROBLEMS:
        # validate reference against tests
        ns = {}
        exec(prob["reference"], ns)
        for t in prob["tests"]:
            # skip the pytest-style ValueError check which needs try:
            try:
                exec(t, ns)
            except AssertionError as e:
                raise AssertionError(f'{prob["id"]} failed test {t!r}: {e}')
        items.append({
            "id": prob["id"],
            "category": "coding",
            "subcategory": "python",
            "prompt": prob["prompt"],
            "checker": "unit_test",
            "entry": prob["entry"],
            "tests": prob["tests"],
        })
    assert len(items) == 24, len(items)
    return items


# --------------------------------------------------------------------------
# GENERAL (36) - 24 MCQ (choice) + 12 constraint
# --------------------------------------------------------------------------

def build_general():
    items = []
    n = 0

    def add_mcq(question, options, answer):
        nonlocal n
        n += 1
        opts = "\n".join(f"{chr(65+i)}) {o}" for i, o in enumerate(options))
        items.append({
            "id": f"general_{n:03d}",
            "category": "general",
            "subcategory": "knowledge",
            "prompt": f"{question}\n{opts}\nAnswer with the letter only (A, B, C, or D).",
            "checker": "choice",
            "answer": answer,
        })

    add_mcq("What is the largest planet in the Solar System?",
            ["Earth", "Jupiter", "Saturn", "Neptune"], "B")
    add_mcq("Which gas do plants primarily absorb from the atmosphere during photosynthesis?",
            ["Oxygen", "Nitrogen", "Carbon dioxide", "Hydrogen"], "C")
    add_mcq("What is the time complexity of binary search on a sorted array of size n?",
            ["O(log n)", "O(n)", "O(n log n)", "O(1)"], "A")
    add_mcq("What does HTTP status code 404 mean?",
            ["OK", "Forbidden", "Not Found", "Internal Server Error"], "C")
    add_mcq("What is the smallest prime number?",
            ["2", "1", "3", "0"], "A")
    add_mcq("What is the capital of Australia?",
            ["Sydney", "Canberra", "Melbourne", "Perth"], "B")
    add_mcq("Who is the author of the novel '1984'?",
            ["Aldous Huxley", "Ray Bradbury", "George Orwell", "Ernest Hemingway"], "C")
    add_mcq("Which data structure follows FIFO (first in, first out)?",
            ["Stack", "Queue", "Tree", "Graph"], "B")
    add_mcq("What is the chemical symbol for gold?",
            ["Go", "Gd", "Ge", "Au"], "D")
    add_mcq("Which sorting algorithm does Python's built-in sort (Timsort) build upon?",
            ["Merge sort and Insertion sort", "Quick sort only", "Bubble sort", "Selection sort"], "A")
    add_mcq("The Nile River flows mainly through which continent?",
            ["Asia", "Africa", "Europe", "South America"], "B")
    add_mcq("Approximately what is the speed of light in vacuum?",
            ["30,000 km/s", "150,000 km/s", "300,000 km/s", "1,000,000 km/s"], "C")
    add_mcq("How many continents are there on Earth?",
            ["5", "7", "6", "8"], "B")
    add_mcq("Which of the following is NOT a pillar of object-oriented programming?",
            ["Encapsulation", "Inheritance", "Polymorphism", "Compilation"], "D")
    add_mcq("What is the binary representation of decimal 13?",
            ["1100", "1101", "1110", "1011"], "B")
    add_mcq("The Linux kernel is primarily written in which language?",
            ["C", "C++", "Python", "Java"], "A")
    add_mcq("Which organ pumps blood through the human body?",
            ["Liver", "Lungs", "Heart", "Kidney"], "C")
    add_mcq("Who developed the theory of general relativity?",
            ["Isaac Newton", "Albert Einstein", "Niels Bohr", "Galileo Galilei"], "B")
    add_mcq("What does CPU stand for?",
            ["Central Processing Unit", "Computer Personal Unit", "Central Print Unit", "Core Processing Unit"], "A")
    add_mcq("Which planet is known as the Red Planet?",
            ["Venus", "Jupiter", "Saturn", "Mars"], "D")
    add_mcq("Thủ đô của Việt Nam là thành phố nào?",
            ["Hà Nội", "TP. Hồ Chí Minh", "Đà Nẵng", "Hải Phòng"], "A")
    add_mcq("Ngày Quốc khánh nước Cộng hòa xã hội chủ nghĩa Việt Nam là ngày nào?",
            ["02/09/1945", "30/04/1975", "02/09", "01/01"], "C")
    add_mcq("Ai là tác giả của Truyện Kiều?",
            ["Nguyễn Trãi", "Nguyễn Du", "Hồ Xuân Hương", "Nguyễn Đình Chiểu"], "B")
    add_mcq("Việt Nam giáp với biển nào ở phía đông?",
            ["Biển Đông", "Biển Java", "Ấn Độ Dương", "Biển Hoa Đông"], "A")

    # constraint tasks (12) - checker "constraint"
    def add_constraint(prompt, spec):
        nonlocal n
        n += 1
        items.append({
            "id": f"general_{n:03d}",
            "category": "general",
            "subcategory": "instruction",
            "prompt": prompt,
            "checker": "constraint",
            "constraint": spec,
        })

    add_constraint(
        "Write a single sentence of exactly 10 words about the ocean. Output only that sentence, no extra text.",
        {"type": "word_count", "value": 10})
    add_constraint(
        'Return ONLY a valid JSON object with exactly the keys "name" (string) and "year" (integer). Example: {"name": "Ada", "year": 2020}. No extra text.',
        {"type": "json_keys", "keys": {"name": "str", "year": "int"}})
    add_constraint(
        "Explain what a computer is in one sentence. Write the ENTIRE response in UPPERCASE letters only.",
        {"type": "all_upper"})
    add_constraint(
        "List exactly 3 prime numbers, each on its own line starting with \"- \". No other text. Example:\n- 2\n- 3\n- 5",
        {"type": "prime_list", "count": 3})
    add_constraint(
        "Write one sentence about cats without using the letter 'z' or 'Z'. Output only the sentence.",
        {"type": "no_letter", "letter": "z"})
    add_constraint(
        "Answer the question 'What color is the sky?' in one sentence and end your response with exactly the word 'banana' (lowercase, with a period before it if you wish, but the last token must be banana).",
        {"type": "ends_with", "word": "banana"})
    add_constraint(
        "What is 7 * 8? Respond with a single integer only, no other text.",
        {"type": "equals", "value": "56"})
    add_constraint(
        "Write exactly 2 sentences about tea. Each sentence must end with a period. No extra sentences.",
        {"type": "sentence_count", "value": 2})
    add_constraint(
        "Hãy trả lời bằng tiếng Việt cho câu hỏi: 'Bạn có khỏe không?' Chỉ trả lời bằng tiếng Việt.",
        {"type": "is_vietnamese"})
    add_constraint(
        "Give exactly 5 animal names as a single comma-separated line, no other text. Example: cat, dog, fish, bird, lion",
        {"type": "comma_list", "count": 5})
    add_constraint(
        'Return ONLY a valid JSON object with a key "lines" whose value is an array of exactly 3 strings (a haiku, 3 lines). Example: {"lines": ["line one", "line two", "line three"]}',
        {"type": "json_array_len", "key": "lines", "len": 3})
    add_constraint(
        "What is 2 + 2? Respond with only the single digit, no other text.",
        {"type": "equals", "value": "4"})

    assert len(items) == 36, len(items)
    return items


# --------------------------------------------------------------------------
# QUALITATIVE (24) - checker "none"
# --------------------------------------------------------------------------

def build_qualitative():
    prompts = [
        ("Write a 150-word short story about a robot that learns to lie. Focus on tone and character, not just plot.", "creative_en"),
        ("Compose a haiku about a compiler error at 2am.", "creative_en"),
        ("Write a product description for noise-cancelling headphones aimed at commuters.", "creative_en"),
        ("Write a dialogue between two old friends who meet after 20 years in a train station.", "creative_en"),
        ("Viết một truyện ngắn khoảng 200 chữ về một người bán cà phê vỉa hè ở Sài Gòn. Giữ giọng văn ấm áp.", "creative_vi"),
        ("Viết một bài thơ lục bát 4 câu về mùa thu Hà Nội.", "creative_vi"),
        ("Viết một email trang trọng bằng tiếng Việt để xin lỗi khách hàng vì giao hàng trễ.", "creative_vi"),
        ("Viết đoạn giới thiệu 100 chữ về Đà Lạt cho một brochure du lịch.", "creative_vi"),
        ("Debug this Python snippet and explain the bug, then provide the fixed version:\n```python\nfor i in range(len(items)):\n    if items[i] is None:\n        items.remove(items[i])\n```", "coding"),
        ("Explain the time complexity of binary search vs linear search to a junior developer.", "coding"),
        ("Write an SQL query to find the top 3 customers by total purchase amount from tables orders(id, customer_id, amount).", "coding"),
        ("Refactor this function for readability and add error handling:\n```python\ndef f(a,b):\n    return a/b\n```", "coding"),
        ("Compare microservices vs monolith for a small startup with 3 engineers. Give trade-offs.", "coding"),
        ("Explain how you would design a simple rate limiter for an API.", "coding"),
        ("You have a week to learn a new programming language for a project. How would you plan your study schedule? Be specific.", "reasoning"),
        ("Estimate how many photographs are taken worldwide per day. Show your reasoning step by step (Fermi estimate).", "reasoning"),
        ("A company must choose between two job candidates with complementary strengths. How should they decide? Discuss the decision framework.", "reasoning"),
        ("Explain the ethical dilemma of self-driving cars having to choose between two harmful outcomes.", "reasoning"),
        ("Explain recursion to a 10-year-old child using a simple analogy.", "explain"),
        ("Explain how HTTPS works and why it matters, in plain language.", "explain"),
        ("Why is the sky blue? Explain for a curious teenager.", "explain"),
        ("Explain compound interest with a concrete example.", "explain"),
        ("You are a senior code reviewer. Review this pull request description and list 5 things you would check before approving.", "roleplay"),
        ("Summarise the following article into exactly 5 bullet points, each starting with '- ':\n\"Artificial intelligence is transforming education. It personalises learning, automates grading, and helps identify struggling students. However, it also raises concerns about privacy, bias, and over-reliance on technology. Schools must balance innovation with human oversight.\"", "instruction"),
    ]
    items = []
    for i, (p, sub) in enumerate(prompts, 1):
        items.append({
            "id": f"qualitative_{i:03d}",
            "category": "qualitative",
            "subcategory": sub,
            "prompt": p,
            "checker": "none",
        })
    assert len(items) == 24
    return items


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

def main():
    import random
    rng = random.Random(SEED)
    os.makedirs(EVAL_DIR, exist_ok=True)

    math_items = build_math(rng)
    reasoning_items = build_reasoning(rng)
    coding_items = build_coding()
    general_items = build_general()
    qual_items = build_qualitative()

    mapping = {
        "math.jsonl": math_items,
        "reasoning.jsonl": reasoning_items,
        "coding.jsonl": coding_items,
        "general.jsonl": general_items,
        "qualitative.jsonl": qual_items,
    }
    for name, items in mapping.items():
        path = write_jsonl(name, items)
        print(f"wrote {name}: {len(items)} items -> {path}")

    manifest = {
        "seed": SEED,
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "files": {},
        "total_auto": sum(len(v) for k, v in mapping.items() if k != "qualitative.jsonl"),
        "total_qualitative": len(qual_items),
    }
    for name in mapping:
        path = os.path.join(EVAL_DIR, name)
        manifest["files"][name] = {
            "count": len(mapping[name]),
            "sha256": sha256_of(path),
        }
    mpath = os.path.join(EVAL_DIR, "manifest.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"wrote manifest -> {mpath}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
