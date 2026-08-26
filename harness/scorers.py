"""Scorers for the frozen benchmark sets."""
import json
import re
import subprocess
import sys
import textwrap
import tempfile
import os
from pathlib import Path


# ------------------------------------------------------------------ helpers
def normalize_exact(s: str) -> str:
    s = s.lower().strip()
    # keep a-z 0-9 / and space, replace others with space
    s = re.sub(r"[^a-z0-9/\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_boxed(text: str):
    m = re.findall(r"\\boxed\{([^}]+)\}", text)
    return m[-1].strip() if m else None


# ------------------------------------------------------------------ numeric
def score_numeric(item, response: str):
    expected = item["answer"]
    accepts = item.get("accept", [])
    candidates = [expected] + accepts

    boxed = extract_boxed(response)
    if boxed is not None:
        cand_raw = boxed
    else:
        nums = re.findall(r"-?\d[\d,]*\.?\d*(?:/\d+)?", response)
        if not nums:
            return 0.0, "no number found"
        cand_raw = nums[-1]

    cand_raw = cand_raw.replace(",", "").strip().rstrip(".")
    # handle $ prefix already removed by regex, but if boxed contains $ strip
    cand_raw = cand_raw.lstrip("$").strip()

    def parse_num(s):
        s = s.strip().lstrip("$").rstrip("%").strip()
        # extract first number in s (for accepts like "5 cents")
        mm = re.search(r"-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", s.replace(",", ""))
        if not mm:
            return None
        token = mm.group(0)
        try:
            if "/" in token:
                a, b = token.split("/")
                return float(a) / float(b)
            return float(token)
        except Exception:
            return None

    cand_val = parse_num(cand_raw)
    if cand_val is None:
        return 0.0, f"parse fail cand={cand_raw!r}"

    for acc in candidates:
        acc_val = parse_num(str(acc))
        if acc_val is None:
            continue
        if abs(cand_val - acc_val) < 1e-4:
            return 1.0, f"match cand={cand_raw} ~ {acc}"
    return 0.0, f"expected {expected} got {cand_raw} ({cand_val})"


# ------------------------------------------------------------------ exact
def score_exact(item, response: str):
    ans = item["answer"]
    accepts = [ans] + item.get("accept", [])
    norm_resp = normalize_exact(response)
    for acc in accepts:
        norm_acc = normalize_exact(str(acc))
        if not norm_acc:
            continue
        if norm_resp == norm_acc:
            return 1.0, "exact"
        # word-boundary substring
        if re.search(r"\b" + re.escape(norm_acc) + r"\b", norm_resp):
            return 1.0, f"contains {norm_acc!r}"
    return 0.0, f"not found {ans!r} in {norm_resp[:80]!r}"


# ------------------------------------------------------------------ choice (MCQ)
def score_choice(item, response: str):
    expected = item["answer"].strip().upper()  # A/B/C/D
    # priority 1: boxed letter
    boxed = extract_boxed(response)
    if boxed and re.match(r"^[A-Da-d]$", boxed.strip()):
        cand = boxed.strip().upper()
        return (1.0 if cand == expected else 0.0), f"boxed {cand} vs {expected}"
    # priority 2: explicit "answer is X" / "đáp án"
    pat = re.compile(r"(?:answer| đáp án|答)[:\s]*\(?([A-Da-d])\)?\b", re.I)
    ms = list(pat.finditer(response))
    if ms:
        cand = ms[-1].group(1).upper()
        return (1.0 if cand == expected else 0.0), f"explicit {cand} vs {expected}"
    # priority 3: line that is just "B" or "B."
    lines = [l.strip() for l in response.strip().splitlines() if l.strip()]
    if lines:
        last = lines[-1].strip().rstrip(".").strip()
        if re.match(r"^[A-Da-d]$", last):
            cand = last.upper()
            return (1.0 if cand == expected else 0.0), f"last line {cand} vs {expected}"
    # fallback: last isolated letter in last 400 chars
    tail = response[-400:]
    letters = re.findall(r"\b([A-Da-d])\b", tail)
    if letters:
        cand = letters[-1].upper()
        return (1.0 if cand == expected else 0.0), f"fallback {cand} vs {expected}"
    return 0.0, f"no choice found, expected {expected}"


# ------------------------------------------------------------------ constraint
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def extract_json(text: str):
    # try fence first
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.S | re.I)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # try first { ... } block (greedy last })
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start:end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            pass
    # try direct
    try:
        return json.loads(text.strip())
    except Exception:
        return None


def score_constraint(item, response: str):
    spec = item["constraint"]
    t = spec["type"]
    resp_strip = response.strip()

    if t == "word_count":
        n = spec["value"]
        words = resp_strip.split()
        # count non-empty tokens
        cnt = len([w for w in words if w.strip()])
        ok = cnt == n
        return (1.0 if ok else 0.0), f"words {cnt} vs {n}"

    if t == "json_keys":
        keys = spec["keys"]  # dict key->type str
        obj = extract_json(response)
        if obj is None or not isinstance(obj, dict):
            return 0.0, "no json object"
        if set(obj.keys()) != set(keys.keys()):
            return 0.0, f"keys {set(obj.keys())} vs {set(keys.keys())}"
        for k, typ in keys.items():
            v = obj[k]
            if typ == "str" and not isinstance(v, str):
                return 0.0, f"{k} not str"
            if typ == "int" and not (isinstance(v, int) and not isinstance(v, bool)):
                return 0.0, f"{k} not int"
        return 1.0, "json keys ok"

    if t == "all_upper":
        stripped = resp_strip
        if not stripped:
            return 0.0, "empty"
        # check every letter is upper
        letters = re.findall(r"[A-Za-z]", stripped)
        if not letters:
            return 0.0, "no letters"
        ok = all(ch.isupper() for ch in letters)
        return (1.0 if ok else 0.0), f"all_upper {ok}"

    if t == "prime_list":
        cnt = spec["count"]
        lines = [l.strip() for l in resp_strip.splitlines() if l.strip()]
        dash = [l for l in lines if l.startswith("- ")]
        if len(dash) != cnt:
            return 0.0, f"dash lines {len(dash)} vs {cnt}"
        for l in dash:
            num_s = l[2:].strip().rstrip(".")
            try:
                nv = int(num_s)
            except Exception:
                return 0.0, f"not int {num_s!r}"
            if not is_prime(nv):
                return 0.0, f"{nv} not prime"
        return 1.0, "prime list ok"

    if t == "no_letter":
        letter = spec["letter"]
        if letter.lower() in resp_strip.lower():
            return 0.0, f"contains {letter!r}"
        if not resp_strip:
            return 0.0, "empty"
        return 1.0, "no letter ok"

    if t == "ends_with":
        word = spec["word"].lower()
        tail = resp_strip.lower().rstrip(".!?,;:")
        # last token
        tokens = re.findall(r"[A-Za-zÀ-ỹ]+", tail)
        if not tokens:
            return 0.0, "no tokens"
        ok = tokens[-1] == word
        return (1.0 if ok else 0.0), f"last {tokens[-1]!r} vs {word!r}"

    if t == "equals":
        val = spec["value"]
        ok = resp_strip == val
        # also allow stripped? require exact
        return (1.0 if ok else 0.0), f"equals {resp_strip!r} vs {val!r}"

    if t == "sentence_count":
        n = spec["value"]
        # split on .!? and count non-empty
        parts = [p.strip() for p in re.split(r"[.!?]+", resp_strip) if p.strip()]
        ok = len(parts) == n
        return (1.0 if ok else 0.0), f"sentences {len(parts)} vs {n}"

    if t == "is_vietnamese":
        # count diacritics
        diac = len(re.findall(r"[ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]", resp_strip.lower()))
        ok = diac >= 4
        return (1.0 if ok else 0.0), f"diacritics {diac}"

    if t == "comma_list":
        cnt = spec["count"]
        # single line expected
        line = resp_strip.splitlines()[0] if resp_strip else ""
        parts = [p.strip() for p in line.split(",")]
        # if response has extra lines, fail? be lenient: take first non-empty line
        if len(parts) != cnt:
            return 0.0, f"parts {len(parts)} vs {cnt}: {parts!r}"
        for p in parts:
            if not p or not re.match(r"^[A-Za-zÀ-ỹ\s\-]+$", p):
                return 0.0, f"bad part {p!r}"
        return 1.0, "comma list ok"

    if t == "json_array_len":
        key = spec["key"]
        n = spec["len"]
        obj = extract_json(response)
        if obj is None or not isinstance(obj, dict):
            return 0.0, "no json"
        if key not in obj or not isinstance(obj[key], list):
            return 0.0, f"key {key!r} not list"
        if len(obj[key]) != n:
            return 0.0, f"len {len(obj[key])} vs {n}"
        return 1.0, "json array len ok"

    return 0.0, f"unknown constraint {t!r}"


# ------------------------------------------------------------------ unit_test
def extract_code(response: str, entry: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*\n?(.*?)\n?```", response, re.S)
    if blocks:
        for b in blocks:
            if f"def {entry}" in b:
                return b
        return blocks[0]
    m = re.search(rf"def\s+{re.escape(entry)}\s*\(.*?\):.*", response, re.S)
    if m:
        return m.group(0)
    return response


def score_unit_test(item, response: str):
    entry = item["entry"]
    tests = item["tests"]
    code = extract_code(response, entry)
    # build a temp script
    script = textwrap.dedent(f"""
import traceback, sys
{code}

tests = {tests!r}
passed = 0
total = len(tests)
for idx, t in enumerate(tests):
    try:
        exec(t, globals())
        passed += 1
    except AssertionError as e:
        print(f"FAIL {{idx}}: {{e}}", flush=True)
    except Exception as e:
        print(f"ERROR {{idx}}: {{type(e).__name__}}: {{e}}", flush=True)
        traceback.print_exc()
print(f"RESULT {{passed}}/{{total}}", flush=True)
""")
    # colab/Windows: write temp file
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tmp:
        tmp.write(script)
        tmp_path = tmp.name
    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=12,
        )
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        m = re.search(r"RESULT (\d+)/(\d+)", out)
        if not m:
            return 0.0, f"no RESULT in output: {out[:500]!r}"
        passed, total = int(m.group(1)), int(m.group(2))
        score = passed / total if total else 0.0
        return score, f"{passed}/{total} " + out.strip().splitlines()[-1]
    except subprocess.TimeoutExpired:
        # Fallback for eval kernel where subprocess is blocked
        try:
            ns = {}
            exec(code, ns)
            passed = 0
            for t in tests:
                try:
                    exec(t, ns)
                    passed += 1
                except AssertionError:
                    pass
                except Exception:
                    pass
            total = len(tests)
            return (passed / total if total else 0.0), f"{passed}/{total} (in-process fallback)"
        except Exception as e:
            return 0.0, f"in-process error {e}"
    except Exception as e:
        return 0.0, f"runner error {e}"
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

# ------------------------------------------------------------------ dispatch
def score_item(item, response: str):
    checker = item.get("checker", "none")
    if checker == "numeric":
        return score_numeric(item, response)
    if checker == "exact":
        return score_exact(item, response)
    if checker == "choice":
        return score_choice(item, response)
    if checker == "constraint":
        return score_constraint(item, response)
    if checker == "unit_test":
        return score_unit_test(item, response)
    if checker == "none":
        return 0.0, "qualitative (no score)"
    return 0.0, f"unknown checker {checker!r}"
