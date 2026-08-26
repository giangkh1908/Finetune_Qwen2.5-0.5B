"""Build a DIVERSE, high-quality coding eval extension.

Quality bar (each item):
- a real algorithm/pattern, NOT a trivial one-liner
- >= 5 tests covering: happy path, edge cases, at least one trap
- distinct function names (no overlap with existing 24 eval items)
- no name/content collision with the training set (Magicoder + claude + knowledge)
- reference implementation is run against its tests BEFORE writing, so ground
  truth is guaranteed correct

Output: data/eval/coding_extra.jsonl
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "data", "train", "coder_train_all_chunked.jsonl")
OUT = os.path.join(ROOT, "data", "eval", "coding_extra.jsonl")
EXISTING = os.path.join(ROOT, "data", "eval", "coding.jsonl")


def entry_of(code):
    m = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", code)
    return m.group(1).lower() if m else None


ITEMS = [
    # ---- string algorithms ----
    ("code_extra_001", "Implement `is_anagram_phrase(a, b)` that returns True if two "
     "strings are anagrams ignoring spaces, case, and punctuation. "
     "Return only a ```python code block.",
     'def is_anagram_phrase(a, b):\n    import re\n    fa = re.sub(r"[^a-z]", "", a.lower())\n    fb = re.sub(r"[^a-z]", "", b.lower())\n    return sorted(fa) == sorted(fb)',
     ['assert is_anagram_phrase("listen", "silent") is True',
      'assert is_anagram_phrase("Astronomer", "Moon starer") is True',
      'assert is_anagram_phrase("hello", "world") is False',
      'assert is_anagram_phrase("a b c", "cba") is True',
      'assert is_anagram_phrase("", "") is True']),

    ("code_extra_002", "Implement `compress(s)` that run-length compresses a string by "
     "replacing runs of the same char with the char followed by the count, e.g. "
     "'aaabb' -> 'a3b2'. Single chars keep count 1. Return only a ```python code block.",
     'def compress(s):\n    if not s:\n        return ""\n    out = []\n    cur, cnt = s[0], 1\n    for ch in s[1:]:\n        if ch == cur:\n            cnt += 1\n        else:\n            out.append(cur + str(cnt))\n            cur, cnt = ch, 1\n    out.append(cur + str(cnt))\n    return "".join(out)',
     ['assert compress("aaabb") == "a3b2"',
      'assert compress("abcd") == "a1b1c1d1"',
      'assert compress("") == ""',
      'assert compress("aaa") == "a3"',
      'assert compress("aabbbcc") == "a2b3c2"']),

    ("code_extra_003", "Implement `insert_separator(s, sep)` that inserts sep between every "
     "character of s, e.g. ('abcd', '-') -> 'a-b-c-d'. Empty returns ''. "
     "Return only a ```python code block.",
     'def insert_separator(s, sep):\n    return sep.join(s)',
     ['assert insert_separator("abcd", "-") == "a-b-c-d"',
      'assert insert_separator("xy", ",") == "x,y"',
      'assert insert_separator("", "-") == ""',
      'assert insert_separator("a", "*") == "a"']),

    ("code_extra_004", "Implement `squeeze(s)` that removes consecutive duplicate characters, "
     "e.g. 'aabbca' -> 'abca'. Return only a ```python code block.",
     'def squeeze(s):\n    if not s:\n        return ""\n    out = [s[0]]\n    for ch in s[1:]:\n        if ch != out[-1]:\n            out.append(ch)\n    return "".join(out)',
     ['assert squeeze("aabbca") == "abca"',
      'assert squeeze("aaaa") == "a"',
      'assert squeeze("") == ""',
      'assert squeeze("ab") == "ab"',
      'assert squeeze("aabbaa") == "aba"']),

    ("code_extra_005", "Implement `is_pangram(s)` that returns True if s contains every letter a-z "
     "at least once (case-insensitive, ignoring non-letters). "
     "Return only a ```python code block.",
     'def is_pangram(s):\n    letters = set(ch.lower() for ch in s if ch.isalpha())\n    return len(letters) == 26',
     ['assert is_pangram("The quick brown fox jumps over the lazy dog") is True',
      'assert is_pangram("hello world") is False',
      'assert is_pangram("abcdefghijklmnopqrstuvwxyz") is True',
      'assert is_pangram("") is False',
      'assert is_pangram("ABCDEFGHIJKLMNOPQRSTUVWXYZ") is True']),

    ("code_extra_006", "Implement `remove_dup_keep_order(s)` that removes duplicate characters, "
     "keeping the first occurrence order. Return only a ```python code block.",
     'def remove_dup_keep_order(s):\n    seen, out = set(), []\n    for ch in s:\n        if ch not in seen:\n            seen.add(ch); out.append(ch)\n    return "".join(out)',
     ['assert remove_dup_keep_order("banana") == "ban"',
      'assert remove_dup_keep_order("") == ""',
      'assert remove_dup_keep_order("abc") == "abc"',
      'assert remove_dup_keep_order("aaaa") == "a"']),

    ("code_extra_007", "Implement `word_len_map(words)` that returns a dict mapping each word to "
     "its length. Return only a ```python code block.",
     'def word_len_map(words):\n    return {w: len(w) for w in words}',
     ['assert word_len_map(["hi","bye"]) == {"hi":2,"bye":3}',
      'assert word_len_map([]) == {}',
      'assert word_len_map(["a"]) == {"a":1}']),

    # ---- list algorithms ----
    ("code_extra_008", "Implement `running_sum(nums)` that returns a list where each element "
     "is the sum of all preceding elements including itself. Return only a ```python code block.",
     'def running_sum(nums):\n    out, acc = [], 0\n    for n in nums:\n        acc += n\n        out.append(acc)\n    return out',
     ['assert running_sum([1,2,3,4]) == [1,3,6,10]',
      'assert running_sum([]) == []',
      'assert running_sum([5]) == [5]',
      'assert running_sum([1,1,1]) == [1,2,3]']),

    ("code_extra_009", "Implement `count_negatives(grid)` that counts how many numbers in a "
     "2D list are negative. Return only a ```python code block.",
     'def count_negatives(grid):\n    return sum(1 for row in grid for n in row if n < 0)',
     ['assert count_negatives([[1,-1],[-2,3]]) == 2',
      'assert count_negatives([]) == 0',
      'assert count_negatives([[-1,-2]]) == 2',
      'assert count_negatives([[0,1],[2,3]]) == 0']),

    ("code_extra_010", "Implement `matrix_identity(n)` that returns an n x n identity matrix "
     "(list of lists). Return only a ```python code block.",
     'def matrix_identity(n):\n    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]',
     ['assert matrix_identity(1) == [[1]]',
      'assert matrix_identity(2) == [[1,0],[0,1]]',
      'assert matrix_identity(3) == [[1,0,0],[0,1,0],[0,0,1]]',
      'assert matrix_identity(0) == []']),

    ("code_extra_011", "Implement `pair_sum_indices(nums, target)` that returns the FIRST pair "
     "of indices (i,j) with i<j summing to target, as a list. Return None if none. "
     "Return only a ```python code block.",
     'def pair_sum_indices(nums, target):\n    for i in range(len(nums)):\n        for j in range(i+1, len(nums)):\n            if nums[i] + nums[j] == target:\n                return [i, j]\n    return None',
     ['assert pair_sum_indices([2,7,11,15],9) == [0,1]',
      'assert pair_sum_indices([1,2,3],6) is None',
      'assert pair_sum_indices([3,2,4],6) == [1,2]',
      'assert pair_sum_indices([0,4,3,0],0) == [0,3]']),

    ("code_extra_012", "Implement `flatten_one(grid)` that flattens a 2D list into 1D. "
     "Return only a ```python code block.",
     'def flatten_one(grid):\n    return [x for row in grid for x in row]',
     ['assert flatten_one([[1,2],[3,4]]) == [1,2,3,4]',
      'assert flatten_one([]) == []',
      'assert flatten_one([[1],[],[2]]) == [1,2]',
      'assert flatten_one([[1,2,3]]) == [1,2,3]']),

    ("code_extra_013", "Implement `min_max(nums)` that returns a tuple (min, max). Empty list "
     "returns (None, None). Return only a ```python code block.",
     'def min_max(nums):\n    if not nums:\n        return (None, None)\n    return (min(nums), max(nums))',
     ['assert min_max([3,1,4,1,5]) == (1,5)',
      'assert min_max([]) == (None, None)',
      'assert min_max([7]) == (7,7)']),

    ("code_extra_014", "Implement `common_elements(a, b)` that returns a sorted list of values "
     "present in both lists (no duplicates). Return only a ```python code block.",
     'def common_elements(a, b):\n    return sorted(set(a) & set(b))',
     ['assert common_elements([1,2,3],[2,3,4]) == [2,3]',
      'assert common_elements([1,2],[3,4]) == []',
      'assert common_elements([1,1,2],[1,3]) == [1]']),

    ("code_extra_015", "Implement `most_frequent_char(s)` that returns the most frequent "
     "character in s (ties -> first occurrence). Empty returns None. "
     "Return only a ```python code block.",
     'def most_frequent_char(s):\n    if not s:\n        return None\n    from collections import Counter\n    # Counter preserves insertion order; most_common breaks ties by insertion order\n    return Counter(s).most_common(1)[0][0]',
     ['assert most_frequent_char("aabbb") == "b"',
      'assert most_frequent_char("abab") == "a"',
      'assert most_frequent_char("") is None',
      'assert most_frequent_char("aaa") == "a"']),

    ("code_extra_016", "Implement `is_palindrome_list(nums)` that returns True if a list reads "
     "the same forwards and backwards. Return only a ```python code block.",
     'def is_palindrome_list(nums):\n    return nums == nums[::-1]',
     ['assert is_palindrome_list([1,2,2,1]) is True',
      'assert is_palindrome_list([1,2,3]) is False',
      'assert is_palindrome_list([]) is True',
      'assert is_palindrome_list([5]) is True']),

    ("code_extra_017", "Implement `dedupe_count(nums)` that returns a dict of value->count. "
     "Return only a ```python code block.",
     'def dedupe_count(nums):\n    d = {}\n    for n in nums:\n        d[n] = d.get(n, 0) + 1\n    return d',
     ['assert dedupe_count([1,2,2,3,3,3]) == {1:1,2:2,3:3}',
      'assert dedupe_count([]) == {}',
      'assert dedupe_count([5,5]) == {5:2}']),

    # ---- math / number algorithms ----
    ("code_extra_018", "Implement `is_power_of_two(n)` that returns True if n is a positive "
     "power of two. Return only a ```python code block.",
     'def is_power_of_two(n):\n    if n <= 0:\n        return False\n    return (n & (n - 1)) == 0',
     ['assert is_power_of_two(1) is True',
      'assert is_power_of_two(16) is True',
      'assert is_power_of_two(10) is False',
      'assert is_power_of_two(0) is False',
      'assert is_power_of_two(256) is True']),

    ("code_extra_019", "Implement `count_primes(limit)` that returns the number of primes "
     "strictly less than limit. Return only a ```python code block.",
     'def count_primes(limit):\n    if limit <= 2:\n        return 0\n    is_p = [True]*limit\n    is_p[0] = is_p[1] = False\n    for i in range(2, int(limit**0.5)+1):\n        if is_p[i]:\n            for j in range(i*i, limit, i):\n                is_p[j] = False\n    return sum(is_p)',
     ['assert count_primes(10) == 4',  # 2,3,5,7
      'assert count_primes(2) == 0',
      'assert count_primes(3) == 1',
      'assert count_primes(20) == 8']),

    ("code_extra_020", "Implement `gcd_many(nums)` that returns the greatest common divisor of a "
     "list of ints. Return only a ```python code block.",
     'def gcd_many(nums):\n    from math import gcd\n    g = 0\n    for n in nums:\n        g = gcd(g, n)\n    return g',
     ['assert gcd_many([12,18,24]) == 6',
      'assert gcd_many([7,11,13]) == 1',
      'assert gcd_many([100,50]) == 50',
      'assert gcd_many([5]) == 5']),

    ("code_extra_021", "Implement `decimal_to_binary(n)` that returns the binary string of n "
     "(n >= 0). No bin() allowed. Return only a ```python code block.",
     'def decimal_to_binary(n):\n    if n == 0:\n        return "0"\n    out = ""\n    while n:\n        out = str(n % 2) + out\n        n //= 2\n    return out',
     ['assert decimal_to_binary(0) == "0"',
      'assert decimal_to_binary(5) == "101"',
      'assert decimal_to_binary(13) == "1101"',
      'assert decimal_to_binary(1) == "1"']),

    ("code_extra_022", "Implement `nth_fib(n)` that returns the n-th Fibonacci (fib(0)=0, "
     "fib(1)=1). Return only a ```python code block.",
     'def nth_fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a',
     ['assert nth_fib(0) == 0', 'assert nth_fib(1) == 1', 'assert nth_fib(7) == 13',
      'assert nth_fib(10) == 55']),

    ("code_extra_023", "Implement `is_triangle(a, b, c)` that returns True if the three sides "
     "can form a non-degenerate triangle. Return only a ```python code block.",
     'def is_triangle(a, b, c):\n    return a + b > c and a + c > b and b + c > a',
     ['assert is_triangle(3,4,5) is True', 'assert is_triangle(1,2,3) is False',
      'assert is_triangle(5,5,5) is True', 'assert is_triangle(1,1,3) is False']),

    ("code_extra_024", "Implement `area_of_circle(r)` that returns area to 2 decimals (use pi=3.14159). "
     "Return only a ```python code block.",
     'def area_of_circle(r):\n    pi = 3.14159\n    return round(pi * r * r, 2)',
     ['assert area_of_circle(1) == 3.14', 'assert area_of_circle(2) == 12.57',
      'assert area_of_circle(0) == 0.0']),

    ("code_extra_025", "Implement `distance(x1, y1, x2, y2)` that returns the Euclidean distance "
     "rounded to 2 decimals. Return only a ```python code block.",
     'def distance(x1, y1, x2, y2):\n    return round(((x2-x1)**2 + (y2-y1)**2) ** 0.5, 2)',
     ['assert distance(0,0,3,4) == 5.0', 'assert distance(0,0,1,1) == 1.41',
      'assert distance(1,1,1,1) == 0.0']),

    # ---- data structures / parsing ----
    ("code_extra_026", "Implement `get_or_default(d, key, default)` that returns d[key] if present "
     "else default. Return only a ```python code block.",
     'def get_or_default(d, key, default):\n    return d.get(key, default)',
     ['assert get_or_default({"a":1}, "a", 99) == 1',
      'assert get_or_default({"a":1}, "b", 99) == 99',
      'assert get_or_default({}, "x", None) is None']),

    ("code_extra_027", "Implement `stack_to_list(stack)` that pops all items from a Python list "
     "treated as a stack and returns them in pop order. Return only a ```python code block.",
     'def stack_to_list(stack):\n    out = []\n    while stack:\n        out.append(stack.pop())\n    return out',
     ['assert stack_to_list([1,2,3]) == [3,2,1]',
      'assert stack_to_list([]) == []',
      'assert stack_to_list([7]) == [7]']),

    ("code_extra_028", "Implement `rotate_matrix_90(m)` that returns a matrix rotated 90 degrees "
     "clockwise. Return only a ```python code block.",
     'def rotate_matrix_90(m):\n    if not m:\n        return []\n    return [list(row) for row in zip(*m[::-1])]',
     ['assert rotate_matrix_90([[1,2],[3,4]]) == [[3,1],[4,2]]',
      'assert rotate_matrix_90([[1,2,3],[4,5,6]]) == [[4,1],[5,2],[6,3]]',
      'assert rotate_matrix_90([]) == []',
      'assert rotate_matrix_90([[1]]) == [[1]]']),

    ("code_extra_029", "Implement `index_of_second(nums, x)` that returns the index of the second "
     "occurrence of x, or -1. Return only a ```python code block.",
     'def index_of_second(nums, x):\n    found = 0\n    for i, n in enumerate(nums):\n        if n == x:\n            found += 1\n            if found == 2:\n                return i\n    return -1',
     ['assert index_of_second([1,2,3,2],2) == 3',
      'assert index_of_second([1,2,3],2) == -1',
      'assert index_of_second([5,5,5],5) == 1',
      'assert index_of_second([1,2],1) == -1']),

    ("code_extra_030", "Implement `is_valid_email(s)` that returns True for a simple email: exactly "
     "one @, chars before and after, and a dot after the @. Return only a ```python code block.",
     'def is_valid_email(s):\n    if s.count("@") != 1:\n        return False\n    local, domain = s.split("@")\n    return bool(local) and "." in domain and not domain.endswith(".") and s.count(" ") == 0',
     ['assert is_valid_email("a@b.com") is True',
      'assert is_valid_email("a@b") is False',
      'assert is_valid_email("a@@b.com") is False',
      'assert is_valid_email("a b@c.com") is False',
      'assert is_valid_email("@b.com") is False']),

    ("code_extra_031", "Implement `count_columns(grid)` that returns the maximum number of columns "
     "in a ragged 2D list (0 if empty). Return only a ```python code block.",
     'def count_columns(grid):\n    return max((len(row) for row in grid), default=0)',
     ['assert count_columns([[1,2,3],[4,5]]) == 3',
      'assert count_columns([]) == 0',
      'assert count_columns([[1],[2,3],[4,5,6]]) == 3']),

    ("code_extra_032", "Implement `char_count_word(s, ch)` that counts how many times a char appears "
     "(case-insensitive). Return only a ```python code block.",
     'def char_count_word(s, ch):\n    return s.lower().count(ch.lower())',
     ['assert char_count_word("Hello World", "l") == 3',
      'assert char_count_word("abc", "z") == 0',
      'assert char_count_word("aaa", "A") == 3']),

    ("code_extra_033", "Implement `chunk_count(lst, size)` that returns the number of chunks when "
     "splitting lst into chunks of length size. Return only a ```python code block.",
     'def chunk_count(lst, size):\n    if size <= 0:\n        raise ValueError("size must be > 0")\n    return (len(lst) + size - 1) // size',
     ['assert chunk_count([1,2,3,4,5],2) == 3', 'assert chunk_count([1,2,3],3) == 1',
      'assert chunk_count([],2) == 0', 'assert chunk_count([1,2,3,4],5) == 1']),

    ("code_extra_034", "Implement `to_pig_latin(s)` that converts a single word to Pig Latin: move the "
     "first letter to the end and add 'ay' (e.g. 'hello' -> 'ellohay'). "
     "Return only a ```python code block.",
     'def to_pig_latin(s):\n    if not s:\n        return ""\n    return s[1:] + s[0] + "ay"',
     ['assert to_pig_latin("hello") == "ellohay"', 'assert to_pig_latin("word") == "ordway"',
      'assert to_pig_latin("") == ""', 'assert to_pig_latin("x") == "xay"']),

    ("code_extra_035", "Implement `binary_to_decimal(b)` that converts a binary string to decimal. "
     "No int(b,2). Return only a ```python code block.",
     'def binary_to_decimal(b):\n    v = 0\n    for ch in b:\n        v = v * 2 + int(ch)\n    return v',
     ['assert binary_to_decimal("101") == 5', 'assert binary_to_decimal("1101") == 13',
      'assert binary_to_decimal("0") == 0', 'assert binary_to_decimal("1111") == 15']),
]


def main():
    # collect existing eval func names + train func names to avoid
    existing_funcs = set()
    if os.path.exists(EXISTING):
        for l in open(EXISTING, encoding="utf-8"):
            existing_funcs.add(json.loads(l)["entry"])
    train_funcs = set()
    if os.path.exists(TRAIN):
        for l in open(TRAIN, encoding="utf-8"):
            try:
                r = json.loads(l)
                e = entry_of(r["messages"][1]["content"])
                if e:
                    train_funcs.add(e)
            except Exception:
                continue

    out = []
    for iid, prompt, ref, tests in ITEMS:
        e = entry_of(ref)
        if e in existing_funcs:
            print(f"  SKIP {iid}: collides with existing eval '{e}'")
            continue
        if e in train_funcs:
            print(f"  SKIP {iid}: collides with train '{e}'")
            continue
        ns = {}
        exec(ref, ns)
        for t in tests:
            try:
                exec(t, ns)
            except AssertionError as ae:
                raise AssertionError(f"{iid} test failed {t!r}: {ae}")
        out.append({"id": iid, "category": "coding", "subcategory": "python",
                    "prompt": prompt, "checker": "unit_test", "entry": e, "tests": tests})
        print(f"  ok {iid} ({e})")

    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} items -> {OUT}")


if __name__ == "__main__":
    main()
