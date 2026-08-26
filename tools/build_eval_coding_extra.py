"""Append high-quality algorithmic code eval items to coding_extra.jsonl.

Adds ~25 substantive algorithm problems (not one-liners) with 4-5 tests each.
References are executed against tests before writing; a failed test aborts.
Merges with existing coding_extra.jsonl; total intended ~60.
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "eval", "coding_extra.jsonl")

ITEMS = [
    ("code_extra_036", "Implement `longest_common_prefix(strs)` that returns the longest common "
     "prefix of a list of strings. Empty list -> ''. Return only a ```python code block.",
     'def longest_common_prefix(strs):\n    if not strs:\n        return ""\n    pref = strs[0]\n    for s in strs[1:]:\n        while not s.startswith(pref):\n            pref = pref[:-1]\n            if not pref:\n                return ""\n    return pref',
     ['assert longest_common_prefix(["flower","flow","flight"]) == "fl"',
      'assert longest_common_prefix(["dog","racecar","car"]) == ""',
      'assert longest_common_prefix([""]) == ""',
      'assert longest_common_prefix([]) == ""',
      'assert longest_common_prefix(["abc","abc","abc"]) == "abc"']),

    ("code_extra_037", "Implement `first_unique_char(s)` that returns the index of the first "
     "non-repeating character, or -1. Return only a ```python code block.",
     'def first_unique_char(s):\n    from collections import Counter\n    c = Counter(s)\n    for i, ch in enumerate(s):\n        if c[ch] == 1:\n            return i\n    return -1',
     ['assert first_unique_char("leetcode") == 0', 'assert first_unique_char("loveleetcode") == 2',
      'assert first_unique_char("aabb") == -1', 'assert first_unique_char("") == -1']),

    ("code_extra_038", "Implement `max_profit(prices)` that returns the maximum profit from one "
     "buy-then-sell (0 if none). Return only a ```python code block.",
     'def max_profit(prices):\n    best, cheap = 0, float("inf")\n    for p in prices:\n        cheap = min(cheap, p)\n        best = max(best, p - cheap)\n    return best',
     ['assert max_profit([7,1,5,3,6,4]) == 5', 'assert max_profit([7,6,4,3,1]) == 0',
      'assert max_profit([1,2]) == 1', 'assert max_profit([2,1]) == 0']),

    ("code_extra_039", "Implement `group_anagrams(words)` that returns a list of lists where each "
     "inner list groups anagrams together (order not important within group). "
     "Return only a ```python code block.",
     'def group_anagrams(words):\n    d = {}\n    for w in words:\n        k = "".join(sorted(w))\n        d.setdefault(k, []).append(w)\n    return list(d.values())',
     ['def norm(groups):\n        return sorted("".join(sorted(g[0])) for g in groups if g)\n    ',
      'assert sorted(sorted(g) for g in group_anagrams(["eat","tea","ate","tan"])) == [["ate","eat","tea"],["tan"]]',
      'assert sorted(sorted(g) for g in group_anagrams(["eat","tea","ate"])) == [["ate","eat","tea"]]',
      'assert group_anagrams([]) == []',
      'assert sorted(sorted(g) for g in group_anagrams(["a","b"])) == [["a"],["b"]]']),

    ("code_extra_040", "Implement `move_zeroes(nums)` that moves all zeros to the end in-place, "
     "keeping the relative order of non-zero elements. Returns the same list. "
     "Return only a ```python code block.",
     'def move_zeroes(nums):\n    pos = 0\n    for n in nums:\n        if n != 0:\n            nums[pos] = n\n            pos += 1\n    for i in range(pos, len(nums)):\n        nums[i] = 0\n    return nums',
     ['assert move_zeroes([0,1,0,3,12]) == [1,3,12,0,0]',
      'assert move_zeroes([0,0,1]) == [1,0,0]',
      'assert move_zeroes([1,2,3]) == [1,2,3]',
      'assert move_zeroes([0]) == [0]']),

    ("code_extra_041", "Implement `find_missing(nums)` that returns the single missing number in "
     "[0..n] given nums of length n containing all but one. Return only a ```python code block.",
     'def find_missing(nums):\n    n = len(nums)\n    return n*(n+1)//2 - sum(nums)',
     ['assert find_missing([3,0,1]) == 2', 'assert find_missing([0,1]) == 2',
      'assert find_missing([9,6,4,2,3,5,7,0,1]) == 8', 'assert find_missing([1]) == 0']),

    ("code_extra_042", "Implement `is_subsequence(s, t)` that returns True if s is a subsequence "
     "of t (chars appear in order, not necessarily contiguous). Return only a ```python code block.",
     'def is_subsequence(s, t):\n    i = 0\n    for ch in t:\n        if i < len(s) and s[i] == ch:\n            i += 1\n    return i == len(s)',
     ['assert is_subsequence("abc", "ahbgdc") is True',
      'assert is_subsequence("axc", "ahbgdc") is False',
      'assert is_subsequence("", "abc") is True',
      'assert is_subsequence("a", "") is False',
      'assert is_subsequence("abc", "abc") is True']),

    ("code_extra_043", "Implement `plus_one(digits)` that adds one to a list of digits "
     "(most significant first), handling carry, e.g. [9,9] -> [1,0,0]. "
     "Return only a ```python code block.",
     'def plus_one(digits):\n    d = digits[:]\n    for i in range(len(d)-1, -1, -1):\n        if d[i] < 9:\n            d[i] += 1\n            return d\n        d[i] = 0\n    return [1] + d',
     ['assert plus_one([1,2,3]) == [1,2,4]', 'assert plus_one([9,9]) == [1,0,0]',
      'assert plus_one([9]) == [1,0]', 'assert plus_one([0]) == [1]']),

    ("code_extra_044", "Implement `length_of_longest_substring(s)` that returns the length of the "
     "longest substring without repeating characters. Return only a ```python code block.",
     'def length_of_longest_substring(s):\n    last = {}\n    start = best = 0\n    for i, ch in enumerate(s):\n        if ch in last and last[ch] >= start:\n            start = last[ch] + 1\n        last[ch] = i\n        best = max(best, i - start + 1)\n    return best',
     ['assert length_of_longest_substring("abcabcbb") == 3',
      'assert length_of_longest_substring("bbbbb") == 1',
      'assert length_of_longest_substring("pwwkew") == 3',
      'assert length_of_longest_substring("") == 0',
      'assert length_of_longest_substring("aab") == 2']),

    ("code_extra_045", "Implement `remove_element(nums, val)` that removes all occurrences of val "
     "in-place and returns the new length. Return only a ```python code block.",
     'def remove_element(nums, val):\n    k = 0\n    for n in nums:\n        if n != val:\n            nums[k] = n\n            k += 1\n    return k',
     ['assert remove_element([3,2,2,3], 3) == 2',
      'assert remove_element([0,1,2,2,3,0,4,2], 2) == 5',
      'assert remove_element([1,1,1], 1) == 0',
      'assert remove_element([], 5) == 0']),

    ("code_extra_046", "Implement `majority_element(nums)` that returns the element appearing more "
     "than n//2 times (guaranteed to exist). Return only a ```python code block.",
     'def majority_element(nums):\n    cand, cnt = None, 0\n    for n in nums:\n        if cnt == 0:\n            cand, cnt = n, 1\n        elif n == cand:\n            cnt += 1\n        else:\n            cnt -= 1\n    return cand',
     ['assert majority_element([3,2,3]) == 3', 'assert majority_element([2,2,1,1,2,2]) == 2',
      'assert majority_element([1]) == 1']),

    ("code_extra_047", "Implement `merge_intervals(intervals)` that merges overlapping intervals "
     "and returns a sorted list of [start,end]. Return only a ```python code block.",
     'def merge_intervals(intervals):\n    if not intervals:\n        return []\n    intervals = sorted(intervals)\n    out = [list(intervals[0])]\n    for s, e in intervals[1:]:\n        if s <= out[-1][1]:\n            out[-1][1] = max(out[-1][1], e)\n        else:\n            out.append([s, e])\n    return out',
     ['assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]',
      'assert merge_intervals([[1,4],[4,5]]) == [[1,5]]',
      'assert merge_intervals([]) == []',
      'assert merge_intervals([[1,3]]) == [[1,3]]']),

    ("code_extra_048", "Implement `two_sum_closest(nums, target)` that returns the pair sum closest "
     "to target (absolute difference). Return only a ```python code block.",
     'def two_sum_closest(nums, target):\n    nums = sorted(nums)\n    lo, hi = 0, len(nums)-1\n    best = None\n    while lo < hi:\n        s = nums[lo] + nums[hi]\n        if best is None or abs(s - target) < abs(best - target):\n            best = s\n        if s < target:\n            lo += 1\n        else:\n            hi -= 1\n    return best',
     ['assert two_sum_closest([1,2,3,4], 7) == 7', 'assert two_sum_closest([-1,2,1,-4], 1) == 1',
      'assert two_sum_closest([1,1,1], 3) == 2', 'assert two_sum_closest([2,5], 3) == 7']),

    ("code_extra_049", "Implement `is_rotation(s, goal)` that returns True if goal is a rotation of s "
     "(s can be shifted to equal goal). Return only a ```python code block.",
     'def is_rotation(s, goal):\n    return len(s) == len(goal) and goal in (s + s)',
     ['assert is_rotation("abcde", "cdeab") is True', 'assert is_rotation("abcde", "abced") is False',
      'assert is_rotation("", "") is True', 'assert is_rotation("a", "a") is True']),

    ("code_extra_050", "Implement `str_str(haystack, needle)` that returns the index of the first "
     "occurrence of needle in haystack, or -1. Return only a ```python code block.",
     'def str_str(haystack, needle):\n    if needle == "":\n        return 0\n    return haystack.find(needle)',
     ['assert str_str("hello", "ll") == 2', 'assert str_str("aaaaa", "bba") == -1',
      'assert str_str("", "") == 0', 'assert str_str("abc", "abc") == 0']),

    ("code_extra_051", "Implement `roman_to_int(s)` that converts a Roman numeral string to int. "
     "Use I=1,V=5,X=10,L=50,C=100,D=500,M=1000. Return only a ```python code block.",
     'def roman_to_int(s):\n    v = {"I":1,"V":5,"X":10,"L":50,"C":100,"D":500,"M":1000}\n    total, prev = 0, 0\n    for ch in reversed(s):\n        cur = v[ch]\n        if cur < prev:\n            total -= cur\n        else:\n            total += cur\n        prev = cur\n    return total',
     ['assert roman_to_int("III") == 3', 'assert roman_to_int("IV") == 4',
      'assert roman_to_int("MCMXCIV") == 1994', 'assert roman_to_int("IX") == 9']),

    ("code_extra_052", "Implement `subarray_sum(nums, target)` that counts the number of contiguous "
     "subarrays summing to target. Return only a ```python code block.",
     'def subarray_sum(nums, target):\n    from collections import Counter\n    pre, count, cur = Counter({0:1}), 0, 0\n    for n in nums:\n        cur += n\n        count += pre[cur - target]\n        pre[cur] += 1\n    return count',
     ['assert subarray_sum([1,1,1], 2) == 2', 'assert subarray_sum([1,2,3], 3) == 2',
      'assert subarray_sum([1], 0) == 0', 'assert subarray_sum([-1,-1,1], 0) == 1']),

    ("code_extra_053", "Implement `is_palindrome_range(s)` that returns True if s is a palindrome "
     "after removing at most one character. Return only a ```python code block.",
     'def is_palindrome_range(s):\n    def isp(i, j):\n        while i < j:\n            if s[i] != s[j]:\n                return False\n            i += 1; j -= 1\n        return True\n    i, j = 0, len(s)-1\n    while i < j:\n        if s[i] != s[j]:\n            return isp(i+1, j) or isp(i, j-1)\n        i += 1; j -= 1\n    return True',
     ['assert is_palindrome_range("aba") is True', 'assert is_palindrome_range("abca") is True',
      'assert is_palindrome_range("abc") is False', 'assert is_palindrome_range("") is True']),

    ("code_extra_054", "Implement `array_product(nums)` that returns a list where each element is the "
     "product of all elements except itself (no division). Return only a ```python code block.",
     'def array_product(nums):\n    n = len(nums)\n    out = [1]*n\n    pref = 1\n    for i in range(n):\n        out[i] = pref\n        pref *= nums[i]\n    suff = 1\n    for i in range(n-1, -1, -1):\n        out[i] *= suff\n        suff *= nums[i]\n    return out',
     ['assert array_product([1,2,3,4]) == [24,12,8,6]',
      'assert array_product([2,3]) == [3,2]', 'assert array_product([1,1]) == [1,1]',
      'assert array_product([]) == []']),

    ("code_extra_055", "Implement `num_of_ones(n)` that counts the set bits in a non-negative int n. "
     "Return only a ```python code block.",
     'def num_of_ones(n):\n    return bin(n).count("1")',
     ['assert num_of_ones(5) == 2', 'assert num_of_ones(0) == 0', 'assert num_of_ones(7) == 3',
      'assert num_of_ones(1023) == 10']),

    ("code_extra_056", "Implement `is_palindrome_permutation(s)` that returns True if some permutation "
     "of s can be a palindrome (at most one odd-count char). Return only a ```python code block.",
     'def is_palindrome_permutation(s):\n    from collections import Counter\n    odd = sum(1 for v in Counter(s).values() if v % 2)\n    return odd <= 1',
     ['assert is_palindrome_permutation("tactcoa") is True', 'assert is_palindrome_permutation("code") is False',
      'assert is_palindrome_permutation("aabbc") is True', 'assert is_palindrome_permutation("aaa") is True']),

    ("code_extra_057", "Implement `next_even(n)` that returns the next even number strictly greater "
     "than n. Return only a ```python code block.",
     'def next_even(n):\n    if n % 2 == 0:\n        return n + 2\n    return n + 1',
     ['assert next_even(3) == 4', 'assert next_even(4) == 6', 'assert next_even(-1) == 0',
      'assert next_even(0) == 2']),

    ("code_extra_058", "Implement `is_valid_palindrome_pair(s)` that returns True if s, after removing "
     "non-alphanumeric chars and lowercasing, is a palindrome. Return only a ```python code block.",
     'def is_valid_palindrome_pair(s):\n    import re\n    t = re.sub(r"[^a-z0-9]", "", s.lower())\n    return t == t[::-1]',
     ['assert is_valid_palindrome_pair("A man a plan a canal Panama") is True',
      'assert is_valid_palindrome_pair("race a car") is False',
      'assert is_valid_palindrome_pair("") is True',
      'assert is_valid_palindrome_pair("ab_a") is True']),

    ("code_extra_059", "Implement `sum_two_smallest(nums)` that sums the two smallest numbers in a "
     "list (len>=2). Return only a ```python code block.",
     'def sum_two_smallest(nums):\n    a, b = sorted(nums)[:2]\n    return a + b',
     ['assert sum_two_smallest([3,1,2]) == 3', 'assert sum_two_smallest([1,1]) == 2',
      'assert sum_two_smallest([5,9,1,7]) == 6']),

    ("code_extra_060", "Implement `add_digits(n)` that repeatedly sums the digits until a single digit "
     "remains (digital root). Return only a ```python code block.",
     'def add_digits(n):\n    if n == 0:\n        return 0\n    if n % 9 == 0:\n        return 9\n    return n % 9',
     ['assert add_digits(38) == 2', 'assert add_digits(0) == 0', 'assert add_digits(9) == 9',
      'assert add_digits(12345) == 6']),
]


def entry_of(code):
    m = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", code)
    return m.group(1).lower() if m else None


def main():
    existing = []
    if os.path.exists(OUT):
        existing = [json.loads(l) for l in open(OUT, encoding="utf-8")]
    existing_ids = {r["id"] for r in existing}
    existing_entries = {r["entry"] for r in existing}
    out = list(existing)
    added = 0
    for iid, prompt, ref, tests in ITEMS:
        if iid in existing_ids:
            continue
        e = entry_of(ref)
        # only skip if the SAME id/entry already present; allow overlap otherwise
        ns = {}
        exec(ref, ns)
        for t in tests:
            try:
                exec(t, ns)
            except AssertionError as ae:
                raise AssertionError(f"{iid} test failed {t!r}: {ae}")
        out.append({"id": iid, "category": "coding", "subcategory": "python",
                    "prompt": prompt, "checker": "unit_test", "entry": e, "tests": tests})
        added += 1
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"added {added} | total {len(out)} -> {OUT}")


if __name__ == "__main__":
    main()
