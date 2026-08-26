import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "eval"

items = []
for name in ["math.jsonl","reasoning.jsonl","coding.jsonl","general.jsonl"]:
    for line in open(EVAL_DIR/name, encoding="utf-8"):
        obj=json.loads(line)
        obj["_suite"]=name.replace(".jsonl","")
        items.append(obj)

coding_refs = {
 "coding_001": 'def fizzbuzz(n):\n    out=[]\n    for i in range(1,n+1):\n        if i%15==0: out.append("FizzBuzz")\n        elif i%3==0: out.append("Fizz")\n        elif i%5==0: out.append("Buzz")\n        else: out.append(str(i))\n    return out',
 "coding_002": 'def is_palindrome(s):\n    import re\n    t="".join(ch.lower() for ch in s if ch.isalnum())\n    return t==t[::-1]',
 "coding_003": 'def fib(n):\n    a,b=0,1\n    for _ in range(n): a,b=b,a+b\n    return a',
 "coding_004": 'def two_sum(nums,target):\n    d={}\n    for i,x in enumerate(nums):\n        if target-x in d: return sorted([d[target-x],i])\n        d[x]=i',
 "coding_005": 'def reverse_words(s):\n    return " ".join(s.split()[::-1])',
 "coding_006": 'def count_vowels(s):\n    return sum(1 for ch in s.lower() if ch in "aeiou")',
 "coding_007": 'def flatten(lst):\n    out=[]\n    for sub in lst:\n        out.extend(sub)\n    return out',
 "coding_008": 'def binary_search(arr,x):\n    lo,hi=0,len(arr)-1\n    while lo<=hi:\n        m=(lo+hi)//2\n        if arr[m]==x: return m\n        elif arr[m]<x: lo=m+1\n        else: hi=m-1\n    return -1',
 "coding_009": 'def gcd(a,b):\n    import math\n    return math.gcd(a,b)',
 "coding_010": 'def is_prime(n):\n    if n<2: return False\n    if n%2==0: return n==2\n    i=3\n    while i*i<=n:\n        if n%i==0: return False\n        i+=2\n    return True',
 "coding_011": 'def merge_sorted(a,b):\n    i=j=0; out=[]\n    while i<len(a) and j<len(b):\n        if a[i]<=b[j]: out.append(a[i]); i+=1\n        else: out.append(b[j]); j+=1\n    out.extend(a[i:]); out.extend(b[j:]); return out',
 "coding_012": 'def remove_duplicates(lst):\n    seen=set(); out=[]\n    for x in lst:\n        if x not in seen:\n            seen.add(x); out.append(x)\n    return out',
 "coding_013": 'def longest_word(s):\n    w=s.split()\n    return max(w, key=len) if w else ""',
 "coding_014": 'def caesar(s,shift):\n    out=""\n    for ch in s:\n        if "a"<=ch<="z": out+=chr((ord(ch)-97+shift)%26+97)\n        elif "A"<=ch<="Z": out+=chr((ord(ch)-65+shift)%26+65)\n        else: out+=ch\n    return out',
 "coding_015": 'def is_anagram(a,b):\n    fa="".join(a.lower().split())\n    fb="".join(b.lower().split())\n    return sorted(fa)==sorted(fb)',
 "coding_016": 'def transpose(m):\n    if not m or not m[0]: return []\n    return [list(row) for row in zip(*m)]',
 "coding_017": 'def prime_factors(n):\n    if n<2: return []\n    out=[]; d=2\n    while d*d<=n:\n        while n%d==0: out.append(d); n//=d\n        d+=1 if d==2 else 2\n    if n>1: out.append(n)\n    return out',
 "coding_018": 'def balanced(s):\n    st=[]; m={")":"(","]":"[","}":"{"}\n    for ch in s:\n        if ch in "([{": st.append(ch)\n        else:\n            if not st or st[-1]!=m[ch]: return False\n            st.pop()\n    return not st',
 "coding_019": 'def rle(s):\n    if not s: return []\n    out=[]; cur=s[0]; cnt=1\n    for ch in s[1:]:\n        if ch==cur: cnt+=1\n        else: out.append([cur,cnt]); cur=ch; cnt=1\n    out.append([cur,cnt]); return out',
 "coding_020": 'def pascal_row(n):\n    row=[1]\n    for k in range(1,n+1): row.append(row[-1]*(n-k+1)//k)\n    return row',
 "coding_021": 'def rotate_list(lst,k):\n    if not lst: return []\n    k%=len(lst)\n    return lst[-k:]+lst[:-k] if k else lst[:]',
 "coding_022": 'def second_largest(lst):\n    u=sorted(set(lst), reverse=True)\n    return u[1] if len(u)>=2 else None',
 "coding_023": 'def is_valid_date(s):\n    import re, datetime\n    if not re.match(r"^\\d{4}-\\d{2}-\\d{2}$", s): return False\n    try: datetime.date.fromisoformat(s); return True\n    except ValueError: return False',
 "coding_024": 'def chunk(lst,size):\n    if size<=0: raise ValueError("size must be >0")\n    return [lst[i:i+size] for i in range(0,len(lst),size)]',
}

constraint_perfect = {
 "general_025": "Waves crash softly against golden sands under endless blue sky",
 "general_026": '{"name": "Ada", "year": 2020}',
 "general_027": "COMPUTERS PROCESS INFORMATION USING ELECTRONIC CIRCUITS EFFICIENTLY TODAY",
 "general_028": "- 2\n- 3\n- 5",
 "general_029": "Cats love to nap in warm sunlight peacefully.",
 "general_030": "The sky is blue banana",
 "general_031": "56",
 "general_032": "Tea is soothing. It helps me relax.",
 "general_033": "Tôi khỏe, cảm ơn bạn rất nhiều nhé!",
 "general_034": "cat, dog, fish, bird, lion",
 "general_035": '{"lines": ["Morning light breaks", "Steam rises softly", "Tea warms hands"]}',
 "general_036": "4",
}

def perfect_response(item):
    c=item["checker"]
    if c=="numeric":
        return f"The answer is \\boxed{{{item['answer']}}}"
    if c=="exact":
        return item["answer"]
    if c=="choice":
        return f"Answer: {item['answer']}"
    if c=="constraint":
        return constraint_perfect.get(item["id"], "dummy")
    if c=="unit_test":
        ref=coding_refs.get(item["id"], "def dummy(): pass")
        return f"```python\n{ref}\n```"
    return ""

def bad_response(item):
    c=item["checker"]
    if c=="numeric":
        try:
            v=float(item["answer"].replace(",",""))
            return f"The answer is {v+1}"
        except: return "I don't know"
    if c=="exact":
        return "I don't know"
    if c=="choice":
        opts=["A","B","C","D"]
        wrong=[o for o in opts if o!=item["answer"]][0]
        return f"Answer: {wrong}"
    if c=="constraint":
        return "hello world"
    if c=="unit_test":
        return "```python\ndef dummy():\n    return None\n```"
    return "bad"

for tag, func in [("dummy_perfect", perfect_response), ("dummy_bad", bad_response)]:
    out_dir = ROOT / "results" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "outputs.jsonl"
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for it in items:
            resp = func(it)
            f.write(json.dumps({"id": it["id"], "suite": it["_suite"], "prompt": it["prompt"], "response": resp}, ensure_ascii=False)+"\n")
    print(f"wrote {out_path} ({len(items)} items)")
