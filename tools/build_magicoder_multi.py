"""Pull additional Magicoder langs (Java, JavaScript, TypeScript) 8k.

Reuses data/raw/magicoder_oss.jsonl (already downloaded 193MB).
Selects those langs, short (<4000 tok), and excludes eval function names.
"""
import json, os, random, re
SEED = 20260912
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "magicoder_oss.jsonl")
EVAL = os.path.join(ROOT, "data", "eval", "coding.jsonl")
OUT = os.path.join(ROOT, "data", "train", "magicoder_java_js_ts_8000.jsonl")
CHARS=3.5
MAX=4000
N=8000
def load_eval():
    funcs=set()
    for l in open(EVAL,encoding="utf-8"):
        funcs.add(json.loads(l)["entry"])
    kws=set()
    for e in funcs:
        for w in re.split(r"[_]+", e):
            if len(w)>=3: kws.add(w.lower())
    return funcs, kws

def main():
    funcs, kws = load_eval()
    rng=random.Random(SEED)
    pool=[]
    with open(RAW,encoding="utf-8") as f:
        for line in f:
            try: r=json.loads(line)
            except: continue
            if r.get("lang") not in ("java","javascript","typescript"): continue
            p,s=r.get("problem",""),r.get("solution","")
            if len(s)<40 or len(p)>3000: continue
            if (len(p)+len(s))/CHARS>MAX: continue
            names=re.findall(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", s)
            # for JS/TS also check function keyword
            if not names:
                names=re.findall(r"function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", s)
            e=(names[0].lower() if names else "")
            if e in funcs: continue
            if e and any(kw in e for kw in kws if len(kw)>=3): continue
            pool.append(r)
    rng.shuffle(pool)
    chosen=pool[:N]
    seen=set(); out=[]
    for r in chosen:
        k=" ".join(r["problem"].lower().split())
        if k in seen: continue
        seen.add(k)
        out.append({"messages":[{"role":"user","content":r["problem"]},{"role":"assistant","content":r["solution"]}]})
    with open(OUT,"w",encoding="utf-8",newline="\n") as f:
        for r in out: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    print(f"java/js/ts eligible: {len(pool)} -> chose {len(out)} -> {OUT}")

if __name__=="__main__": main()
