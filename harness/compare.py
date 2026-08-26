"""Compare multiple benchmark runs into a markdown table."""
import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

def load_result(tag):
    p = RESULTS / tag / "result.json"
    if not p.exists():
        raise FileNotFoundError(f"not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))

def main():
    ap = argparse.ArgumentParser(description="Compare benchmark runs")
    ap.add_argument("tags", nargs="+", help="result tags, e.g. base_before lora_r8 lora_r32")
    ap.add_argument("--out", default=None, help="output markdown path")
    args = ap.parse_args()

    rows = []
    for tag in args.tags:
        r = load_result(tag)
        agg = r.get("aggregate", {})
        rows.append({
            "tag": tag,
            "math": agg.get("math", 0)*100,
            "coding": agg.get("coding", 0)*100,
            "reasoning": agg.get("reasoning", 0)*100,
            "general": agg.get("general", 0)*100,
            "overall": agg.get("overall", 0)*100,
        })

    header = "| Model | Math | Coding | Reasoning | General | Avg |"
    sep = "|---|---|---|---|---|---|"
    lines = [header, sep]
    for row in rows:
        lines.append(f"| {row['tag']} | {row['math']:.0f} | {row['coding']:.0f} | {row['reasoning']:.0f} | {row['general']:.0f} | {row['overall']:.1f} |")
    out = "\n".join(lines)
    print(out)
    if args.out:
        Path(args.out).write_text(out, encoding="utf-8")
        print(f"\nwrote {args.out}")

if __name__ == "__main__":
    main()
