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
    ap.add_argument("tags", nargs="+", help="result tags, e.g. pandas_base pandas_r8")
    ap.add_argument("--out", default=None, help="output markdown path")
    args = ap.parse_args()

    # union of suite keys across runs (result.json "aggregate", minus "overall")
    rows = []
    suites = []
    for tag in args.tags:
        r = load_result(tag)
        agg = r.get("aggregate", {})
        for k in agg:
            if k != "overall" and k not in suites:
                suites.append(k)
        rows.append((tag, agg))
    suites.sort()

    header = "| Model | " + " | ".join(suites) + " | Avg |"
    sep = "|---" * (len(suites) + 2) + "|"
    lines = [header, sep]
    for tag, agg in rows:
        cells = " | ".join(f"{agg.get(s, 0)*100:.0f}" for s in suites)
        lines.append(f"| {tag} | {cells} | {agg.get('overall', 0)*100:.1f} |")
    out = "\n".join(lines)
    print(out)
    if args.out:
        Path(args.out).write_text(out, encoding="utf-8")
        print(f"\nwrote {args.out}")

if __name__ == "__main__":
    main()
