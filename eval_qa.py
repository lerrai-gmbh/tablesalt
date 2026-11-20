#!/usr/bin/env python3
"""
TableSalt benchmark — LLM-readiness evaluation (subtarget).

Idea: an extraction is usable downstream only if the questions whose
answers live in the table can be answered FROM THE EXTRACTION ALONE.
This measures information fidelity rather than structural similarity —
see README references. Evaluation is fully deterministic: no judge LLM.

Modes:

1) --auto   (default evaluation mode)
   Parses each team's submitted <table_id>.json (or .html), locates cells by row
   label (stub column) and column header, answers the lookup/aggregate
   questions programmatically, and grades against gt qa.json.

2) --make-prompts / --grade answers.json
   Optional variant for experiments with a reader LLM (not used for
   official scoring). --grade applies the same tolerant numeric matching
   (handles 1,234.56 / 1.234,56 / (123) negatives).

Score: fraction of questions answered correctly, reported per tier.
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path

NUM_RE = re.compile(r"\(?-?\d[\d.,\s]*\)?")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    return re.sub(r"\s+", " ", s).strip()


def parse_num(tok: str):
    t = tok.strip().rstrip(".,; \t")  # drop sentence punctuation
    if t.startswith("(") and not t.endswith(")"):
        t = t.lstrip("(")
    neg = t.startswith("(") and t.endswith(")")
    t = t.strip("()").replace(" ", "")
    # one comma + exactly 3 digits after it and no dot = thousands ("12,345"),
    # never a German decimal: the generator always prints de with 2 decimals
    if t.count(",") == 1 and (t.count(".") == 0 or t.rfind(",") > t.rfind(".")) \
            and not (t.count(".") == 0 and re.fullmatch(r"-?\d{1,3},\d{3}", t)):
        t = t.replace(".", "").replace(",", ".")
    else:
        t = t.replace(",", "")
    try:
        v = float(t)
        return -v if neg else v
    except ValueError:
        return None


def nums_in(text: str):
    out = []
    for m in NUM_RE.finditer(text or ""):
        v = parse_num(m.group(0))
        if v is not None:
            out.append(v)
    return out


def grade_one(qa: dict, answer: str) -> bool:
    if qa.get("answer_numeric") is not None:
        target = float(qa["answer_numeric"])
        tol = max(0.011, abs(target) * 1e-4)
        return any(abs(v - target) <= tol for v in nums_in(answer))
    return norm(str(qa["answer"])) in norm(answer)


def span(v) -> int:
    """rowspan/colspan -> int in [1, 100]; same rule as evaluate.span."""
    m = re.match(r"\s*(\d+)", str(v if v is not None else ""))
    return min(max(int(m.group(1)), 1), 100) if m else 1


def load_grid(pred_dir: Path, tid: str):
    """Grid of the submitted <tid>.json (preferred) or <tid>.html; None if
    missing or unparseable. Same file precedence as evaluate.load_prediction."""
    pj, ph = pred_dir / f"{tid}.json", pred_dir / f"{tid}.html"
    if pj.exists():
        try:
            grid = {}
            for c in json.loads(pj.read_bytes().decode("utf-8-sig", "replace")):
                for dr in range(span(c.get("rs", 1))):
                    for dc in range(span(c.get("cs", 1))):
                        grid[(int(c["r"]) + dr, int(c["c"]) + dc)] = \
                            norm(str(c.get("text", "")))
            return grid
        except Exception:
            return None
    if ph.exists():
        return grid_from_html(ph.read_bytes().decode("utf-8-sig", "replace"))
    return None


def grid_from_html(html_str):
    """(r, c) -> text with spans footprinted; None on parse failure."""
    from lxml import html as lxml_html
    try:
        root = lxml_html.fromstring(html_str)
        tables = root.xpath("//table")
        if not tables:
            return None
    except Exception:
        return None
    occupied, grid = {}, {}
    r = 0
    for tr in tables[0].xpath(".//tr"):
        c = 0
        for cell in tr.xpath("./td|./th"):
            while (r, c) in occupied:
                c += 1
            rs = span(cell.get("rowspan", 1))
            cs = span(cell.get("colspan", 1))
            text = norm(" ".join("".join(cell.itertext()).split()))
            for dr in range(rs):
                for dc in range(cs):
                    occupied[(r + dr, c + dc)] = True
                    grid[(r + dr, c + dc)] = text
            c += cs
        r += 1
    return grid


def auto_answer(qa, grid):
    """Answer a gt question from the submitted grid. Returns string or ''."""
    if grid is None:
        return ""
    n_rows = max((r for r, _ in grid), default=-1) + 1
    n_cols = max((c for _, c in grid), default=-1) + 1
    q = norm(qa["question"])
    import re as _re
    labels = _re.findall(r'"([^"]+)"', qa["question"])
    if qa["type"] == "lookup" and len(labels) == 2:
        row_lab, col_lab = norm(labels[0]), norm(labels[1])
        rows = [r for r in range(n_rows) if grid.get((r, 0), "") == row_lab]
        cols = [c for c in range(n_cols)
                if any(grid.get((r, c), "") == col_lab for r in range(min(3, n_rows)))]
        if rows and cols:
            return grid.get((rows[0], cols[0]), "")
        return ""
    if qa["type"] == "aggregate" and labels:
        col_lab = norm(labels[0])
        row_labs = [norm(x) for x in _re.findall(r"'([^']+)'", qa["question"])]
        cols = [c for c in range(n_cols)
                if any(grid.get((r, c), "") == col_lab for r in range(min(3, n_rows)))]
        if not cols:
            return ""
        total, found = 0.0, 0
        for rl in row_labs:
            rows = [r for r in range(n_rows) if grid.get((r, 0), "") == rl]
            if rows:
                vals = nums_in(grid.get((rows[0], cols[0]), ""))
                if vals:
                    total += vals[0]
                    found += 1
        return str(round(total, 2)) if found == len(row_labs) and found > 0 else ""
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True,
                    help="dir with <table_id>.json / .html (--make-prompts: .md / .html)")
    ap.add_argument("--make-prompts", default=None,
                    help="write prompts.jsonl to this path and exit")
    ap.add_argument("--grade", default=None,
                    help="answers.json from an external reader run")
    ap.add_argument("--auto", action="store_true",
                    help="answer questions programmatically from <table_id>.json/.html")
    args = ap.parse_args()

    gt_dir, pred_dir = Path(args.gt), Path(args.pred)
    manifest = json.loads((gt_dir / "manifest.json").read_text(encoding="utf-8"))

    if args.make_prompts:
        records = []
        for e in manifest:
            tid = e["id"]
            rep = None
            for ext in (".md", ".html"):
                p = pred_dir / f"{tid}{ext}"
                if p.exists():
                    rep = p.read_text(encoding="utf-8")
                    break
            qa = json.loads((gt_dir / tid / "gt" / "qa.json").read_text(encoding="utf-8"))
            for qi, q in enumerate(qa):
                records.append(dict(
                    key=f"{tid}/{qi}",
                    prompt=("You are given a table extracted from a document.\n"
                            "Answer the question using ONLY this table. "
                            "Reply with the value only.\n\n"
                            f"TABLE:\n{rep or '(no extraction submitted)'}\n\n"
                            f"QUESTION: {q['question']}")))
        Path(args.make_prompts).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")
        print(f"wrote {len(records)} prompts to {args.make_prompts}")
        return

    if args.auto:
        per_tier, total = {}, []
        for e in manifest:
            tid = e["id"]
            grid = load_grid(pred_dir, tid)
            qa = json.loads((gt_dir / tid / "gt" / "qa.json").read_text(encoding="utf-8"))
            for q in qa:
                ok = grade_one(q, auto_answer(q, grid))
                per_tier.setdefault(e["tier"], []).append(ok)
                total.append(ok)
        for t in sorted(per_tier):
            v = per_tier[t]
            print(f"{t}: QA accuracy = {sum(v)/len(v):.4f}  (n={len(v)})")
        print(f"\nOVERALL QA accuracy: {sum(total)/len(total):.4f} "
              f"({sum(total)}/{len(total)})")
        return

    if args.grade:
        answers = json.loads(Path(args.grade).read_text(encoding="utf-8"))
        per_tier, total = {}, []
        for e in manifest:
            tid = e["id"]
            qa = json.loads((gt_dir / tid / "gt" / "qa.json").read_text(encoding="utf-8"))
            for qi, q in enumerate(qa):
                ok = grade_one(q, answers.get(f"{tid}/{qi}", ""))
                per_tier.setdefault(e["tier"], []).append(ok)
                total.append(ok)
        for t in sorted(per_tier):
            v = per_tier[t]
            print(f"{t}: QA accuracy = {sum(v)/len(v):.4f}  (n={len(v)})")
        print(f"\nOVERALL QA accuracy: {sum(total)/len(total):.4f} "
              f"({sum(total)}/{len(total)})")
        return

    ap.error("pass --auto, --make-prompts or --grade")


if __name__ == "__main__":
    main()
