#!/usr/bin/env python3
"""
Full-workflow self-test, run before handing the benchmark to anyone.

Simulates many teams producing different output types and quality levels
(perfect JSON, perfect HTML, typos, dropped rows, lost spans, swapped
columns, extra cells, malformed files, empty, missing), scores them all,
and asserts the invariants a trustworthy evaluator must satisfy:

  I1  perfect submissions (JSON and HTML) score exactly 1.0
  I2  every perturbation scores strictly below perfect
  I3  heavier damage never scores above lighter damage of the same kind
  I4  unparseable input (malformed JSON, empty, no table, missing)
      scores exactly 0 and never crashes the evaluator; malformed but
      recoverable HTML earns only its recovered structure (near 0)
  I5  JSON and HTML encodings of the same prediction score identically

Usage: python selftest_workflow.py --gt data/dev
Exit code 0 = all invariants hold.
"""

import argparse
import copy
import importlib.util
import json
import random
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev, qa = load("evaluate"), load("eval_qa")
rng = random.Random(1234)


# -- perturbation generators (input: GT cell list, output: cell list/raw) ----

def p_typos(cells, rate):
    out = copy.deepcopy(cells)
    for c in out:
        if c["text"] and rng.random() < rate:
            t = list(c["text"])
            i = rng.randrange(len(t))
            t[i] = rng.choice("0189XO#")
            c["text"] = "".join(t)
    return out


def p_drop_rows(cells, n):
    max_r = max(c["r"] for c in cells)
    return [c for c in cells if c["r"] <= max_r - n]


def p_lose_spans(cells):
    out = copy.deepcopy(cells)
    for c in out:
        c["rs"], c["cs"] = 1, 1
    return out


def p_swap_cols(cells):
    out = copy.deepcopy(cells)
    a = [c for c in out if c["c"] == 1 and c["cs"] == 1]
    b = [c for c in out if c["c"] == 2 and c["cs"] == 1]
    for x in a:
        x["c"] = 2
    for x in b:
        x["c"] = 1
    return out


def p_extra_cells(cells, n):
    out = copy.deepcopy(cells)
    max_c = max(c["c"] + c["cs"] for c in cells)
    for i in range(n):
        out.append(dict(text=f"ghost{i}", r=i, c=max_c + 1, rs=1, cs=1,
                        header=False))
    return out


SUBMISSIONS = {
    "perfect_json":   ("json",  lambda c: c),
    "perfect_html":   ("html",  lambda c: c),
    "typos_10":       ("json",  lambda c: p_typos(c, 0.10)),
    "typos_40":       ("json",  lambda c: p_typos(c, 0.40)),
    "drop_1_row":     ("json",  lambda c: p_drop_rows(c, 1)),
    "drop_3_rows":    ("json",  lambda c: p_drop_rows(c, 3)),
    "lost_spans":     ("json",  lambda c: p_lose_spans(c)),
    "swapped_cols":   ("json",  lambda c: p_swap_cols(c)),
    "extra_cells":    ("json",  lambda c: p_extra_cells(c, 2)),
    "malformed_json": ("raw_json", '{"this is": broken'),
    "malformed_html": ("raw_html", "<table><tr><td>unclosed"),
    "not_a_table":    ("raw_html", "<div>hello</div>"),
    "empty_file":     ("raw_json", ""),
    "missing":        ("none", None),
}


def build_submissions(gt_dir, work):
    manifest = json.loads((gt_dir / "manifest.json").read_text(encoding="utf-8"))
    for name, (kind, gen) in SUBMISSIONS.items():
        d = work / name
        d.mkdir(parents=True, exist_ok=True)
        for e in manifest:
            tid = e["id"]
            if kind == "none":
                continue
            if kind.startswith("raw_"):
                ext = kind.split("_")[1]
                (d / f"{tid}.{ext}").write_text(gen, encoding="utf-8")
                continue
            cells = json.loads((gt_dir / tid / "gt" / "cells.json").read_text(encoding="utf-8"))
            cells = gen(cells)
            if kind == "json":
                (d / f"{tid}.json").write_text(
                    json.dumps(cells, ensure_ascii=False), encoding="utf-8")
            else:
                (d / f"{tid}.html").write_text(ev.cells_json_to_html(cells), encoding="utf-8")
    return manifest


def score(gt_dir, manifest, pred_dir):
    teds_all, f1_all, qa_ok = [], [], []
    for e in manifest:
        tid = e["id"]
        gt_html = (gt_dir / tid / "gt" / "table.html").read_text(encoding="utf-8")
        pred_html = ev.load_prediction(pred_dir, tid)
        teds_all.append(ev.teds(gt_html, pred_html) if pred_html else 0.0)
        f1_all.append(ev.cell_f1(gt_html, pred_html) if pred_html else 0.0)
        grid = qa.load_grid(pred_dir, tid)
        for q in json.loads((gt_dir / tid / "gt" / "qa.json").read_text(encoding="utf-8")):
            qa_ok.append(qa.grade_one(q, qa.auto_answer(q, grid)))
    n = len(teds_all)
    return (round(sum(teds_all) / n, 4), round(sum(f1_all) / n, 4),
            round(sum(qa_ok) / len(qa_ok), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="data/dev")
    args = ap.parse_args()
    gt_dir = Path(args.gt)
    work = Path("/tmp/selftest_submissions")
    shutil.rmtree(work, ignore_errors=True)

    manifest = build_submissions(gt_dir, work)
    results = {}
    print(f"{'submission':16s} {'TEDS':>7s} {'CellF1':>7s} {'QA':>7s}")
    for name in SUBMISSIONS:
        try:
            results[name] = score(gt_dir, manifest, work / name)
        except Exception as exc:
            print(f"{name:16s} CRASHED: {exc}")
            results[name] = None
        else:
            t, f1, q = results[name]
            print(f"{name:16s} {t:7.4f} {f1:7.4f} {q:7.4f}")

    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    # I4 first: nothing crashed
    for name, r in results.items():
        check(r is not None, f"I4 CRASH: {name}")
    if not failures:
        t = {k: v[0] for k, v in results.items()}
        # I1
        check(t["perfect_json"] == 1.0, "I1: perfect_json != 1.0")
        check(t["perfect_html"] == 1.0, "I1: perfect_html != 1.0")
        check(results["perfect_json"][2] == 1.0, "I1: perfect QA != 1.0")
        # I2
        for name in ("typos_10", "typos_40", "drop_1_row", "drop_3_rows",
                     "lost_spans", "swapped_cols", "extra_cells"):
            check(t[name] < 1.0, f"I2: {name} not < 1.0")
        # I3 monotonicity
        check(t["typos_40"] < t["typos_10"], "I3: typos not monotone")
        check(t["drop_3_rows"] < t["drop_1_row"], "I3: drops not monotone")
        # I4 garbage: unparseable input scores exactly 0; malformed-but-
        # recoverable HTML may earn only its recovered structure (near 0)
        for name in ("malformed_json", "not_a_table", "empty_file", "missing"):
            check(t[name] == 0.0, f"I4: {name} != 0 (got {t[name]})")
        check(t["malformed_html"] < 0.05,
              f"I4: malformed_html too high ({t['malformed_html']})")
        # I5 encoding equivalence
        check(results["perfect_json"] == results["perfect_html"],
              "I5: JSON vs HTML scores differ")

    print()
    if failures:
        for f in failures:
            print("FAIL:", f)
        sys.exit(1)
    print("ALL INVARIANTS HOLD (I1-I5)")


if __name__ == "__main__":
    main()
