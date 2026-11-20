#!/usr/bin/env python3
"""
Official scorer: one command, final score and ranking.

  final = 0.85 * tier-weighted TEDS + 0.15 * QA accuracy
  tier-weighted TEDS = 0.1*T0 + 0.2*T1 + 0.3*T2 + 0.4*T3 (mean TEDS per tier)
  tie-breaker: cell-level F1

Score your own predictions (teams):
  python score_all.py --gt data/dev --pred predictions/

Rank all teams (organizers):
  python score_all.py --gt data/test --teams teams/ [--real data/test_real]
  expects <teams>/<team>/predictions/ (falls back to <teams>/<team>/ itself),
  plus <team>/predictions_real/ with --real and <team>/run.json (runtime and
  image size, written by the official run) when present.

The real-scan score (--real) is the mean TEDS on the print-and-rescan set. It
is reported in its own column and never enters the final score or the ranking.
"""

import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev, qa = load("evaluate"), load("eval_qa")


def score_set(gt_dir: Path, pred_dir: Path):
    """-> (per-tier TEDS lists, cell-F1 list, QA correctness list)"""
    manifest = json.loads((gt_dir / "manifest.json").read_text(encoding="utf-8"))
    per_tier_teds, f1s, qa_ok = {}, [], []
    for e in manifest:
        tid = e["id"]
        gt_html = (gt_dir / tid / "gt" / "table.html").read_text(encoding="utf-8")
        pred_html = ev.load_prediction(pred_dir, tid)
        per_tier_teds.setdefault(e["tier"], []).append(ev.teds(gt_html, pred_html) if pred_html else 0.0)
        f1s.append(ev.cell_f1(gt_html, pred_html) if pred_html else 0.0)
        grid = qa.load_grid(pred_dir, tid)
        for q in json.loads((gt_dir / tid / "gt" / "qa.json").read_text(encoding="utf-8")):
            qa_ok.append(qa.grade_one(q, qa.auto_answer(q, grid)))
    return per_tier_teds, f1s, qa_ok


def score_team(name: str, gt_dir: Path, pred_dir: Path) -> dict:
    per_tier_teds, f1s, qa_ok = score_set(gt_dir, pred_dir)
    per_tier = {t: sum(v) / len(v) for t, v in sorted(per_tier_teds.items())}
    teds_w = ev.tier_weighted(per_tier)
    qa_acc = sum(qa_ok) / len(qa_ok) if qa_ok else 0.0
    return dict(team=name,
                final=round(ev.final_score(teds_w, qa_acc), 4),
                teds_weighted=round(teds_w, 4),
                qa_accuracy=round(qa_acc, 4),
                cell_f1=round(sum(f1s) / len(f1s), 4),
                per_tier={t: round(v, 4) for t, v in per_tier.items()})


def team_dirs(teams_root: Path):
    return sorted(p for p in teams_root.iterdir() if p.is_dir())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="dataset dir with manifest.json")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pred", help="one prediction dir (score a single submission)")
    src.add_argument("--teams", help="dir with one sub-dir per team (official ranking)")
    ap.add_argument("--real", default=None, help="print-and-rescan set, reported separately")
    ap.add_argument("--out", default="ranking.json")
    args = ap.parse_args()
    gt_dir = Path(args.gt)

    if args.pred:
        entries = [(Path(args.pred).name or "submission", Path(args.pred), None)]
    else:
        entries = []
        for d in team_dirs(Path(args.teams)):
            pred = d / "predictions"
            entries.append((d.name, pred if pred.is_dir() else d, d))

    results = []
    for name, pred_dir, team_dir in entries:
        row = score_team(name, gt_dir, pred_dir)
        if args.real and team_dir is not None:
            real, _, _ = score_set(Path(args.real), team_dir / "predictions_real")
            vals = [x for v in real.values() for x in v]
            row["real_scan_teds"] = round(sum(vals) / len(vals), 4) if vals else 0.0
        run_json = team_dir / "run.json" if team_dir is not None else None
        if run_json is not None and run_json.exists():
            row["run"] = json.loads(run_json.read_text(encoding="utf-8"))
            row["team"] = row["run"].get("team", row["team"])  # display name, not the folder slug
        results.append(row)

    results.sort(key=lambda r: (r["final"], r["cell_f1"]), reverse=True)
    real_hdr = f" {'Real':>6s}" if args.real else ""
    print(f"{'#':>2} {'team':20s} {'FINAL':>6s} {'TEDSw':>6s} {'QA':>6s} {'CellF1':>6s}{real_hdr}"
          f" {'run_s':>6s} {'img_GB':>6s}  per-tier TEDS")
    for i, r in enumerate(results, 1):
        run = r.get("run", {})
        real = f" {r.get('real_scan_teds', 0.0):6.4f}" if args.real else ""
        secs = f"{run['seconds']:6.0f}" if run.get("seconds") is not None else f"{'-':>6s}"
        size = f"{run['image_gb']:6.2f}" if run.get("image_gb") is not None else f"{'-':>6s}"
        tiers = " ".join(f"{t}:{v:.3f}" for t, v in r["per_tier"].items())
        status = f"  [{run['status']}]" if run.get("status", "ok") != "ok" else ""
        print(f"{i:>2} {r['team']:20s} {r['final']:6.4f} {r['teds_weighted']:6.4f} {r['qa_accuracy']:6.4f}"
              f" {r['cell_f1']:6.4f}{real} {secs} {size}  {tiers}{status}")
    print(f"\nFINAL = {ev.W_TEDS} x tier-weighted TEDS {ev.TIER_WEIGHTS} + {ev.W_QA} x QA accuracy;"
          " tie-breaker CellF1.")
    if args.real:
        print("Real = mean TEDS on the print-and-rescan set (reported, not ranked).")
    if not args.pred:
        print("run_s / img_GB = runtime and image size of the official run (reported, not ranked).")
    Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
