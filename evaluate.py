#!/usr/bin/env python3
"""
TableSalt benchmark — structure & content evaluation.

Metrics:
  * TEDS   — Tree-Edit-Distance-based Similarity (Zhong et al., ECCV 2020),
             computed on the <table> HTML tree; cell text compared via
             normalized Levenshtein inside the edit cost.
  * TEDS-S — structure-only variant (cell text ignored).
  * CellF1 — exact-match F1 over (row, col, normalized text) triples,
             spans expanded to their grid footprint.

Submission format: for every table id in the manifest, a file
  <pred_dir>/<table_id>.json   cell list [{"text","r","c","rs","cs"}] (preferred)
  <pred_dir>/<table_id>.html   containing a single <table>...</table>
If both exist, .json wins.

Usage:
  python evaluate.py --gt data/dev --pred predictions/ [--report report.json]
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path

from apted import APTED, Config
from apted.helpers import Tree
from lxml import html as lxml_html


# ----------------------------------------------------------------------------
# HTML -> tree
# ----------------------------------------------------------------------------

class TableTree(Tree):
    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content          # list of characters, or None
        self.children = list(children)

    def bracket(self):
        if self.tag in ("td", "th"):
            s = (f'"tag": {self.tag}, "colspan": {self.colspan}, '
                 f'"rowspan": {self.rowspan}, "text": {self.content}')
        else:
            s = f'"tag": {self.tag}'
        return "{{{}{}}}".format(s, "".join(c.bracket() for c in self.children))


def norm_text(t: str) -> str:
    t = unicodedata.normalize("NFKC", t or "")
    t = t.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", t).strip()


MAX_SPAN = 100  # far above any generated table; stops runaway spans hanging the run

# The official score. Degraded tiers weigh more: robustness is the point.
TIER_WEIGHTS = {"T0": 0.1, "T1": 0.2, "T2": 0.3, "T3": 0.4}
W_TEDS, W_QA = 0.85, 0.15


def tier_weighted(per_tier_mean: dict) -> float:
    """Weighted mean over tiers; a tier missing from the set contributes 0."""
    return sum(w * per_tier_mean.get(t, 0.0) for t, w in TIER_WEIGHTS.items())


def final_score(teds_weighted: float, qa_accuracy: float) -> float:
    return W_TEDS * teds_weighted + W_QA * qa_accuracy


def span(v) -> int:
    """rowspan/colspan value -> int in [1, MAX_SPAN]; junk counts as 1."""
    m = re.match(r"\s*(\d+)", str(v if v is not None else ""))
    return min(max(int(m.group(1)), 1), MAX_SPAN) if m else 1


def read_pred(path: Path) -> str:
    """Team files: tolerate a BOM and non-UTF-8 bytes instead of crashing."""
    return path.read_bytes().decode("utf-8-sig", errors="replace")


def html_to_tree(html_str: str, structure_only: bool) -> TableTree:
    root = lxml_html.fromstring(html_str)
    tables = root.xpath("//table")
    if not tables:
        raise ValueError("no <table> element found")
    table = tables[0]

    def convert(node):
        tag = node.tag.lower() if isinstance(node.tag, str) else "x"
        if tag in ("td", "th"):
            text = norm_text("".join(node.itertext()))
            content = None if structure_only else list(text)
            return TableTree("td",  # th/td unified: semantics live in text
                             span(node.get("colspan", 1)),
                             span(node.get("rowspan", 1)),
                             content)
        sections = ("thead", "tbody", "tfoot")
        children = [convert(ch) for ch in node
                    if isinstance(ch.tag, str)
                    and ch.tag.lower() in sections + ("tr", "td", "th")]
        # flatten thead/tbody/tfoot so section tags don't distort the distance
        flat = []
        for ch_node, ch_tree in zip(
                [c for c in node if isinstance(c.tag, str)
                 and c.tag.lower() in sections + ("tr", "td", "th")],
                children):
            if ch_node.tag.lower() in sections:
                flat.extend(ch_tree.children)
            else:
                flat.append(ch_tree)
        return TableTree(tag if tag in ("table", "tr") else "table",
                         None, None, None, *flat)

    return convert(table)


def levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class TEDSConfig(Config):
    @staticmethod
    def maximum(*seqs):
        return max(map(len, seqs))

    def rename(self, n1, n2):
        if (n1.tag != n2.tag or n1.colspan != n2.colspan
                or n1.rowspan != n2.rowspan):
            return 1.0
        if n1.tag == "td" and n1.content is not None:
            m = self.maximum(n1.content, n2.content)
            return levenshtein(n1.content, n2.content) / m if m else 0.0
        return 0.0


def count_nodes(t: TableTree) -> int:
    return 1 + sum(count_nodes(c) for c in t.children)


def teds(gt_html: str, pred_html: str, structure_only=False) -> float:
    t_gt = html_to_tree(gt_html, structure_only)
    try:
        t_pred = html_to_tree(pred_html, structure_only)
    except Exception:
        return 0.0
    n = max(count_nodes(t_gt), count_nodes(t_pred))
    if n == 0:
        return 1.0
    dist = APTED(t_gt, t_pred, TEDSConfig()).compute_edit_distance()
    return max(0.0, 1.0 - dist / n)


# ----------------------------------------------------------------------------
# Cell-level F1
# ----------------------------------------------------------------------------

def html_to_grid_cells(html_str: str):
    """Expand the table into {(r, c, text)} with spans footprinted."""
    try:
        tables = lxml_html.fromstring(html_str).xpath("//table")
    except Exception:  # empty / unparseable document
        return set()
    if not tables:
        return set()
    occupied, out = {}, set()
    r = 0
    for tr in tables[0].xpath(".//tr"):
        c = 0
        for cell in tr.xpath("./td|./th"):
            while (r, c) in occupied:
                c += 1
            rs = span(cell.get("rowspan", 1))
            cs = span(cell.get("colspan", 1))
            text = norm_text("".join(cell.itertext()))
            for dr in range(rs):
                for dc in range(cs):
                    occupied[(r + dr, c + dc)] = True
            out.add((r, c, text))
            c += cs
        r += 1
    return out


def cell_f1(gt_html: str, pred_html: str) -> float:
    gt, pred = html_to_grid_cells(gt_html), html_to_grid_cells(pred_html)
    if not gt and not pred:
        return 1.0
    tp = len(gt & pred)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(gt)
    return 2 * p * r / (p + r)


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------

def cells_json_to_html(cells) -> str:
    """Convert a cell list [{text,r,c,rs,cs}] to canonical <table> HTML."""
    def e(t):
        return (str(t).replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;"))
    rows = {}
    for cell in cells:
        rows.setdefault(int(cell["r"]), []).append(cell)
    out = ["<table>"]
    for r in sorted(rows):
        out.append("<tr>")
        for cell in sorted(rows[r], key=lambda x: int(x["c"])):
            rs, cs = span(cell.get("rs", 1)), span(cell.get("cs", 1))
            attrs = (f' rowspan="{rs}"' if rs > 1 else "") + \
                    (f' colspan="{cs}"' if cs > 1 else "")
            out.append(f"<td{attrs}>{e(cell.get('text', ''))}</td>")
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


def load_prediction(pred_dir: Path, tid: str):
    """Return prediction as HTML string; JSON (preferred) or HTML file."""
    pj = pred_dir / f"{tid}.json"
    if pj.exists():
        try:
            return cells_json_to_html(json.loads(read_pred(pj)))
        except Exception:
            return None  # unparseable JSON = no valid submission
    ph = pred_dir / f"{tid}.html"
    return read_pred(ph) if ph.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="dataset dir with manifest.json")
    ap.add_argument("--pred", required=True, help="dir with <table_id>.json or .html files")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    gt_dir = Path(args.gt)
    pred_dir = Path(args.pred)
    manifest = json.loads((gt_dir / "manifest.json").read_text(encoding="utf-8"))

    rows, by_tier, by_style = [], {}, {}
    for entry in manifest:
        tid = entry["id"]
        gt_html = (gt_dir / tid / "gt" / "table.html").read_text(encoding="utf-8")
        pred_html = load_prediction(pred_dir, tid)
        if pred_html is not None:
            scores = dict(teds=round(teds(gt_html, pred_html), 4),
                          teds_s=round(teds(gt_html, pred_html, True), 4),
                          cell_f1=round(cell_f1(gt_html, pred_html), 4),
                          missing=False)
        else:
            scores = dict(teds=0.0, teds_s=0.0, cell_f1=0.0, missing=True)
        row = dict(id=tid, tier=entry["tier"], style=entry["style"], **scores)
        rows.append(row)
        by_tier.setdefault(entry["tier"], []).append(row)
        by_style.setdefault(entry["style"], []).append(row)

    def avg(items, key):
        return round(sum(i[key] for i in items) / len(items), 4)

    print(f"{'id':12s} {'tier':4s} {'style':10s} {'TEDS':>7s} {'TEDS-S':>7s} {'CellF1':>7s}")
    for r in rows:
        flag = "  MISSING" if r["missing"] else ""
        print(f"{r['id']:12s} {r['tier']:4s} {r['style']:10s} "
              f"{r['teds']:7.4f} {r['teds_s']:7.4f} {r['cell_f1']:7.4f}{flag}")
    print("\n-- by tier --")
    for t in sorted(by_tier):
        print(f"{t}: TEDS={avg(by_tier[t], 'teds')}  "
              f"TEDS-S={avg(by_tier[t], 'teds_s')}  "
              f"CellF1={avg(by_tier[t], 'cell_f1')}  (n={len(by_tier[t])})")
    print("-- by style --")
    for s in sorted(by_style):
        print(f"{s}: TEDS={avg(by_style[s], 'teds')}  (n={len(by_style[s])})")

    headline = round(tier_weighted({t: avg(v, "teds") for t, v in by_tier.items()}), 4)
    print(f"\nTier-weighted TEDS (w={TIER_WEIGHTS}): {headline}")
    print("This is the TEDS part of the final score; score_all.py adds QA and prints the final score.")

    if args.report:
        Path(args.report).write_text(json.dumps(
            dict(per_table=rows, headline=headline), indent=1), encoding="utf-8")
        print(f"report written to {args.report}")


if __name__ == "__main__":
    main()
