#!/usr/bin/env python3
"""
TableSalt benchmark — synthetic dataset generator.

Generates scanned-looking table images with exact ground truth:
  * table spec sampled from a seeded RNG (fully reproducible)
  * three visual styles: full grid / horizontal rules only / borderless
  * four degradation tiers: T0 clean, T1 office scan, T2 phone photo, T3 archive
  * ground truth per table: HTML (canonical), cells.json, markdown (when
    representable without spans), qa.json (lookup + aggregate questions)

Organizers regenerate the hidden test set with a private --seed.
Usage:
  python generate_dataset.py --out data/dev --n 24 --seed 42
"""

import argparse
import json
import math
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

import os


def _find_font(kind: str) -> str:
    """Resolve a usable TTF across Linux / macOS / Windows.

    Override with env vars TABLESALT_FONT / TABLESALT_FONT_BOLD if your
    system has none of the defaults.
    """
    env = {"regular": "TABLESALT_FONT", "bold": "TABLESALT_FONT_BOLD"}
    if os.environ.get(env.get(kind, "")):
        return os.environ[env[kind]]
    candidates = {
        "regular": [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",          # Linux
            "/System/Library/Fonts/Supplemental/Arial.ttf",             # macOS
            "/System/Library/Fonts/Supplemental/Verdana.ttf",
            "C:/Windows/Fonts/arial.ttf",                               # Windows
        ],
        "bold": [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
        ],
    }
    for p in candidates[kind]:
        if Path(p).exists():
            return p
    if kind == "bold":                       # degrade gracefully: no bold
        return _find_font("regular")
    raise SystemExit(
        "No usable font found. Set TABLESALT_FONT=/path/to/a/.ttf "
        "(and optionally TABLESALT_FONT_BOLD) and retry.")


FONTS = {
    "regular": _find_font("regular"),
    "bold": _find_font("bold"),
}

# ----------------------------------------------------------------------------
# 1. Table content generation
# ----------------------------------------------------------------------------

ROW_LABEL_POOLS = [
    ["Revenue", "Cost of sales", "Gross profit", "Operating expenses", "EBITDA",
     "Depreciation", "EBIT", "Interest expense", "Taxes", "Net income",
     "Cash flow", "Capex", "Working capital", "Inventory", "Receivables"],
    ["Umsatzerlöse", "Materialaufwand", "Rohertrag", "Personalaufwand",
     "Abschreibungen", "Sonstige Kosten", "Betriebsergebnis", "Zinsaufwand",
     "Steuern", "Jahresüberschuss", "Rückstellungen", "Verbindlichkeiten",
     "Forderungen", "Vorräte", "Eigenkapital"],
    ["Hydrogen", "Helium", "Lithium", "Carbon", "Nitrogen", "Oxygen",
     "Sodium", "Magnesium", "Aluminium", "Silicon", "Phosphorus", "Sulfur",
     "Chlorine", "Potassium", "Calcium"],
    ["Baseline", "Variant A", "Variant B", "Variant C", "Ablation 1",
     "Ablation 2", "Ablation 3", "Ours (small)", "Ours (base)", "Ours (large)",
     "Prior SOTA", "Human", "Random", "Majority", "Oracle"],
]

COL_LABEL_POOLS = [
    ["Q1", "Q2", "Q3", "Q4", "FY", "YoY"],
    ["2021", "2022", "2023", "2024", "2025", "2026"],
    ["Min", "Max", "Mean", "Median", "Std", "N"],
    ["Train", "Dev", "Test", "OOD", "Avg", "Rank"],
    ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun"],
]

UNITS = ["", " %", " €", " kg", " ms", ""]


def sample_number_format(rng: random.Random) -> dict:
    """One number format per table, as in real documents: locale, thousands
    separator and negative notation never change within a table. Locales
    still differ between tables — a classic OCR confusion source."""
    style = rng.choice(["en", "de", "int"])
    return dict(style=style,
                int_sep=rng.choice([",", " "]) if style == "int" else ",",
                parens=rng.random() < 0.4)   # accounting negatives "(1,234)"


def fmt_number(rng: random.Random, fmt: dict) -> str:
    v = rng.uniform(-9999, 99999)
    if fmt["style"] == "de":
        s = f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    elif fmt["style"] == "int":
        s = f"{int(v):,}".replace(",", fmt["int_sep"])
    else:
        s = f"{v:,.2f}"
    if s.startswith("-") and fmt["parens"]:
        s = "(" + s[1:] + ")"
    return s


def make_table_spec(rng: random.Random):
    """Return dict with grid of logical cells. Cell: dict(text, r, c, rs, cs, header)."""
    n_rows = rng.randint(6, 12)         # body rows
    n_cols = rng.randint(4, 7)          # incl. stub column
    two_header_rows = rng.random() < 0.45
    num_fmt = sample_number_format(rng)
    unit = rng.choice(UNITS)

    row_labels = rng.sample(rng.choice(ROW_LABEL_POOLS), n_rows)
    col_pool = rng.choice(COL_LABEL_POOLS)
    col_labels = (col_pool * 2)[: n_cols - 1]

    cells = []
    header_rows = 2 if two_header_rows else 1

    if two_header_rows:
        # top header row: stub spans both header rows; grouped spans over data cols
        cells.append(dict(text=rng.choice(["Position", "Item", "Metric", ""]),
                          r=0, c=0, rs=2, cs=1, header=True))
        c = 1
        groups = []
        while c < n_cols:
            span = min(rng.randint(1, 3), n_cols - c)
            groups.append((c, span))
            c += span
        group_names = ["Actual", "Plan", "Delta", "Reported", "Adjusted"]
        rng.shuffle(group_names)
        for gi, (c0, span) in enumerate(groups):
            cells.append(dict(text=group_names[gi % len(group_names)],
                              r=0, c=c0, rs=1, cs=span, header=True))
        for j in range(1, n_cols):
            cells.append(dict(text=col_labels[j - 1], r=1, c=j, rs=1, cs=1, header=True))
    else:
        cells.append(dict(text=rng.choice(["Position", "Item", "Metric", ""]),
                          r=0, c=0, rs=1, cs=1, header=True))
        for j in range(1, n_cols):
            cells.append(dict(text=col_labels[j - 1], r=0, c=j, rs=1, cs=1, header=True))

    # body
    merged_body = set()
    for i in range(n_rows):
        r = header_rows + i
        cells.append(dict(text=row_labels[i], r=r, c=0, rs=1, cs=1, header=False))
        j = 1
        while j < n_cols:
            if (rng.random() < 0.06 and j < n_cols - 1
                    and (r, j) not in merged_body):
                # occasional horizontal merge in body ("n/a — discontinued")
                span = 2
                cells.append(dict(text=rng.choice(["n/a", "—", "see note 3"]),
                                  r=r, c=j, rs=1, cs=span, header=False))
                j += span
            else:
                txt = fmt_number(rng, num_fmt)
                if rng.random() < 0.10:
                    txt += unit
                if rng.random() < 0.05:
                    txt = rng.choice(["–", "n/a", ""])
                cells.append(dict(text=txt, r=r, c=j, rs=1, cs=1, header=False))
                j += 1

    total_rows = header_rows + n_rows
    has_spans = any(c["rs"] > 1 or c["cs"] > 1 for c in cells)
    return dict(cells=cells, n_rows=total_rows, n_cols=n_cols,
                header_rows=header_rows, has_spans=has_spans)


# ----------------------------------------------------------------------------
# 2. Ground truth serializations
# ----------------------------------------------------------------------------

def esc(t: str) -> str:
    return (t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def spec_to_html(spec) -> str:
    grid = {}
    for cell in spec["cells"]:
        grid.setdefault(cell["r"], []).append(cell)
    out = ["<table>"]
    for r in range(spec["n_rows"]):
        out.append("<tr>")
        for cell in sorted(grid.get(r, []), key=lambda c: c["c"]):
            tag = "th" if cell["header"] else "td"
            attrs = ""
            if cell["rs"] > 1:
                attrs += f' rowspan="{cell["rs"]}"'
            if cell["cs"] > 1:
                attrs += f' colspan="{cell["cs"]}"'
            out.append(f"<{tag}{attrs}>{esc(cell['text'])}</{tag}>")
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


def spec_to_markdown(spec):
    """GitHub-flavored markdown. Returns None if spans make it unrepresentable."""
    if spec["has_spans"]:
        return None
    rows = [[None] * spec["n_cols"] for _ in range(spec["n_rows"])]
    for cell in spec["cells"]:
        rows[cell["r"]][cell["c"]] = cell["text"]
    lines = ["| " + " | ".join(x or "" for x in rows[0]) + " |",
             "|" + "---|" * spec["n_cols"]]
    for row in rows[1:]:
        lines.append("| " + " | ".join(x or "" for x in row) + " |")
    return "\n".join(lines)


def parse_num(s: str):
    t = s.strip().replace("€", "").replace("%", "").replace("kg", "") \
         .replace("ms", "").strip()
    neg = t.startswith("(") and t.endswith(")")
    t = t.strip("()").replace(" ", "")
    # one comma + exactly 3 digits after it and no dot = thousands ("12,345"
    # from the "int" style); "de" style always has exactly 2 decimals
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


def make_qa(spec, rng: random.Random):
    """Lookup and aggregate questions answerable from the table alone."""
    hr = spec["header_rows"]
    body = [c for c in spec["cells"] if not c["header"] and c["c"] > 0
            and c["rs"] == 1 and c["cs"] == 1 and parse_num(c["text"]) is not None]
    stubs = {c["r"]: c["text"] for c in spec["cells"] if c["c"] == 0 and not c["header"]}
    colhdr = {c["c"]: c["text"] for c in spec["cells"]
              if c["header"] and c["r"] == hr - 1 and c["cs"] == 1}
    qa = []
    # only ask about columns whose header is unique (no ambiguous lookups)
    from collections import Counter
    hdr_counts = Counter(colhdr.values())
    unique_cols = {c for c, h in colhdr.items() if hdr_counts[h] == 1}
    candidates = [c for c in body if c["r"] in stubs and c["c"] in unique_cols]
    rng.shuffle(candidates)
    for c in candidates[:4]:
        qa.append(dict(type="lookup",
                       question=f'What is the value for "{stubs[c["r"]]}" '
                                f'in column "{colhdr[c["c"]]}"?',
                       answer=c["text"],
                       answer_numeric=parse_num(c["text"])))
    # one column aggregate
    cols = {}
    for c in candidates:
        cols.setdefault(c["c"], []).append(c)
    for col, items in cols.items():
        if len(items) >= 3 and col in colhdr:
            vals = [parse_num(c["text"]) for c in items]
            qa.append(dict(type="aggregate",
                           question=f'What is the sum of all numeric values in '
                                    f'column "{colhdr[col]}" for the rows '
                                    f'{[stubs[c["r"]] for c in items]}?',
                           answer=round(sum(vals), 2),
                           answer_numeric=round(sum(vals), 2)))
            break
    return qa


# ----------------------------------------------------------------------------
# 3. Rendering
# ----------------------------------------------------------------------------

def render_table(spec, style: str, rng: random.Random) -> Image.Image:
    pad, cell_pad_x, cell_pad_y = 60, 14, 10
    f_body = ImageFont.truetype(FONTS["regular"], 20)
    f_head = ImageFont.truetype(FONTS["bold"], 20)
    tmp = Image.new("L", (10, 10)); meas = ImageDraw.Draw(tmp)

    n_rows, n_cols = spec["n_rows"], spec["n_cols"]
    col_w = [40] * n_cols
    row_h = [0] * n_rows
    for c in spec["cells"]:
        f = f_head if c["header"] else f_body
        bb = meas.textbbox((0, 0), c["text"], font=f)
        w = (bb[2] - bb[0]) + 2 * cell_pad_x
        h = (bb[3] - bb[1]) + 2 * cell_pad_y + 8
        if c["cs"] == 1:
            col_w[c["c"]] = max(col_w[c["c"]], w)
        for rr in range(c["r"], c["r"] + c["rs"]):
            row_h[rr] = max(row_h[rr], int(math.ceil(h / c["rs"])))
    col_x = [pad]
    for w in col_w:
        col_x.append(col_x[-1] + w)
    row_y = [pad]
    for h in row_h:
        row_y.append(row_y[-1] + h)

    W, H = col_x[-1] + pad, row_y[-1] + pad + 30
    img = Image.new("RGB", (W, H), (252, 251, 248))
    d = ImageDraw.Draw(img)

    # header fill
    if rng.random() < 0.6:
        d.rectangle([col_x[0], row_y[0], col_x[-1],
                     row_y[spec["header_rows"]]], fill=(228, 228, 224))
    # zebra stripes sometimes
    if style != "grid" and rng.random() < 0.4:
        for r in range(spec["header_rows"], n_rows, 2):
            d.rectangle([col_x[0], row_y[r], col_x[-1], row_y[r + 1]],
                        fill=(240, 240, 236))

    ink = (25, 25, 30)
    if style == "grid":
        # borders per cell, so merged cells (rowspan/colspan) stay unbroken
        for c in spec["cells"]:
            d.rectangle([col_x[c["c"]], row_y[c["r"]],
                         col_x[c["c"] + c["cs"]], row_y[c["r"] + c["rs"]]],
                        outline=ink, width=2)
        d.rectangle([col_x[0], row_y[0], col_x[-1], row_y[-1]],
                    outline=ink, width=3)
    elif style == "hlines":
        d.line([(col_x[0], row_y[0]), (col_x[-1], row_y[0])], fill=ink, width=3)
        d.line([(col_x[0], row_y[spec["header_rows"]]),
                (col_x[-1], row_y[spec["header_rows"]])], fill=ink, width=2)
        d.line([(col_x[0], row_y[-1]), (col_x[-1], row_y[-1])], fill=ink, width=3)
    # borderless: nothing

    for c in spec["cells"]:
        f = f_head if c["header"] else f_body
        x0, x1 = col_x[c["c"]], col_x[c["c"] + c["cs"]]
        y0, y1 = row_y[c["r"]], row_y[c["r"] + c["rs"]]
        bb = meas.textbbox((0, 0), c["text"], font=f)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        if c["header"] or c["c"] == 0 or parse_num(c["text"]) is None:
            tx = x0 + cell_pad_x if c["c"] == 0 and not c["header"] \
                 else x0 + (x1 - x0 - tw) / 2
        else:
            tx = x1 - cell_pad_x - tw  # numbers right-aligned
        d.text((tx, y0 + (y1 - y0 - th) / 2 - bb[1]), c["text"], font=f, fill=ink)
    return img


# ----------------------------------------------------------------------------
# 4. Degradations
# ----------------------------------------------------------------------------

def _perspective_coeffs(src, dst):
    A, B = [], []
    for (x, y), (u, v) in zip(dst, src):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); B.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y]); B.append(v)
    return np.linalg.solve(np.array(A, dtype=float), np.array(B, dtype=float))


# Degradation model per tier. None / 0 switches a step off.
#   rotate       max skew in degrees            perspective  max corner jitter (share of size)
#   shadow       darkest end of the gradient    bleed        ghost text opacity range
#   blur         max Gaussian blur radius       noise        Gaussian noise sigma
#   salt_pepper  share of flipped pixels        contrast     contrast factor
#   binarize     (black below, white above)     jpeg         JPEG quality
# T3 is meant to be hard but legible: every character, separator and sign
# must stay readable to a careful human, otherwise the ground truth is a guess.
TIER_PARAMS = {
    "T1": dict(rotate=1.5, perspective=0, shadow=None, bleed=None, blur=0.6,
               noise=6, salt_pepper=0, contrast=0.95, binarize=None, jpeg=55),
    "T2": dict(rotate=4, perspective=0.04, shadow=(0.55, 0.8), bleed=None,
               blur=1.0, noise=10, salt_pepper=0, contrast=0.85, binarize=None,
               jpeg=45),
    "T3": dict(rotate=4, perspective=0.06, shadow=None, bleed=(0.12, 0.2),
               blur=1.0, noise=12, salt_pepper=0.002, contrast=0.9,
               binarize=(110, 200), jpeg=35),
}


def degrade(img: Image.Image, tier: str, rng: random.Random) -> Image.Image:
    if tier == "T0":
        return img
    p = TIER_PARAMS[tier]
    arr_rng = np.random.default_rng(rng.randrange(2**31))

    angle = rng.uniform(-p["rotate"], p["rotate"])
    img = img.rotate(angle, expand=True, fillcolor=(250, 249, 246),
                     resample=Image.BICUBIC)

    if p["perspective"]:  # photo taken at an angle
        W, H = img.size
        jit = lambda: rng.uniform(0, p["perspective"])
        src = [(0, 0), (W, 0), (W, H), (0, H)]
        dst = [(W * jit(), H * jit()), (W * (1 - jit()), H * jit()),
               (W * (1 - jit()), H * (1 - jit())), (W * jit(), H * (1 - jit()))]
        img = img.transform((W, H), Image.PERSPECTIVE,
                            _perspective_coeffs(src, dst),
                            resample=Image.BICUBIC, fillcolor=(245, 244, 240))

    if p["shadow"]:  # shadow gradient
        W, H = img.size
        grad = np.tile(np.linspace(1.0, rng.uniform(*p["shadow"]), W,
                                   dtype=np.float32), (H, 1))
        a = np.asarray(img, dtype=np.float32) * grad[..., None]
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

    if p["bleed"]:  # bleed-through ghost text
        ghost = img.transpose(Image.FLIP_LEFT_RIGHT).filter(
            ImageFilter.GaussianBlur(2))
        img = Image.blend(img, ghost, rng.uniform(*p["bleed"]))

    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, p["blur"])))
    a = np.asarray(img, dtype=np.float32)
    a = a + arr_rng.normal(0, p["noise"], a.shape)
    if p["salt_pepper"]:
        mask = arr_rng.random(a.shape[:2])
        a[mask < p["salt_pepper"]] = 0
        a[mask > 1 - p["salt_pepper"]] = 255
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    img = ImageEnhance.Contrast(img).enhance(p["contrast"])
    if p["binarize"]:  # fax-style soft binarization
        lo, hi = p["binarize"]
        img = img.convert("L").point(
            lambda v: 255 if v > hi else (0 if v < lo else v)).convert("RGB")
    return img


# ----------------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------------

TIERS = ["T0", "T1", "T2", "T3"]
STYLES = ["grid", "hlines", "borderless"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/dev")
    ap.add_argument("--n", type=int, default=24, help="number of base tables")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jpeg-quality", type=int, default=0,
                    help="if >0, save JPEG at this quality instead of PNG")
    ap.add_argument("--clean", action="store_true",
                    help="render every table at tier T0 (clean PNG) for the "
                         "print-and-rescan workflow: print these, scan them "
                         "back, replace each image file with the scan - the "
                         "gt/ folder stays valid because content is known")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    manifest = []
    for i in range(args.n):
        tid = f"table_{i:03d}"
        spec = make_table_spec(rng)
        style = STYLES[i % len(STYLES)]
        tier = "T0" if args.clean else TIERS[i % len(TIERS)]

        base = render_table(spec, style, rng)
        img = degrade(base, tier, rng)

        tdir = out / tid
        (tdir / "gt").mkdir(parents=True, exist_ok=True)
        if args.jpeg_quality > 0 or tier in ("T1", "T2", "T3"):
            q = args.jpeg_quality or TIER_PARAMS.get(tier, {}).get("jpeg", 90)
            img_path = tdir / "image.jpg"
            img.save(img_path, quality=q)
        else:
            img_path = tdir / "image.png"
            img.save(img_path)

        (tdir / "gt" / "table.html").write_text(spec_to_html(spec), encoding="utf-8")
        (tdir / "gt" / "cells.json").write_text(
            json.dumps(spec["cells"], ensure_ascii=False, indent=1), encoding="utf-8")
        md = spec_to_markdown(spec)
        if md:
            (tdir / "gt" / "table.md").write_text(md, encoding="utf-8")
        qa = make_qa(spec, rng)
        (tdir / "gt" / "qa.json").write_text(
            json.dumps(qa, ensure_ascii=False, indent=1), encoding="utf-8")

        manifest.append(dict(id=tid, image=img_path.relative_to(out).as_posix(),
                             style=style, tier=tier,
                             has_spans=spec["has_spans"],
                             n_rows=spec["n_rows"], n_cols=spec["n_cols"],
                             n_qa=len(qa)))
        print(f"{tid}: style={style:10s} tier={tier} "
              f"{spec['n_rows']}x{spec['n_cols']} spans={spec['has_spans']}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"\nWrote {len(manifest)} tables to {out}/")


if __name__ == "__main__":
    main()
