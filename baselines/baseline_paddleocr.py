#!/usr/bin/env python3
"""
Reference baseline: untuned PaddleOCR table recognition (PP-Structure).

Runs the stock table-recognition pipeline on every dev image and writes
its HTML as <table_id>.html. No fine-tuning, no pre/post-processing —
this is "what you get out of the box", a strong open baseline to beat.
Layout detection is switched off because every input is already a
single-table crop.

Handles both interfaces:
  * paddleocr 3.x  TableRecognitionPipelineV2 (verified: paddleocr 3.7.0,
                   paddlex 3.7.2, paddlepaddle 3.3.1, Python 3.12, macOS arm64)
  * paddleocr 2.x  PPStructure(layout=False)   (not executed here)

Install (deliberately not in requirements.txt — heavy, platform-specific):
  pip install paddlepaddle paddleocr "paddlex[ocr]"
First run downloads model weights (~1.3 GB) to ~/.paddlex.

Usage:
  python baselines/baseline_paddleocr.py --gt data/dev --out predictions_paddle/
"""

import argparse
import json
import time
from pathlib import Path

import paddleocr


def make_engine():
    major = int(paddleocr.__version__.split(".")[0])
    if major >= 3:
        from paddleocr import TableRecognitionPipelineV2
        pipe = TableRecognitionPipelineV2(use_layout_detection=False)

        def run(img_path):
            for res in pipe.predict(str(img_path)):
                j = res.json
                j = j.get("res", j)
                for t in j.get("table_res_list", []):
                    if t.get("pred_html"):
                        return t["pred_html"]
            return "<table></table>"
        return run

    from paddleocr import PPStructure
    engine = PPStructure(layout=False, show_log=False)

    def run(img_path):
        import cv2
        for region in engine(cv2.imread(str(img_path))):
            html = (region.get("res") or {}).get("html")
            if html:
                return html
        return "<table></table>"
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(args.gt) / "manifest.json").read_text(encoding="utf-8"))
    run = make_engine()
    print(f"paddleocr {paddleocr.__version__}")
    for e in manifest:
        img = Path(args.gt) / e["id"] / Path(e["image"]).name
        t0 = time.time()
        (out / f"{e['id']}.html").write_text(run(img), encoding="utf-8")
        print(e["id"], f"done {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
