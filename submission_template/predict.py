#!/usr/bin/env python3
"""
Submission entrypoint contract:
    python predict.py <input_dir> <output_dir>

<input_dir>  contains one image per table, named <table_id>.png or .jpg
<output_dir> must receive one <table_id>.json (preferred) or .html each

This template is a WORKING example: it runs the Tesseract floor baseline.
Replace its internals with your pipeline; keep the entrypoint identical.
The official run happens offline (no network), so bundle all weights.
"""
import sys
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "bt", HERE.parent / "baselines" / "baseline_tesseract.py")
bt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bt)


def main(inp, out):
    inp, out = Path(inp), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for img in sorted(inp.iterdir()):
        if img.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        (out / f"{img.stem}.html").write_text(bt.to_html(img), encoding="utf-8")
        print(img.stem, "done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
