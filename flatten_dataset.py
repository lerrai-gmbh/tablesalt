#!/usr/bin/env python3
"""
Copy a dataset's images into one flat folder named <table_id>.<ext> — exactly
what a submission container sees in /data/images during the official run.

Usage:
  python flatten_dataset.py data/dev flat/
"""

import json
import shutil
import sys
from pathlib import Path


def flatten(dataset: Path, dst: Path) -> int:
    """Replace dst with the flat image folder of dataset -> number of images."""
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    for e in manifest:
        src = dataset / e["id"] / Path(e["image"]).name
        shutil.copy(src, dst / f"{e['id']}{src.suffix}")
    return len(manifest)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    n = flatten(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"{n} images -> {sys.argv[2]}")
