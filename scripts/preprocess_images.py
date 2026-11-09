"""Resize every image once (short side 256 px, JPEG q=95) so training is not I/O bound.

python scripts/preprocess_images.py --csv data/splits/splits.csv --out-root data/processed
"""
import argparse
import os
from multiprocessing import Pool

import pandas as pd
from PIL import Image

SHORT = 256


def work(job):
    src, dst = job
    if os.path.exists(dst):
        return 0
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    img = Image.open(src).convert("RGB")
    w, h = img.size
    s = SHORT / min(w, h)
    if s < 1:
        img = img.resize((round(w * s), round(h * s)), Image.BICUBIC)
    img.save(dst, quality=95)
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/splits/splits.csv")
    ap.add_argument("--out-root", default="data/processed")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    jobs = [(s, os.path.join(a.out_root, p)) for s, p in zip(df["src_path"], df["path"])]
    with Pool(a.workers) as pool:
        done = sum(pool.imap_unordered(work, jobs, chunksize=64))
    print(f"wrote {done} new images, {len(jobs)} total under {a.out_root}")


if __name__ == "__main__":
    main()
