"""Freeze the splits, label fractions and class mapping (task: "Freeze splits ...").

In-distribution: ISIC 2019 training set (dermoscopy) -> train / val / test, grouped by lesion_id
so that images of the same lesion never cross splits.
External: PAD-UFES-20 (smartphone clinical photos) -> 'external', test only.
Labelled subsets (1/5/10 %) are stratified, nested (1% c 5% c 10%) and drawn per seed.

python scripts/prepare_splits.py \
  --isic-gt data/raw/isic2019/ISIC_2019_Training_GroundTruth.csv \
  --isic-meta data/raw/isic2019/ISIC_2019_Training_Metadata.csv \
  --isic-images data/raw/isic2019/ISIC_2019_Training_Input \
  --pad-meta data/raw/pad_ufes20/metadata.csv --pad-images data/raw/pad_ufes20/images
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ISIC_COLS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
PAD_TO_ISIC = {"ACK": "AK", "BCC": "BCC", "MEL": "MEL", "NEV": "NV", "SCC": "SCC", "SEK": "BKL"}
DEFAULT_CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "SCC"]  # classes shared by ISIC 2019 and PAD-UFES-20


def isic_table(gt_csv, meta_csv, img_dir, classes):
    gt = pd.read_csv(gt_csv)
    gt["label_name"] = gt[ISIC_COLS].idxmax(axis=1)
    gt = gt[gt["label_name"].isin(classes)].copy()
    meta = pd.read_csv(meta_csv)[["image", "lesion_id"]]
    gt = gt.merge(meta, on="image", how="left")
    gt["group"] = gt["lesion_id"].fillna(gt["image"])
    files = {}  # search recursively: Kaggle copies of ISIC 2019 use different folder layouts
    for root, _, fs in os.walk(img_dir, followlinks=True):
        for f in fs:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                files.setdefault(f.rsplit(".", 1)[0], os.path.join(root, f))
    missing = [i for i in gt["image"] if i not in files]
    if missing:
        print(f"WARNING: {len(missing)} ISIC images not found, e.g. {missing[:3]}")
        gt = gt[~gt["image"].isin(missing)]
    return pd.DataFrame({
        "image_id": gt["image"], "dataset": "isic2019",
        "src_path": [files[i] for i in gt["image"]],
        "path": [f"isic2019/{i}.jpg" for i in gt["image"]],
        "label_name": gt["label_name"], "group": "isic_" + gt["group"].astype(str)})


def pad_table(meta_csv, img_dir, classes):
    meta = pd.read_csv(meta_csv)
    files = {}
    for root, _, fs in os.walk(img_dir, followlinks=True):
        for f in fs:
            files[f] = os.path.join(root, f)
    meta["label_name"] = meta["diagnostic"].map(PAD_TO_ISIC)
    meta = meta[meta["label_name"].isin(classes)]
    missing = [i for i in meta["img_id"] if i not in files]
    if missing:
        print(f"WARNING: {len(missing)} PAD-UFES-20 images not found, e.g. {missing[:3]}")
        meta = meta[~meta["img_id"].isin(missing)]
    stem = meta["img_id"].str.rsplit(".", n=1).str[0]
    return pd.DataFrame({
        "image_id": "pad_" + stem, "dataset": "pad_ufes20",
        "src_path": [files[i] for i in meta["img_id"]],
        "path": [f"pad_ufes20/{s}.jpg" for s in stem],
        "label_name": meta["label_name"].values, "group": "pad_" + meta["lesion_id"].astype(str)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isic-gt", required=True)
    ap.add_argument("--isic-meta", required=True)
    ap.add_argument("--isic-images", required=True)
    ap.add_argument("--pad-meta", required=True)
    ap.add_argument("--pad-images", required=True)
    ap.add_argument("--out", default="data/splits/splits.csv")
    ap.add_argument("--classes", default=",".join(DEFAULT_CLASSES))
    ap.add_argument("--fractions", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--test-folds", type=int, default=5, help="1/k of ISIC lesions -> test (k=5: 20%)")
    ap.add_argument("--val-folds", type=int, default=8, help="1/k of the rest -> val (k=8: 10% overall)")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--min-per-class", type=int, default=2)
    a = ap.parse_args()
    classes = a.classes.split(",")

    isic = isic_table(a.isic_gt, a.isic_meta, a.isic_images, classes)
    pad = pad_table(a.pad_meta, a.pad_images, classes)
    isic["split"] = "train"
    y = isic["label_name"].values
    g = isic["group"].values
    tr, te = next(StratifiedGroupKFold(a.test_folds, shuffle=True, random_state=a.split_seed).split(isic, y, g))
    isic.iloc[te, isic.columns.get_loc("split")] = "test"
    tr2, va = next(StratifiedGroupKFold(a.val_folds, shuffle=True, random_state=a.split_seed + 1)
                   .split(tr, y[tr], g[tr]))
    isic.iloc[tr[va], isic.columns.get_loc("split")] = "val"
    pad["split"] = "external"
    df = pd.concat([isic, pad], ignore_index=True)
    df["label"] = df["label_name"].map({c: i for i, c in enumerate(classes)}).astype(int)

    train_rows = np.where(df["split"] == "train")[0]
    for s in a.seeds:
        rng = np.random.default_rng(1000 + s)
        perms = {c: rng.permutation(train_rows[df["label_name"].values[train_rows] == c]) for c in classes}
        for f in sorted(a.fractions):
            col = f"lab{f}_s{s}"
            df[col] = 0
            for c, rows in perms.items():
                n = min(len(rows), max(a.min_per_class, int(round(f / 100 * len(rows)))))
                df.loc[rows[:n], col] = 1

    assert not set(df[df.split == "train"].group) & set(df[df.split == "test"].group), "lesion leak"
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    print(df.groupby(["split", "label_name"]).size().unstack(fill_value=0))
    for col in [c for c in df.columns if re.fullmatch(r"lab\d+_s\d+", c)]:
        print(col, int(df[col].sum()), "labelled")


if __name__ == "__main__":
    main()
