"""Why is the zero-shot teacher weak? Prints, on VALIDATION data:
  - what each prompt paraphrase set predicts (balanced accuracy + how often each class is predicted)
  - the confusion matrix of the prompt ensemble, before and after bias calibration
  - the teacher's linear probe for comparison
Fast (CPU, cached embeddings only).

python scripts/diagnose_teacher.py --set data.label_fraction=5
"""
import argparse
import os
import sys

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rwkd.data import load_splits  # noqa: E402
from rwkd.reliability import load_cache, teacher_probs  # noqa: E402
from rwkd.utils import load_config  # noqa: E402


def show_cm(name, pred, y, classes):
    cm = confusion_matrix(y, pred, labels=range(len(classes)))
    print(f"\n{name}: balanced acc {balanced_accuracy_score(y, pred):.3f}  (rows = true class, cols = predicted)")
    print("        " + " ".join(f"{c:>6s}" for c in classes))
    for c, row in zip(classes, cm):
        print(f"{c:>6s}  " + " ".join(f"{v:6d}" for v in row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    df, idx, lab = load_splits(cfg)
    classes = cfg["data"]["classes"]
    y = torch.as_tensor(df["label"].values)
    cache = load_cache(cfg["teacher"]["cache"], df["image_id"].values)
    val = idx["val"]
    yv = y[val].numpy()
    print("val class counts:", dict(zip(classes, np.bincount(yv, minlength=len(classes)).tolist())))

    img = cache["img_emb"].float()[:, 0]  # view 0
    txt = cache["txt_emb"].float()        # [K, C, D]
    print("\nPer prompt set (view 0, no calibration):")
    print(f"{'k':>2s} {'bal_acc':>7s}  predicted-class share   prompt for class 0")
    for k in range(txt.shape[0]):
        pred = (img[val] @ txt[k].T).argmax(1).numpy()
        share = np.bincount(pred, minlength=len(classes)) / len(pred)
        print(f"{k:2d} {balanced_accuracy_score(yv, pred):7.3f}  " +
              " ".join(f"{c}:{s:.2f}" for c, s in zip(classes, share)) + f"   '{cache['prompts'][k][0]}'")

    tcfg = dict(cfg["teacher"])
    for cal in ("none", "temperature", "temperature_bias"):
        tcfg["calibrate"] = cal
        p, info = teacher_probs(cache, "zeroshot", lab, y, tcfg)
        show_cm(f"zero-shot ensemble, calibrate={cal}", p.mean((1, 2))[val].argmax(1).numpy(), yv, classes)
    p, _ = teacher_probs(cache, "probe", lab, y, tcfg)
    show_cm(f"linear probe on {len(lab)} labelled images", p.mean((1, 2))[val].argmax(1).numpy(), yv, classes)


if __name__ == "__main__":
    main()
