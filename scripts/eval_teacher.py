"""Teacher baselines: zero-shot (prompt ensemble, optionally calibrated on the labelled images)
and the teacher's own linear probe. Writes runs/<name>/metrics.json like a student run.

python scripts/eval_teacher.py --config configs/default.yaml --set data.label_fraction=5 train.seed=0
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rwkd.data import load_splits  # noqa: E402
from rwkd.metrics import classification_metrics, selective_metrics  # noqa: E402
from rwkd.reliability import build_teacher_targets  # noqa: E402
from rwkd.utils import load_config, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    df, idx, lab = load_splits(cfg)
    y = torch.as_tensor(df["label"].values)
    frac, seed = cfg["data"]["label_fraction"], cfg["data"]["split_seed"]
    for target in ("zeroshot", "probe"):
        cfg["teacher"]["target"] = target
        if target == "probe":  # no prompts for the probe: only augmentation agreement exists
            cfg["reliability"]["components"] = [c for c in cfg["reliability"]["components"] if c != "prompt"] or ["aug"]
        tt = build_teacher_targets(cfg, df, lab, idx["train"], y)
        name = f"teacher_{target}_lab{frac}_s{seed}"
        out_dir = os.path.join(cfg.get("out_dir", "runs"), name)
        os.makedirs(out_dir, exist_ok=True)
        res = {"run": name, "method": f"teacher_{target}", "tag": "", "arch": "teacher",
               "label_fraction": frac, "seed": seed, "n_labelled": int(len(lab)),
               "teacher_calibration": tt["calibration"]}
        for split in ("val", "test", "external"):
            rows = idx[split]
            if len(rows) == 0:
                continue
            p, yy, r = tt["pbar"][rows].numpy(), y[rows].numpy(), tt["r"][rows].numpy()
            m = classification_metrics(p, yy)
            correct = p.argmax(1) == yy
            m["selective_msp"] = selective_metrics(p.max(1), correct)
            m["selective_rel"] = selective_metrics(r, correct)
            res[split] = m
            np.savez_compressed(os.path.join(out_dir, f"preds_{split}.npz"), probs=p, y=yy, idx=rows, rel=r)
            print(f"{name} {split}: bal_acc {m['bal_acc']:.4f} auroc {m['auroc']:.4f}")
        save_json(res, os.path.join(out_dir, "metrics.json"))


if __name__ == "__main__":
    main()
