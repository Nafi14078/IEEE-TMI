"""Task: "Implement the reliability score and check it on validation data".

Does the label-free reliability actually predict when the teacher is right? For every component
and combination: AUROC(score -> teacher correct), AURC, and teacher accuracy per reliability quintile.
Uses VAL labels only (plus, with --hidden-train, the hidden train labels, for analysis only).

python scripts/check_reliability.py --config configs/default.yaml --set data.label_fraction=5
"""
import argparse
import itertools
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rwkd.data import load_splits  # noqa: E402
from rwkd.metrics import classification_metrics, selective_metrics, risk_coverage  # noqa: E402
from rwkd.reliability import build_teacher_targets, combine  # noqa: E402
from rwkd.utils import load_config, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--hidden-train", action="store_true")
    ap.add_argument("--out", default="results/reliability_check")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    df, idx, lab = load_splits(cfg)
    y = torch.as_tensor(df["label"].values)
    tt = build_teacher_targets(cfg, df, lab, idx["train"], y)
    comps = tt["comps"]
    names = list(comps)
    scores = dict(comps)
    for k in range(2, len(names) + 1):
        for combo in itertools.combinations(names, k):
            scores["*".join(combo)] = combine(comps, list(combo), "product")

    report = {"calibration": tt["calibration"]}
    splits = ["val"] + (["train"] if a.hidden_train else [])
    os.makedirs(a.out, exist_ok=True)
    for split in splits:
        rows = idx[split]
        p, yy = tt["pbar"][rows].numpy(), y[rows].numpy()
        correct = p.argmax(1) == yy
        rep = {"teacher": classification_metrics(p, yy), "scores": {}}
        for name, s in scores.items():
            sv = s[rows].numpy()
            bins = np.argsort(np.argsort(sv, kind="stable"), kind="stable") * 5 // len(sv)  # rank quintiles
            rep["scores"][name] = {
                "auroc_correct": float(roc_auc_score(correct, sv)) if 0 < correct.mean() < 1 else float("nan"),
                **selective_metrics(sv, correct),
                "acc_by_quintile": [float(correct[bins == b].mean()) for b in range(5)],
            }
        rep["oracle_aurc"] = selective_metrics(correct.astype(float), correct)["aurc"]
        report[split] = rep
        print(f"\n== {split}: teacher bal_acc {rep['teacher']['bal_acc']:.3f} acc {rep['teacher']['acc']:.3f}")
        print(f"{'score':28s} AUROC   AURC   acc by quintile (low->high)")
        for name, r in sorted(rep["scores"].items(), key=lambda kv: -kv[1]["auroc_correct"]):
            print(f"{name:28s} {r['auroc_correct']:.3f}  {r['aurc']:.3f}  " +
                  " ".join(f"{v:.2f}" for v in r["acc_by_quintile"]))
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            for name in [n for n in scores if n in ("conf", "prompt", "aug", "prompt*aug", "aug*prompt")]:
                cov, risk = risk_coverage(scores[name][rows].numpy(), correct)
                ax.plot(cov, risk, label=name)
            ax.set_xlabel("coverage")
            ax.set_ylabel("teacher error on kept images")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(a.out, f"teacher_risk_coverage_{split}.pdf"))
        except ImportError:
            pass
    save_json(report, os.path.join(a.out, f"lab{cfg['data']['label_fraction']}_s{cfg['data']['split_seed']}.json"))


if __name__ == "__main__":
    main()
