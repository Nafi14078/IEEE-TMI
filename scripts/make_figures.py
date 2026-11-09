"""Figures from script outputs only: (1) balanced accuracy vs label fraction (test & external),
(2) student risk-coverage curves.

python scripts/make_figures.py --runs runs --arch resnet18
"""
import argparse
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from make_tables import load  # noqa: E402
from rwkd.metrics import risk_coverage  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results/figs")
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--frac", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = load(a.runs)
    df = df[df.arch.isin([a.arch, "teacher"])]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=False)
    for ax, split in zip(axes, ("test", "external")):
        for method, s in df[df.split == split].groupby("method"):
            g = s.groupby("frac")["bal_acc"].agg(["mean", "std"]).reset_index()
            ax.errorbar(g["frac"], g["mean"], yerr=g["std"].fillna(0), marker="o", capsize=2, label=method)
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 10], ["1%", "5%", "10%"])
        ax.set_xlabel("labelled fraction")
        ax.set_ylabel("balanced accuracy")
        ax.set_title(split)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, f"bacc_vs_labels_{a.arch}.pdf"))

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    sub = df[(df.split == "test") & (df.frac == a.frac) & (df.seed == a.seed)]
    for _, r in sub.iterrows():
        d = np.load(os.path.join(r.dir, "preds_test.npz"))
        correct = d["probs"].argmax(1) == d["y"]
        score = d["rel"] if "rel" in d.files else d["probs"].max(1)
        cov, risk = risk_coverage(score, correct)
        ax.plot(cov, risk, label=r.method)
    ax.set_xlabel("coverage")
    ax.set_ylabel("error on kept images")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, f"risk_coverage_{a.arch}_lab{a.frac}.pdf"))
    print(f"figures written to {a.out}")


if __name__ == "__main__":
    main()
