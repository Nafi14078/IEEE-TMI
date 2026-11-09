"""Aggregate runs/*/metrics.json -> results/main_table.{csv,tex} (mean +- std over seeds),
plus paired bootstrap of OURS vs every other method (images pooled over seeds).
Tables in the paper must come from this script, never typed by hand.

python scripts/make_tables.py --runs runs --ours rwkd
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rwkd.metrics import paired_bootstrap  # noqa: E402

COLS = [("bal_acc", "BAcc"), ("macro_f1", "F1"), ("auroc", "AUROC"), ("ece", "ECE")]


def label(m):
    return f"{m['tag']}_{m['method']}" if m.get("tag") else m["method"]


def load(runs):
    rows = []
    for f in glob.glob(os.path.join(runs, "*", "metrics.json")):
        m = json.load(open(f))
        for split in ("test", "external"):
            if split not in m:
                continue
            r = {"dir": os.path.dirname(f), "method": label(m), "arch": m["arch"],
                 "frac": m["label_fraction"], "seed": m["seed"], "split": split}
            r.update({k: m[split][k] for k, _ in COLS})
            sel = m[split].get("selective_rel") or m[split]["selective_msp"]
            r["aurc"] = sel["aurc"]
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--ours", default="rwkd")
    a = ap.parse_args()
    df = load(a.runs)
    if df.empty:
        sys.exit("no runs found")
    os.makedirs(a.out, exist_ok=True)
    metrics = [k for k, _ in COLS] + ["aurc"]
    g = df.groupby(["split", "arch", "frac", "method"])[metrics].agg(["mean", "std", "count"])
    g.to_csv(os.path.join(a.out, "main_table.csv"))

    lines = []
    for (split, arch), sub in df.groupby(["split", "arch"]):
        lines += [f"% {split} / {arch}", r"\begin{tabular}{ll" + "c" * len(metrics) + "}", r"\toprule",
                  "Labels & Method & " + " & ".join([n for _, n in COLS] + ["AURC"]) + r" \\", r"\midrule"]
        for (frac, method), s in sub.groupby(["frac", "method"]):
            cells = [f"{s[k].mean() * 100:.1f}$\\pm${s[k].std(ddof=1) * 100 if len(s) > 1 else 0:.1f}" for k in metrics]
            lines.append(f"{frac}\\% & {method.replace('_', ' ')} & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", ""]
    open(os.path.join(a.out, "main_table.tex"), "w").write("\n".join(lines))

    tests = []
    for (split, arch, frac), sub in df.groupby(["split", "arch", "frac"]):
        ours = sub[sub.method == a.ours]
        for other in set(sub.method) - {a.ours}:
            ya, pa, pb = [], [], []
            for seed in sorted(set(ours.seed) & set(sub[sub.method == other].seed)):
                da = np.load(os.path.join(ours[ours.seed == seed].dir.iloc[0], f"preds_{split}.npz"))
                db = np.load(os.path.join(sub[(sub.method == other) & (sub.seed == seed)].dir.iloc[0], f"preds_{split}.npz"))
                assert (da["idx"] == db["idx"]).all()
                ya.append(da["y"]); pa.append(da["probs"].argmax(1)); pb.append(db["probs"].argmax(1))
            if ya:
                r = paired_bootstrap(np.concatenate(ya), np.concatenate(pa), np.concatenate(pb))
                tests.append({"split": split, "arch": arch, "frac": frac, "vs": other, **r})
    pd.DataFrame(tests).to_csv(os.path.join(a.out, "paired_bootstrap.csv"), index=False)
    print(g.xs("mean", axis=1, level=1).round(4).to_string())
    print(f"\nwrote {a.out}/main_table.csv, main_table.tex, paired_bootstrap.csv")


if __name__ == "__main__":
    main()
