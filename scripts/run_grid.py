"""Cross-platform (Windows/Linux/macOS) version of run_grid.sh. Skips runs that already finished,
so you can stop it and start it again at any time.

python scripts/run_grid.py teacher | main | ablation | all
"""
import os
import subprocess
import sys

CFG = "configs/default.yaml"
FRACS = [1, 5, 10]
SEEDS = [0, 1, 2]
ARCHS = ["resnet18", "mobilenetv3_large_100"]
METHODS = ["supervised", "kd_logit", "kd_embed", "pseudo_label", "rwkd"]
ABLATIONS = {
    "abl_promptonly": ["reliability.components=[prompt]"],
    "abl_augonly": ["reliability.components=[aug]"],
    "abl_confonly": ["reliability.components=[conf]"],
    "abl_rawweights": ["reliability.transform=raw"],
    "abl_vote": ["reliability.agreement=vote"],
    "abl_norelhead": ["student.rel_head=false"],
    "abl_uncalibrated": ["teacher.calibrate=none"],
    "abl_probeteacher": ["teacher.target=probe", "reliability.components=[aug]"],
}


def done(name):
    return os.path.exists(os.path.join("runs", name, "metrics.json"))


def run(cmd):
    print(">>", " ".join(cmd), flush=True)
    subprocess.run([sys.executable] + cmd, check=True)


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("teacher", "all"):
        for f in FRACS:
            for s in SEEDS:
                if not (done(f"teacher_zeroshot_lab{f}_s{s}") and done(f"teacher_probe_lab{f}_s{s}")):
                    run(["scripts/eval_teacher.py", "--config", CFG, "--set",
                         f"data.label_fraction={f}", f"data.split_seed={s}", f"train.seed={s}"])
    if what in ("main", "all"):
        for a in ARCHS:
            for f in FRACS:
                for s in SEEDS:
                    for m in METHODS:
                        if not done(f"{m}_{a}_lab{f}_s{s}"):
                            run(["-m", "rwkd.train", "--config", CFG, "--set", f"method={m}", f"student.arch={a}",
                                 f"data.label_fraction={f}", f"data.split_seed={s}", f"train.seed={s}"])
    if what in ("ablation", "all"):
        for s in SEEDS:
            for tag, extra in ABLATIONS.items():
                if not done(f"{tag}_rwkd_resnet18_lab5_s{s}"):
                    run(["-m", "rwkd.train", "--config", CFG, "--set", "method=rwkd", "student.arch=resnet18",
                         "data.label_fraction=5", f"data.split_seed={s}", f"train.seed={s}", f"tag={tag}"] + extra)


if __name__ == "__main__":
    main()
