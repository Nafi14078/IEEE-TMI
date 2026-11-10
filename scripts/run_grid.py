"""Cross-platform experiment grid. Skips runs that already finished, so it can be stopped and
restarted at any time (e.g. across Kaggle sessions).

python scripts/run_grid.py teacher | main | ablation | all
  [--seeds 0 1 2] [--archs resnet18] [--methods rwkd kd_logit] [--fracs 1 5 10]
  [--shard 0 --nshards 2]       split the list between 2 GPUs / 2 accounts
  [--max-hours 10.5]            do not START a new run after this many hours
  [--git-sync]                  commit+push each finished run's small result files to GitHub
  [--extra data.num_workers=2 train.save_ckpt=false]   extra --set overrides for every run
"""
import argparse
import glob
import os
import subprocess
import sys
import time
from contextlib import contextmanager

METHODS = ["supervised", "kd_logit", "kd_embed", "pseudo_label", "rwkd"]
ABLATIONS = {  # Option B: probe teacher, reliability = aug x conf. Each line removes or swaps one piece.
    "abl_augonly": ["reliability.components=[aug]"],
    "abl_confonly": ["reliability.components=[conf]"],
    "abl_rawweights": ["reliability.transform=raw"],
    "abl_vote": ["reliability.agreement=vote"],
    "abl_norelhead": ["student.rel_head=false"],
    "abl_probeunbalanced": ["teacher.probe_balanced=false"],
    "abl_zeroshot": ["teacher.target=zeroshot"],  # zero-shot teacher, same aug x conf reliability
    "abl_zeroshot_promptaug": ["teacher.target=zeroshot", "reliability.components=[prompt,aug]"],  # original proposal
}


def jobs(a):
    """List of (run names that must exist, command)."""
    out = []
    if a.what in ("teacher", "all"):
        for f in a.fracs:
            for s in a.seeds:
                out.append(([f"teacher_zeroshot_lab{f}_s{s}", f"teacher_probe_lab{f}_s{s}"],
                            ["scripts/eval_teacher.py", "--config", a.config, "--set",
                             f"data.label_fraction={f}", f"data.split_seed={s}", f"train.seed={s}"]))
    if a.what in ("main", "all"):
        for arch in a.archs:
            for f in a.fracs:
                for s in a.seeds:
                    for m in a.methods:
                        out.append(([f"{m}_{arch}_lab{f}_s{s}"],
                                    ["-m", "rwkd.train", "--config", a.config, "--set", f"method={m}",
                                     f"student.arch={arch}", f"data.label_fraction={f}",
                                     f"data.split_seed={s}", f"train.seed={s}"] + a.extra))
    if a.what in ("ablation", "all"):
        for s in a.seeds:
            for tag, ov in ABLATIONS.items():
                out.append(([f"{tag}_rwkd_resnet18_lab5_s{s}"],
                            ["-m", "rwkd.train", "--config", a.config, "--set", "method=rwkd",
                             "student.arch=resnet18", "data.label_fraction=5", f"data.split_seed={s}",
                             f"train.seed={s}", f"tag={tag}"] + ov + a.extra))
    return out[a.shard::a.nshards]


def done(name):
    return os.path.exists(os.path.join("runs", name, "metrics.json"))


@contextmanager
def git_lock():
    try:
        import fcntl
    except ImportError:  # Windows: no parallel sync needed
        yield
        return
    with open(".git/grid_sync.lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def git_sync(names):
    files = [f for n in names for pat in ("metrics.json", "config.json", "preds_*.npz")
             for f in glob.glob(os.path.join("runs", n, pat))]
    if not files:
        return
    with git_lock():
        subprocess.run(["git", "add", "-f"] + files, check=False)
        subprocess.run(["git", "commit", "-q", "-m", f"results: {', '.join(names)}"], check=False)
        for attempt in range(4):
            subprocess.run(["git", "pull", "-q", "--rebase"], check=False)
            if subprocess.run(["git", "push", "-q"], check=False).returncode == 0:
                return
            time.sleep(2 ** (attempt + 1))
        print("WARNING: git push failed; results are still on disk", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="all", choices=["teacher", "main", "ablation", "all"])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--fracs", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--archs", nargs="+", default=["resnet18", "mobilenetv3_large_100"])
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--git-sync", action="store_true")
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    t0 = time.time()
    todo = [(names, cmd) for names, cmd in jobs(a) if not all(done(n) for n in names)]
    print(f"{len(todo)} runs to do (shard {a.shard}/{a.nshards})", flush=True)
    for k, (names, cmd) in enumerate(todo):
        if a.max_hours and (time.time() - t0) / 3600 > a.max_hours:
            print(f"time limit reached; {len(todo) - k} runs left for the next session", flush=True)
            break
        print(f">> [{k + 1}/{len(todo)}] {' '.join(cmd)}", flush=True)
        if subprocess.run([sys.executable] + cmd).returncode != 0:
            print(f"FAILED: {names}", flush=True)
            continue
        if a.git_sync:
            git_sync(names)
    print(f"grid finished/stopped after {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
