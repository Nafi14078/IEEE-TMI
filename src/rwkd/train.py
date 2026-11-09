"""Train one student (any method) and evaluate it on val / test / external.

python -m rwkd.train --config configs/default.yaml --set method=rwkd data.label_fraction=5 train.seed=0
"""
import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import ImageTable, infinite, load_splits, make_loader, student_transform
from .losses import embed_loss, pseudo_label_loss, reliability_loss, soften, weighted_kd
from .metrics import classification_metrics, selective_metrics
from .models import Student
from .reliability import build_teacher_targets
from .utils import get_device, load_config, run_name, save_json, set_seed

METHODS = ("supervised", "kd_logit", "kd_embed", "pseudo_label", "rwkd")


def class_weights(y_lab, C):
    counts = torch.bincount(y_lab, minlength=C).float().clamp_min(1)
    w = counts.sum() / (C * counts)
    return w / w.mean()


@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    probs, rel, idxs = [], [], []
    for x, idx in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            out = model(x.to(device, non_blocking=True))
        probs.append(torch.softmax(out["logits"].float(), -1).cpu())
        if "rel_logit" in out:
            rel.append(torch.sigmoid(out["rel_logit"].float()).cpu())
        idxs.append(idx)
    model.train()
    return (torch.cat(probs).numpy(), torch.cat(rel).numpy() if rel else None, torch.cat(idxs).numpy())


def evaluate(probs, rel, y):
    m = classification_metrics(probs, y)
    correct = probs.argmax(1) == y
    m["selective_msp"] = selective_metrics(probs.max(1), correct)
    if rel is not None:
        m["selective_rel"] = selective_metrics(rel, correct)
    return m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. method=rwkd train.seed=1")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    method = cfg["method"]
    assert method in METHODS, method
    seed = int(cfg["train"]["seed"])
    set_seed(seed)
    device = get_device()
    amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    out_dir = os.path.join(cfg.get("out_dir", "runs"), run_name(cfg))
    os.makedirs(out_dir, exist_ok=True)
    save_json(cfg, os.path.join(out_dir, "config.json"))

    df, idx, lab_idx = load_splits(cfg)
    classes = cfg["data"]["classes"]
    C = len(classes)
    y = torch.as_tensor(df["label"].values, dtype=torch.long)
    print(f"[{run_name(cfg)}] train={len(idx['train'])} labelled={len(lab_idx)} "
          f"val={len(idx['val'])} test={len(idx['test'])} external={len(idx['external'])}")

    tt = None
    if method != "supervised":
        tt = build_teacher_targets(cfg, df, lab_idx, idx["train"], y)
        tp_train = tt["pbar"][idx["train"]].argmax(1) == y[idx["train"]]
        print(f"teacher acc on train pool (hidden labels, info only) = {tp_train.float().mean():.3f}")

    use_rel = method == "rwkd" and cfg["student"].get("rel_head", True)
    model = Student(cfg["student"]["arch"], C, pretrained=cfg["student"].get("pretrained", True),
                    rel_head=use_rel, rel_detach=cfg["student"].get("rel_detach", False),
                    proj_dim=tt["emb"].shape[1] if method == "kd_embed" else None).to(device)

    tcfg, lcfg, dcfg = cfg["train"], cfg["loss"], cfg["data"]
    steps = int(tcfg["steps"])
    nw = int(dcfg.get("num_workers", 4))
    tr_tf = student_transform(dcfg["img_size"], True)
    ev_tf = student_transform(dcfg["img_size"], False)
    lab_iter = infinite(make_loader(ImageTable(df, dcfg["image_root"], lab_idx, tr_tf),
                                    min(int(tcfg["batch_labelled"]), len(lab_idx)), True, nw, drop_last=True))
    unl_iter = None
    if method != "supervised":
        unl_iter = infinite(make_loader(ImageTable(df, dcfg["image_root"], idx["train"], tr_tf),
                                        int(tcfg["batch_unlabelled"]), True, nw, drop_last=True))

    opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))
    warm = int(tcfg.get("warmup_steps", 0))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(warm, 1) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps - warm, 1))))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    cw = class_weights(y[lab_idx], C).to(device) if lcfg.get("ce_class_weights", True) else None
    T_kd = float(cfg["teacher"].get("kd_temperature", 1.0))
    lam, mu = float(lcfg.get("lambda_kd", 1.0)), float(lcfg.get("mu_rel", 0.5))
    ramp_steps = max(int(float(lcfg.get("rampup_frac", 0.1)) * steps), 1)
    if tt is not None:
        pbar, wts, emb = tt["pbar"].to(device), tt["w"].float().to(device), tt["emb"].to(device)

    t0 = time.time()
    model.train()
    for step in range(steps):
        xl, il = next(lab_iter)
        yl = y[il].to(device)
        xs = [xl]
        if unl_iter is not None:
            xu, iu = next(unl_iter)
            xs.append(xu)
            iu = iu.to(device)
        x = torch.cat(xs).to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            out = model(x)
        nl = len(xl)
        logits = out["logits"].float()
        loss = F.cross_entropy(logits[:nl], yl, weight=cw)
        logs = {"ce": loss.item()}
        if unl_iter is not None:
            ramp = min(1.0, (step + 1) / ramp_steps)
            zu = logits[nl:]
            if method in ("kd_logit", "rwkd"):
                w = wts[iu] if method == "rwkd" else torch.ones(len(iu), device=device)
                kd = weighted_kd(zu, soften(pbar[iu], T_kd), w, T_kd)
                loss = loss + lam * ramp * kd
                logs["kd"] = kd.item()
                if use_rel:
                    rl = reliability_loss(out["rel_logit"][nl:], wts[iu])
                    loss = loss + mu * ramp * rl
                    logs["rel"] = rl.item()
            elif method == "kd_embed":
                el = embed_loss(out["proj"][nl:], emb[iu])
                loss = loss + lam * ramp * el
                logs["emb"] = el.item()
            elif method == "pseudo_label":
                pl, frac = pseudo_label_loss(zu, pbar[iu], float(lcfg.get("pl_threshold", 0.7)))
                loss = loss + lam * ramp * pl
                logs.update(pl=pl.item(), pl_mask=frac.item())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if (step + 1) % int(tcfg.get("log_every", 100)) == 0 or step == 0:
            print(f"step {step + 1}/{steps} loss {loss.item():.4f} " +
                  " ".join(f"{k} {v:.4f}" for k, v in logs.items()) + f" | {time.time() - t0:.0f}s", flush=True)

    # Model selection: the LAST checkpoint is used (no early stopping on a large labelled val set,
    # which would be unrealistic under label scarcity). Val is reported for monitoring only.
    if tcfg.get("save_ckpt", True):
        torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    results = {"run": run_name(cfg), "method": method, "tag": cfg.get("tag", ""),
               "arch": cfg["student"]["arch"], "label_fraction": dcfg["label_fraction"], "seed": seed,
               "n_labelled": int(len(lab_idx)), "classes": classes, "train_seconds": time.time() - t0}
    if tt is not None:
        results["teacher_calibration"] = tt["calibration"]
    bs = int(tcfg.get("batch_eval", 128))
    for split in ("val", "test", "external"):
        if len(idx[split]) == 0:
            continue
        probs, rel, ids = predict(model, make_loader(ImageTable(df, dcfg["image_root"], idx[split], ev_tf),
                                                     bs, False, nw), device, amp)
        yy = y[ids].numpy()
        np.savez_compressed(os.path.join(out_dir, f"preds_{split}.npz"), probs=probs, y=yy, idx=ids,
                            **({"rel": rel} if rel is not None else {}))
        results[split] = evaluate(probs, rel, yy)
        print(f"{split}: bal_acc {results[split]['bal_acc']:.4f} auroc {results[split]['auroc']:.4f} "
              f"ece {results[split]['ece']:.4f}")
    save_json(results, os.path.join(out_dir, "metrics.json"))
    return results


if __name__ == "__main__":
    main()
