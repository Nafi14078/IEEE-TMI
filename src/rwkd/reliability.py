"""Label-free teacher reliability and teacher targets (the core of the method).

Teacher probabilities have shape [N, A, K, C]:
  N images, A augmented views, K prompt paraphrase sets, C classes.
Reliability components (all in [0, 1], higher = more reliable, no labels used):
  prompt  - agreement across the K prompt paraphrase sets
  aug     - agreement across the A image augmentations
  teacher - agreement between teachers (only if a second teacher is cached)
  conf    - 1 - normalised entropy of the averaged prediction (ablation / baseline)
"""
import math
import warnings

import numpy as np
import torch
import torch.nn.functional as F


def load_cache(path, image_ids):
    c = torch.load(path, map_location="cpu", weights_only=False)
    if list(c["image_ids"]) != list(image_ids):
        raise ValueError(f"{path}: image order differs from the splits CSV; re-run cache_teacher.py")
    return c


def entropy(p, dim=-1):
    return -(p * torch.log(p.clamp_min(1e-12))).sum(dim)


def gjsd(p, dim):
    """Generalised Jensen-Shannon divergence among the distributions stacked along `dim`."""
    return entropy(p.mean(dim)) - entropy(p).mean(dim)


def agreement(p, dim, kind="jsd"):
    """Agreement in [0,1] among distributions stacked along `dim` (last dim = classes)."""
    C = p.shape[-1]
    if kind == "jsd":
        return (1.0 - gjsd(p, dim) / math.log(C)).clamp(0.0, 1.0)
    if kind == "vote":
        consensus = p.mean(dim).argmax(-1, keepdim=True)
        return (p.argmax(-1) == consensus).float().mean(dim)
    raise ValueError(kind)


def zeroshot_logits(cache, temperature=1.0, bias=None):
    img = cache["img_emb"].float()  # [N, A, D]
    txt = cache["txt_emb"].float()  # [K, C, D]
    logits = cache["logit_scale"] * torch.einsum("nad,kcd->nakc", img, txt) / temperature
    return logits + bias if bias is not None else logits


def fit_calibration(logits, y, use_bias=False, iters=200, balanced=True):
    """Fit a temperature (and optional per-class bias) on the *labelled* images only.
    balanced=True weights every class equally (matches balanced accuracy); without it the
    bias learns the class prior and collapses onto the majority class (NV)."""
    logT = torch.zeros((), requires_grad=True)
    b = torch.zeros(logits.shape[-1], requires_grad=use_bias)
    params = [logT] + ([b] if use_bias else [])
    C = logits.shape[-1]
    w = None
    if balanced:
        counts = torch.bincount(y, minlength=C).float().clamp_min(1)
        w = counts.sum() / (C * counts)
    opt = torch.optim.LBFGS(params, lr=0.1, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        p = torch.softmax(logits / logT.exp() + b, -1).mean((1, 2))
        loss = F.nll_loss(torch.log(p.clamp_min(1e-8)), y, weight=w) + 1e-3 * (b ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return float(logT.detach().exp()), (b.detach() if use_bias else None)


def probe_probs(cache, labelled_idx, y, C, balanced=True, C_reg="cv"):
    """Teacher's own linear probe: standardised view-0 embeddings of the labelled images ->
    logistic regression (class-balanced; C chosen by stratified CV on the labelled images only),
    applied to every view -> [N, A, 1, C]."""
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    emb = cache["img_emb"].float().numpy()
    N, A, D = emb.shape
    X, yl = emb[labelled_idx, 0], y[labelled_idx].numpy()
    cw = "balanced" if balanced else None
    if C_reg == "cv":
        folds = int(min(5, np.bincount(yl).min()))
        if folds >= 2:
            lr = LogisticRegressionCV(Cs=[0.001, 0.01, 0.1, 1.0, 10.0], cv=StratifiedKFold(folds, shuffle=True, random_state=0),
                                      scoring="balanced_accuracy", class_weight=cw, max_iter=5000)
        else:
            lr = LogisticRegression(C=0.1, class_weight=cw, max_iter=5000)
    else:
        lr = LogisticRegression(C=float(C_reg), class_weight=cw, max_iter=5000)
    clf = make_pipeline(StandardScaler(), lr).fit(X, yl)
    out = np.zeros((N * A, C), dtype=np.float32)
    out[:, clf.classes_] = clf.predict_proba(emb.reshape(N * A, D))
    chosen = getattr(clf[-1], "C_", [getattr(clf[-1], "C", None)])
    return torch.from_numpy(out).view(N, A, 1, C), {"probe_C": float(np.ravel(chosen)[0])}


def teacher_probs(cache, mode, labelled_idx, y, tcfg):
    C = cache["txt_emb"].shape[1]
    if mode == "zeroshot":
        cal = tcfg.get("calibrate", "none")
        if cal == "none":
            return torch.softmax(zeroshot_logits(cache), -1), {"temperature": 1.0}
        logits = zeroshot_logits(cache)
        temp, bias = fit_calibration(logits[labelled_idx], y[labelled_idx],
                                     use_bias=(cal == "temperature_bias"),
                                     balanced=tcfg.get("calibrate_balanced", True))
        return torch.softmax(logits / temp + (bias if bias is not None else 0.0), -1), \
            {"temperature": temp, "bias": None if bias is None else bias.tolist()}
    if mode == "probe":
        return probe_probs(cache, labelled_idx, y, C, balanced=tcfg.get("probe_balanced", True),
                           C_reg=tcfg.get("probe_C", "cv"))
    raise ValueError(mode)


def reliability_components(probs_list, kind="jsd"):
    """probs_list: one [N, A, K, C] tensor per teacher. Returns (p_bar [N, C], {name: [N]})."""
    pbars, r_prompt, r_aug = [], [], []
    for p in probs_list:
        pbars.append(p.mean((1, 2)))
        if p.shape[2] > 1:
            r_prompt.append(agreement(p.mean(1), 1, kind))
        if p.shape[1] > 1:
            r_aug.append(agreement(p.mean(2), 1, kind))
    pbar = torch.stack(pbars, 1).mean(1)
    comps = {}
    if r_prompt:
        comps["prompt"] = torch.stack(r_prompt).mean(0)
    if r_aug:
        comps["aug"] = torch.stack(r_aug).mean(0)
    if len(pbars) > 1:
        comps["teacher"] = agreement(torch.stack(pbars, 1), 1, kind)
    comps["conf"] = 1.0 - entropy(pbar) / math.log(pbar.shape[-1])
    return pbar, comps


def combine(comps, names, how="product"):
    avail = [n for n in names if n in comps]
    missing = set(names) - set(avail)
    if missing:
        warnings.warn(f"reliability components {sorted(missing)} unavailable; using {avail}")
    if not avail:
        raise ValueError("no reliability component available")
    stack = torch.stack([comps[n] for n in avail])
    if how == "product":
        return stack.prod(0)
    if how == "geomean":
        return stack.clamp_min(1e-8).log().mean(0).exp()
    if how == "mean":
        return stack.mean(0)
    raise ValueError(how)


def to_weights(r, pool_idx, transform="rank", power=1.0):
    """Turn raw reliability into per-image weights in [0,1]; 'rank' = percentile within the pool."""
    w = r.clone()
    if transform == "rank":
        pool = torch.as_tensor(pool_idx)
        order = r[pool].argsort().argsort().float()
        w[pool] = order / max(len(pool) - 1, 1)
    elif transform == "power":
        w = r.clamp(0, 1) ** power
    elif transform != "raw":
        raise ValueError(transform)
    return w


def build_teacher_targets(cfg, df, labelled_idx, train_idx, y):
    """Everything the student needs from the (cached) teacher(s), for every row of the CSV."""
    tcfg, rcfg = cfg["teacher"], cfg["reliability"]
    caches = [load_cache(p, df["image_id"].values) for p in [tcfg["cache"]] + list(tcfg.get("extra_caches") or [])]
    probs, info = [], []
    for c in caches:
        p, i = teacher_probs(c, tcfg.get("target", "zeroshot"), labelled_idx, y, tcfg)
        probs.append(p)
        info.append(i)
    pbar, comps = reliability_components(probs, rcfg.get("agreement", "jsd"))
    r = combine(comps, rcfg.get("components", ["prompt", "aug"]), rcfg.get("combine", "product"))
    w = to_weights(r, train_idx, rcfg.get("transform", "rank"), float(rcfg.get("power", 1.0)))
    emb = F.normalize(caches[0]["img_emb"].float().mean(1), dim=-1)
    return {"pbar": pbar, "r": r, "w": w, "comps": comps, "emb": emb, "calibration": info}
