import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score


def macro_auroc(probs, y):
    aucs = [roc_auc_score(y == c, probs[:, c]) for c in range(probs.shape[1]) if 0 < (y == c).sum() < len(y)]
    return float(np.mean(aucs)) if aucs else float("nan")


def ece(probs, y, n_bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(out)


def classification_metrics(probs, y):
    C = probs.shape[1]
    pred = probs.argmax(1)
    labels = list(range(C))
    return {
        "bal_acc": float(balanced_accuracy_score(y, pred)),
        "acc": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=labels, zero_division=0)),
        "auroc": macro_auroc(probs, y),
        "ece": ece(probs, y),
        "nll": float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None)).mean()),
        "recall_per_class": recall_score(y, pred, average=None, labels=labels, zero_division=0).tolist(),
        "n": int(len(y)),
    }


def risk_coverage(score, correct):
    """Keep the most confident first. Returns (coverage, selective risk) arrays."""
    order = np.argsort(-score, kind="stable")
    err = 1.0 - correct[order].astype(float)
    k = np.arange(1, len(err) + 1)
    return k / len(err), np.cumsum(err) / k


def selective_metrics(score, correct, coverages=(0.5, 0.8, 0.9)):
    cov, risk = risk_coverage(score, correct)
    out = {"aurc": float(risk.mean())}
    for c in coverages:
        out[f"risk@{int(c * 100)}"] = float(risk[max(int(np.ceil(c * len(risk))) - 1, 0)])
    return out


def bootstrap_ci(metric_fn, y, pred_or_probs, n=1000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    vals = [metric_fn(y[i], pred_or_probs[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def paired_bootstrap(y, pred_a, pred_b, metric_fn=balanced_accuracy_score, n=2000, seed=0):
    """Difference metric(a) - metric(b) on the same images: point estimate, 95% CI, one-sided p (a<=b)."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    for j in range(n):
        i = rng.integers(0, len(y), len(y))
        diffs[j] = metric_fn(y[i], pred_a[i]) - metric_fn(y[i], pred_b[i])
    return {"diff": float(metric_fn(y, pred_a) - metric_fn(y, pred_b)),
            "ci": [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))],
            "p_one_sided": float((diffs <= 0).mean())}
