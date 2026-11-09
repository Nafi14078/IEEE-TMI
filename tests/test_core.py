import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rwkd.losses import soften, weighted_kd  # noqa: E402
from rwkd.metrics import classification_metrics, selective_metrics  # noqa: E402
from rwkd.reliability import agreement, combine, reliability_components, to_weights  # noqa: E402


def test_agreement_extremes():
    same = torch.tensor([[0.9, 0.05, 0.05]]).repeat(4, 1).unsqueeze(0)  # [1, 4, 3]
    diff = torch.eye(3)[[0, 1, 2, 0]].unsqueeze(0).clamp(1e-6)
    diff = diff / diff.sum(-1, keepdim=True)
    assert agreement(same, 1).item() > 0.999
    assert agreement(diff, 1).item() < 0.2
    assert agreement(same, 1, "vote").item() == 1.0
    assert math.isclose(agreement(diff, 1, "vote").item(), 0.5)


def test_reliability_ranks_consistent_teacher_higher():
    N, A, K, C = 2, 4, 5, 6
    p = torch.full((N, A, K, C), 1e-3)
    p[0, :, :, 2] = 1.0                       # image 0: every view/prompt agrees
    p[1] = torch.softmax(torch.randn(A, K, C) * 5, -1)  # image 1: noisy
    p = p / p.sum(-1, keepdim=True)
    pbar, comps = reliability_components([p])
    r = combine(comps, ["prompt", "aug"])
    assert r[0] > r[1]
    assert pbar.shape == (N, C) and torch.allclose(pbar.sum(-1), torch.ones(N))
    w = to_weights(r, np.arange(N), "rank")
    assert w[0] == 1.0 and w[1] == 0.0


def test_weighted_kd_zero_weight_and_match():
    z = torch.randn(8, 6)
    p = torch.softmax(z, -1)
    assert weighted_kd(z, p, torch.ones(8), 1.0).abs() < 1e-5        # matching -> 0
    q = soften(torch.softmax(torch.randn(8, 6), -1), 2.0)
    assert weighted_kd(z, q, torch.zeros(8), 2.0) == 0               # weight 0 -> ignored


def test_metrics():
    y = np.array([0, 1, 2, 0, 1, 2])
    probs = np.eye(3)[y] * 0.9 + 0.1 / 3
    m = classification_metrics(probs, y)
    assert m["bal_acc"] == 1.0 and m["auroc"] == 1.0
    s = selective_metrics(np.linspace(1, 0, 6), np.array([1, 1, 1, 0, 0, 0], bool))
    assert s["risk@50"] == 0.0
