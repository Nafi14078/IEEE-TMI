import torch
import torch.nn.functional as F


def soften(p, T):
    """Temperature-soften a probability vector: p^(1/T), renormalised."""
    return torch.softmax(torch.log(p.clamp_min(1e-8)) / T, -1)


def weighted_kd(student_logits, teacher_p, weights, T):
    """sum_i w_i KL(p_T,i || softmax(z_i / T)) / B * T^2. Uniform weights = standard logit KD."""
    log_ps = F.log_softmax(student_logits.float() / T, -1)
    kl = (teacher_p * (torch.log(teacher_p.clamp_min(1e-8)) - log_ps)).sum(-1)
    return (weights * kl).mean() * T * T


def embed_loss(proj, teacher_emb):
    return (1.0 - F.cosine_similarity(proj.float(), teacher_emb, dim=-1)).mean()


def pseudo_label_loss(student_logits, teacher_p, threshold):
    conf, target = teacher_p.max(-1)
    mask = (conf >= threshold).float()
    ce = F.cross_entropy(student_logits.float(), target, reduction="none")
    return (mask * ce).mean(), mask.mean()


def reliability_loss(rel_logit, target):
    return F.binary_cross_entropy_with_logits(rel_logit.float(), target)
