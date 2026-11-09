import timm
import torch
import torch.nn as nn


class Student(nn.Module):
    """Compact CNN with a class head, an optional reliability head (for abstention) and an
    optional projection to the teacher embedding space (embedding-distillation baseline)."""

    def __init__(self, arch, num_classes, pretrained=True, rel_head=True, rel_detach=False, proj_dim=None):
        super().__init__()
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0)
        with torch.no_grad():  # feature width after pooling/head conv (differs from num_features for MobileNetV3)
            was = self.backbone.training
            f = self.backbone.eval()(torch.zeros(1, 3, 64, 64)).shape[-1]
            self.backbone.train(was)
        self.cls = nn.Linear(f, num_classes)
        self.rel = nn.Sequential(nn.Linear(f, 128), nn.ReLU(inplace=True), nn.Linear(128, 1)) if rel_head else None
        self.rel_detach = rel_detach
        self.proj = nn.Linear(f, proj_dim) if proj_dim else None

    def forward(self, x):
        h = self.backbone(x)
        out = {"logits": self.cls(h)}
        if self.rel is not None:
            out["rel_logit"] = self.rel(h.detach() if self.rel_detach else h).squeeze(-1)
        if self.proj is not None:
            out["proj"] = self.proj(h)
        return out
