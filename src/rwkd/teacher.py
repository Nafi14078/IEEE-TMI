"""Frozen vision-language teachers. Only used offline by scripts/cache_teacher.py."""
import torch
import torch.nn.functional as F
from torchvision import transforms as T

from .data import RandomRot90


class Teacher:
    """Wraps an open_clip model (spec 'hf-hub:<repo>') or a HF transformers CLIP model
    (spec 'hf-transformers:<repo>'). Outputs L2-normalised embeddings."""

    def __init__(self, spec: str, device):
        self.spec, self.device = spec, device
        if spec.startswith("hf-transformers:"):
            from transformers import CLIPModel, CLIPProcessor
            repo = spec.split(":", 1)[1]
            self.kind = "transformers"
            self.model = CLIPModel.from_pretrained(repo).to(device).eval()
            proc = CLIPProcessor.from_pretrained(repo)
            self.tokenizer = proc.tokenizer
            ip = proc.image_processor
            self.mean, self.std = tuple(ip.image_mean), tuple(ip.image_std)
            cs = ip.crop_size
            self.size = cs["height"] if isinstance(cs, dict) else int(cs)
        else:
            import open_clip
            self.kind = "open_clip"
            self.model, preprocess = open_clip.create_model_from_pretrained(spec)
            self.model = self.model.to(device).eval()
            self.tokenizer = open_clip.get_tokenizer(spec)
            self.mean, self.std, self.size = (0.481, 0.458, 0.408), (0.269, 0.261, 0.276), 224
            for t in getattr(preprocess, "transforms", []):
                if isinstance(t, T.Normalize):
                    self.mean, self.std = tuple(t.mean), tuple(t.std)
                if isinstance(t, T.CenterCrop):
                    self.size = int(t.size[0])

    @property
    def logit_scale(self) -> float:
        return float(self.model.logit_scale.exp().item())

    @torch.no_grad()
    def encode_images(self, x):
        x = x.to(self.device, non_blocking=True)
        f = self.model.get_image_features(pixel_values=x) if self.kind == "transformers" \
            else self.model.encode_image(x)
        return F.normalize(f.float(), dim=-1)

    @torch.no_grad()
    def encode_texts(self, texts):
        if self.kind == "transformers":
            tok = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            f = self.model.get_text_features(**tok)
        else:
            f = self.model.encode_text(self.tokenizer(texts).to(self.device))
        return F.normalize(f.float(), dim=-1)

    def view_transforms(self):
        """(deterministic base view, random augmented view) at the teacher's resolution."""
        norm = T.Normalize(self.mean, self.std)
        base = T.Compose([T.Resize(self.size, interpolation=T.InterpolationMode.BICUBIC),
                          T.CenterCrop(self.size), T.ToTensor(), norm])
        aug = T.Compose([RandomRot90(),
                         T.RandomResizedCrop(self.size, scale=(0.7, 1.0),
                                             interpolation=T.InterpolationMode.BICUBIC),
                         T.RandomHorizontalFlip(), T.ColorJitter(0.1, 0.1, 0.1, 0.0),
                         T.ToTensor(), norm])
        return base, aug
