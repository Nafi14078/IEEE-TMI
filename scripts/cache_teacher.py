"""Compute teacher embeddings ONCE (A augmented views per image) + text embeddings for the
K prompt paraphrase sets. Every training run then only reads this file.

python scripts/cache_teacher.py --teacher hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
    --out cache/biomedclip.pt --views 8
"""
import argparse
import os
import sys

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rwkd.prompts import build_prompt_sets  # noqa: E402
from rwkd.teacher import Teacher  # noqa: E402
from rwkd.utils import get_device, load_config  # noqa: E402


class SeededViews(Dataset):
    """View 0 = deterministic resize+centre-crop; views 1..A-1 = seeded random augmentations,
    so the cache is exactly reproducible."""

    def __init__(self, paths, n_views, base, aug, seed=0):
        self.paths, self.n_views, self.base, self.aug, self.seed = paths, n_views, base, aug, seed

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        views = [self.base(img)]
        for v in range(1, self.n_views):
            torch.manual_seed(self.seed * 1_000_003 + i * 101 + v)
            views.append(self.aug(img))
        return torch.stack(views), i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32, help="images per batch (x views)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--text-only", action="store_true", help="only refresh text embeddings in an existing cache")
    a = ap.parse_args()
    cfg = load_config(a.config)
    device = get_device()
    df = pd.read_csv(cfg["data"]["csv"])
    classes = cfg["data"]["classes"]
    teacher = Teacher(a.teacher, device)

    prompt_sets = build_prompt_sets(classes)
    K, C = len(prompt_sets), len(classes)
    txt = teacher.encode_texts([p for s in prompt_sets for p in s]).view(K, C, -1).cpu()

    if a.text_only:
        cache = torch.load(a.out, map_location="cpu", weights_only=False)
        cache.update(txt_emb=txt, prompts=prompt_sets, classes=classes)
    else:
        base, aug = teacher.view_transforms()
        paths = [os.path.join(cfg["data"]["image_root"], p) for p in df["path"]]
        loader = DataLoader(SeededViews(paths, a.views, base, aug), batch_size=a.batch,
                            num_workers=a.workers, pin_memory=device.type == "cuda")
        img = None
        for bi, (x, i) in enumerate(loader):
            B, A = x.shape[:2]
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                f = teacher.encode_images(x.flatten(0, 1)).view(B, A, -1)
            if img is None:
                img = torch.zeros(len(paths), A, f.shape[-1], dtype=torch.float16)
            img[i] = f.half().cpu()
            if bi % 50 == 0:
                print(f"{(bi + 1) * a.batch}/{len(paths)}", flush=True)
        cache = {"image_ids": df["image_id"].tolist(), "img_emb": img, "txt_emb": txt,
                 "logit_scale": teacher.logit_scale, "classes": classes, "prompts": prompt_sets,
                 "teacher": a.teacher, "views": a.views}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save(cache, a.out)
    print(f"saved {a.out}: img_emb {tuple(cache['img_emb'].shape)} txt_emb {tuple(cache['txt_emb'].shape)}")


if __name__ == "__main__":
    main()
