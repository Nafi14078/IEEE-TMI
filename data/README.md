# data/
- `raw/` — downloads (git-ignored). `raw/isic2019/`, `raw/pad_ufes20/` (metadata.csv + images/)
- `processed/` — images resized to short side 256 px by `scripts/preprocess_images.py` (git-ignored)
- `splits/splits.csv` — **the frozen protocol, committed.** One row per image:
  `image_id, dataset, src_path, path, label_name, label, group, split, lab{1,5,10}_s{0,1,2}`
  - `split`: train / val / test (ISIC 2019, grouped by `lesion_id`) or external (PAD-UFES-20)
  - `labN_sS = 1` → this training image keeps its label at N % labels, seed S (subsets are nested 1 ⊂ 5 ⊂ 10)
