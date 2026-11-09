# Team 2 — When to trust the teacher

**Reliability-weighted distillation of a medical vision-language foundation model into a compact skin-lesion classifier under label scarcity.**
Target: *IEEE Transactions on Medical Imaging* (TMI). Submit by **31 Dec 2026**.

## Idea in one paragraph
A frozen, openly licensed medical vision-language model (**BiomedCLIP**; optional second teacher: a dermatology-specific CLIP) labels every training image zero-shot. The labels are only partly right. For each image we measure **without labels** how much to trust the teacher:
- **prompt agreement**: do K paraphrased text prompts give the same answer?
- **augmentation agreement**: do A augmented views of the image give the same answer?

The compact **student** (ResNet-18 or MobileNetV3-Large) learns from the 1–10 % labelled images (cross-entropy) and from the teacher on all images, with each image's distillation term **weighted by that reliability**. A small extra head learns to **predict the reliability**, so the student can **abstain** at test time. Teacher embeddings are computed **once**, so every run fits a modest GPU.

## Layout
```
configs/default.yaml        every hyper-parameter (override with --set key=value)
src/rwkd/                   the library
  prompts.py                class names + K prompt paraphrase sets
  teacher.py                frozen teacher wrapper (open_clip / HF CLIP)
  reliability.py            ** the method **: teacher targets, label-free reliability, weights
  models.py                 student: backbone + class head + reliability head (+ projection)
  losses.py                 CE, (weighted) logit KD, embedding KD, pseudo-label, reliability BCE
  train.py                  one training run for any method -> runs/<name>/metrics.json
  metrics.py                bal. acc, macro-F1, AUROC, ECE, risk-coverage/AURC, bootstrap
scripts/
  download_data.sh          ISIC 2019 (+ instructions for PAD-UFES-20)
  prepare_splits.py         FREEZE splits, label fractions, class mapping
  preprocess_images.py      resize once to 256 px
  cache_teacher.py          teacher image (A views) + text (K prompts) embeddings, once
  check_reliability.py      does reliability predict teacher correctness? (val)
  eval_teacher.py           baselines: teacher zero-shot, teacher linear probe
  run_grid.sh               all experiments (main / ablation / teacher)
  make_tables.py            LaTeX/CSV tables + paired bootstrap tests
  make_figures.py           figures
tests/test_core.py          unit tests (pytest)
docs/                       PLAN, kickoff decisions, data & licences, novelty search, review checklist
paper/                      IEEE TMI LaTeX skeleton (headings only — the team writes the text)
```

## Methods compared (all share data, student, schedule)
| `method` | what it is |
|---|---|
| `supervised` | labelled images only |
| `kd_logit` | standard logit distillation, every image weight 1 |
| `kd_embed` | student regresses the teacher image embedding (cosine) |
| `pseudo_label` | teacher argmax as hard label when teacher confidence ≥ τ |
| `teacher_zeroshot`, `teacher_probe` | the teacher itself (`scripts/eval_teacher.py`) |
| **`rwkd`** | **ours**: reliability-weighted logit KD + reliability head |

Evaluation: in-distribution test (ISIC 2019, lesion-grouped split) and **external** smartphone set (PAD-UFES-20); 1 / 5 / 10 % labels; 3 seeds (each seed also re-draws the labelled subset); balanced accuracy, macro-F1, macro-AUROC, ECE, and selective prediction (risk–coverage, AURC).

## Quick start
```bash
pip install -r requirements.txt && pip install -e .
pytest -q tests                                   # sanity

bash scripts/download_data.sh                     # + PAD-UFES-20 by hand (see message)
python scripts/prepare_splits.py \
  --isic-gt data/raw/isic2019/ISIC_2019_Training_GroundTruth.csv \
  --isic-meta data/raw/isic2019/ISIC_2019_Training_Metadata.csv \
  --isic-images data/raw/isic2019/ISIC_2019_Training_Input \
  --pad-meta data/raw/pad_ufes20/metadata.csv --pad-images data/raw/pad_ufes20/images
python scripts/preprocess_images.py
python scripts/cache_teacher.py --out cache/biomedclip.pt --views 8 \
  --teacher hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224

python scripts/check_reliability.py --set data.label_fraction=5    # task: check reliability on val
python scripts/eval_teacher.py --set data.label_fraction=5          # teacher baselines
python -m rwkd.train --set method=rwkd data.label_fraction=5 train.seed=0 data.split_seed=0

bash scripts/run_grid.sh all                       # everything
python scripts/make_tables.py && python scripts/make_figures.py
```
Commit `data/splits/splits.csv` once frozen; never commit images, caches or checkpoints.

See **docs/PLAN.md** for the step-by-step plan mapped to the portal tasks.
