# Kickoff decisions (fill in, date and sign; change only with mentor agreement)

| # | Decision | Proposed default (in code) | Agreed value | Date / by |
|---|---|---|---|---|
| 1 | Author list and order | — | | |
| 2 | Roles | — | | |
| 3 | In-distribution dataset | ISIC 2019 training set | | |
| 4 | External dataset | PAD-UFES-20 (smartphone) | | |
| 5 | Classes | MEL, NV, BCC, AK, BKL↔SEK, SCC (shared by both) | | |
| 6 | Splits | 70/10/20, grouped by lesion_id, split seed 0 | | |
| 7 | Label fractions / seeds | 1, 5, 10 % × seeds 0, 1, 2 (nested subsets) | | |
| 8 | Teacher(s) | BiomedCLIP; optional derm CLIP | | |
| 9 | Prompts / views | 8 paraphrase sets (`prompts.py`) / 8 views | | |
| 10 | Reliability | prompt × aug agreement (JSD), rank → weights | | |
| 11 | Students | ResNet-18, MobileNetV3-Large (ImageNet init) | | |
| 12 | Baselines | supervised, logit KD, embedding KD, pseudo-label, teacher zero-shot, teacher probe | | |
| 13 | Model selection | last checkpoint, hyper-params tuned on val at 5 % only | | |
| 14 | Primary metric | balanced accuracy; also macro-F1, AUROC, ECE, AURC | | |
| 15 | Statistics | mean ± std over 3 seeds, paired bootstrap vs each baseline | | |
