# Plan — task by task (portal tasks, in order)

Today is 9 Nov 2026. Items 2, 3, 6, 7, 8 are **overdue**: close them this week, before any training.
Rough compute: teacher cache ≈ 30 min on one GPU; one student run (6 000 steps, ResNet-18, AMP) ≈ 15–30 min;
full grid (90 main + 24 ablation runs) ≈ 30–50 GPU-hours. Start the grid by 24 Nov at the latest.

---
### 1. Kickoff — DONE (24 Oct)

### 2. Get the author's decision on the idea and the author list — overdue (27 Oct)
- PI (Abhijit Kumar Ghosh) confirms the idea as written and the author order; record it in `docs/KICKOFF_DECISIONS.md`.
- Agree roles: data/splits owner, teacher/reliability owner, training owner, writing lead, reference checker.

### 6. Download the open datasets and confirm every licence — overdue (30 Oct)
- `bash scripts/download_data.sh`, PAD-UFES-20 by hand. Fill the table in `docs/DATA_AND_LICENCES.md` (licence, URL, date, who checked).

### 7. Confirm teacher model weights and licences — overdue (31 Oct)
- BiomedCLIP (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`): read the model card, record licence + intended-use terms.
- Optional dermatology teacher (e.g. a dermatology CLIP such as MONET): only if weights are public **and** the licence allows research use **and** it has a text encoder (needed for prompts). Otherwise drop it — it is optional.
- Note possible **data overlap**: BiomedCLIP was trained on figures from PubMed Central papers, which may contain ISIC images. Mention as a limitation.

### 8. Re-run and extend the novelty search — overdue (3 Nov)
- Use `docs/NOVELTY_SEARCH.md`. Goal: make sure nobody already published "label-free reliability from prompt + augmentation agreement as KD weights from a medical VLM, with a learned abstention head". Adjust the claimed contribution if needed.

### 3. Freeze splits, label fractions and the evaluation protocol — overdue (6 Nov)
- `python scripts/prepare_splits.py ...` → `data/splits/splits.csv`. Commit it. **Never change it afterwards.**
- Protocol (fixed): 6 shared classes MEL, NV, BCC, AK, BKL(↔SEK), SCC; ISIC 2019 70/10/20 lesion-grouped; PAD-UFES-20 external test only; 1/5/10 % labels × 3 seeds; last checkpoint (no early stopping on a big labelled val set); metrics in `metrics.py`.
- Write the protocol into `KICKOFF_DECISIONS.md` *before* seeing any test number.

### 9. Write the loading code and cache teacher outputs (10 Nov)
- Code is ready: `preprocess_images.py`, `cache_teacher.py`. Run them; check `cache/biomedclip.pt` shapes `[N, 8, 512]` and `[8, 6, 512]`.

### 4. Implement the reliability score and check it on validation data (14 Nov)
- Implemented in `src/rwkd/reliability.py`. Run `scripts/check_reliability.py` for 1/5/10 %.
- **Go / no-go**: reliability must predict teacher correctness (AUROC clearly > 0.5, accuracy rising across quintiles, `prompt*aug` better than `conf` alone). If it does not, discuss with the mentor *before* training (adjust prompts in `prompts.py`, views, calibration).
- Only touch prompts / views using VAL, never test.

### 10. Train the baselines (24 Nov)
- `bash scripts/run_grid.sh teacher` then the non-`rwkd` methods of `run_grid.sh main`.
- Light tuning (λ, τ, lr) on VAL only, at 5 % labels, then freeze for all settings.

### 11. Train the reliability-weighted student and its ablations (1 Dec)
- `rwkd` rows of `run_grid.sh main` + `bash scripts/run_grid.sh ablation`.

### 12. Run the analysis: main comparison, external test, selective prediction (8 Dec)
- `make_tables.py` (mean ± std over seeds + paired bootstrap), risk–coverage curves, per-class recall, does reliability-head abstention beat max-softmax abstention?
- Report honestly where the method does *not* help (e.g. 10 % labels, external set).

### 13. Make tables and figures from script outputs (12 Dec)
- Every number in the paper comes from `results/` produced by scripts. No hand-typed numbers.
- Suggested: Fig. 1 method diagram; Fig. 2 reliability vs teacher accuracy (val); Fig. 3 BAcc vs label fraction (test + external); Fig. 4 risk–coverage; Table I main; Table II ablations; Table III teacher baselines.

### 14. Write the manuscript in the team's own words (19 Dec)
- `paper/main.tex` contains only the IEEE structure and section goals. **The team writes the text.**

### 15. Verify every reference and cited claim (22 Dec)
- Open every paper you cite; check authors/year/venue/DOI and that it says what you claim. `docs/REVIEW_CHECKLIST.md`.

### 16. Internal review by someone outside the team (26 Dec)
### 17. Format for TMI: official template, page limit, AI-use disclosure (29 Dec)
- Download the current TMI template and author instructions from the IEEE Author Center / TMI site; check page limit, figure rules, and the IEEE policy on disclosing AI-generated content (fill `paper/AI_USE_DISCLOSURE.md`).

### 5. Submit to IEEE TMI (31 Dec)

---
## Out of scope (state as future work)
Federated distillation across sites.
