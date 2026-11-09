# Running everything on Kaggle (free T4 GPUs)

Kaggle gives about **30 GPU-hours per week**, with sessions of up to **12 h**. The project is split into two notebooks:

| Notebook | What it does | Time (estimate) |
|---|---|---|
| `kaggle/01_prepare_and_cache.ipynb` | finds the datasets, freezes splits, resizes images, caches BiomedCLIP, reliability check, teacher baselines | 1.5–3 h, once |
| `kaggle/02_train.ipynb` | tuning, then the training grid on **both T4s in parallel**; pushes every finished run to GitHub; resumes automatically | ~132 runs ≈ 20–40 h → 2–4 sessions |

## One-time setup
1. Kaggle account → *Settings* → **verify your phone number** (needed for GPU + Internet).
2. GitHub → *Settings → Developer settings → Fine-grained tokens* → new token for repository
   `Nafi14078/IEEE-TMI`, permission **Contents: Read and write**. Copy it.
3. On Kaggle, create **Notebook 01**: *File → Import Notebook* → upload `kaggle/01_prepare_and_cache.ipynb`.
   - *Settings*: Accelerator **GPU T4 x2**, Internet **On**.
   - *Add Input → Datasets*: search **"ISIC 2019"** and add a copy that contains the ~25 000 training images;
     search **"PAD-UFES-20"** and add a copy that contains `metadata.csv` and the `PAT_*.png` images.
     (The notebook finds the files automatically and stops with a clear message if one is missing.)
   - **Save Version → Save & Run All (Commit)**. It runs in the background; you may close the browser.
4. When it finishes, open its log (*Output/Logs*), read the reliability check printout and **show it to your mentor**
   (go / no-go, task 4).
5. Create **Notebook 02**: import `kaggle/02_train.ipynb`.
   - *Settings*: **GPU T4 x2**, Internet **On**.
   - *Add Input → Your Work → Notebooks* → choose **Notebook 01**.
   - *Add-ons → Secrets* → add `GITHUB_TOKEN` (the token from step 2) and tick it for this notebook.

## Running notebook 02
Edit the first code cell, then **Save Version → Save & Run All (Commit)** each time:
1. `MODE = "tune"` once → read the **VAL** numbers at the end of the log, put the best `lambda_kd`
   and `pl_threshold` into `configs/default.yaml` on GitHub.
2. `MODE = "grid"` → repeat until the log says `0 runs to do` on both GPUs.
   Every run is pushed to GitHub (`runs/…`) as soon as it ends, so a session ending loses at most the run in progress.
   **To finish faster,** teammates can run the same notebook from their own Kaggle accounts with different `SEEDS`
   (e.g. one uses `"0"`, another `"1 2"`). Each account has its own 30 h per week.
3. `MODE = "tables"` → `results/main_table.tex`, `paired_bootstrap.csv`, `figs/*.pdf` are pushed to GitHub.

## Working from your laptop (no GPU needed)
After the runs are on GitHub you can make tables and figures on your Windows laptop:
```
git pull
python scripts/make_tables.py --runs runs --ours rwkd
python scripts/make_figures.py --runs runs --arch resnet18
```

## If something goes wrong
- *"Add an ISIC 2019 dataset as input"*: the ISIC copy you added has fewer than 20 000 images. Pick another one.
- *HTTP 403 / cannot download BiomedCLIP*: Internet is off for the notebook, or the phone is not verified.
- *CUDA out of memory*: add `train.batch_unlabelled=48` to `common` in notebook 02.
- *git push failed*: the token is missing, expired, or lacks write access.
- Anything else: copy the last ~30 lines of the log and ask.
