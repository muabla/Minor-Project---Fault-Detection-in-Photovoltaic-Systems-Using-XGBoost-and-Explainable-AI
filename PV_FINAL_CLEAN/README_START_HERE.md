# PV Fault Detection: final files (start here)

This folder (`MyDrive/PV_Project/PV_FINAL_CLEAN`) holds **only the final files**: exact copies, nothing edited except `make_datasets.py`. Your old folders were not touched. If a file is not in this folder, you do not need it.

```
PV_FINAL_CLEAN/
├── README_START_HERE.md               ← this file
├── notebooks/
│   ├── PV_XGBoost.ipynb               ← our model (main one going forward)
│   ├── PV_ANN.ipynb  PV_SVM.ipynb  PV_RF.ipynb  PV_DT.ipynb  PV_LR.ipynb   ← the paper's 5 models
│   ├── PV_Fault_Final_v3_ALL_IN_ONE.ipynb   ← the notebook behind the 2nd-evaluation slides
│   └── data/
│       ├── S1_sim500_realistic_4features.csv
│       ├── S1_sim500_realistic_IV.csv
│       ├── S1_sim500_realistic_hires_IV.csv
│       ├── S2_sim200_realistic_4features.csv
│       ├── S2_sim200_realistic_IV.csv
│       └── RESULTS_presentation.md    ← all slide tables in one file
├── simulator/
│   ├── pv_simulator_v2.py             ← the final simulator
│   ├── make_datasets.py               ← one command to create new datasets
│   ├── Data_300.csv                   ← paper's real System 1 data (simulator reads its G/T range)
│   └── Data_60.csv                    ← paper's real System 2 data (same use)
└── results/
    ├── final_SUMMARY.csv              ← one-table summary behind the slides
    ├── final_A_system1.csv            ← System 1, every model/run
    ├── final_C_transfer.csv           ← System 2 zero-shot / few-shot, every run
    └── final_D_system2_within.csv     ← System 2 side result
```

---

## 1. Which notebook is "final"?

There are **two final sets**, made at different times. Both use the same datasets and the same simulator.

| Notebook | Made | What it is | Use it for |
|---|---|---|---|
| `PV_XGBoost.ipynb` + `PV_ANN / SVM / RF / DT / LR.ipynb` | 12 Oct | One clean notebook per model, paper style. Each does: System 1 initial → System 1 final → System 2 zero-shot + few-shot → summary. Outputs are saved, so you can read results without running. | **Showing code to faculty, and all future work** (SHAP, demo). Since the project now continues with XGBoost only, `PV_XGBoost.ipynb` is the one to build on. |
| `PV_Fault_Final_v3_ALL_IN_ONE.ipynb` | 5–7 Oct | Everything in one notebook: all models, all feature variants, few-shot, confusion matrices, ANOVA, sensitivity tests. | **The record behind the 2nd-evaluation slides.** Every number on the slides comes from this one. |

### The numbers are slightly different between the two. Here is why.

| | v3 all-in-one (slides) | Per-model notebooks (12 Oct) |
|---|---|---|
| Features | Each model picks its own extra features by cross-validation (e.g. XGBoost used I-V + Isc/G only) | Every model gets the same inputs: I-V + Isc/G + all 3 physics ratios |
| Few-shot tuning | Settings tuned once on System 1, then reused | Settings re-tuned on each few-shot training set |
| Random seeds | Fixed per experiment | Different per run |

Both are honest (no test data used for any choice); they are just two slightly different recipes.

### Results side by side (accuracy, mean of 5 runs)

| Model | S1 initial (4f + Isc/G) | S1 final (I-V, hi-res) | S2 zero-shot | S2 few-shot k=5 | S2 few-shot k=10 |
|---|---|---|---|---|---|
| **XGBoost**: slides | 0.810 | 0.972 | 0.805 | 0.995 | 0.996 |
| **XGBoost**: notebook | 0.810 | 0.970 | 0.805 | 0.980 | 0.995 |
| ANN: slides | 0.794 | 0.970 | 0.696 | 0.909 | 0.966 |
| ANN: notebook | 0.794 | 0.966 | 0.740 | 0.973 | 0.991 |
| SVM: slides | 0.816 | 0.972 | 0.800 | 0.966 | 0.994 |
| SVM: notebook | 0.816 | 0.972 | 0.803 | 0.937 | 0.998 |
| RF: slides | 0.806 | 0.974 | 0.805 | 0.973 | 0.994 |
| RF: notebook | 0.806 | 0.974 | 0.805 | 0.989 | 0.995 |
| DT: slides | 0.742 | 0.958 | 0.775 | 0.945 | 0.944 |
| DT: notebook | 0.742 | 0.958 | 0.762 | 0.936 | 0.945 |
| LR: slides | 0.830 | 0.976 | 0.800 | 0.969 | 0.973 |
| LR: notebook | 0.830 | 0.976 | 0.800 | 0.969 | 0.973 |

**System 1 matches almost exactly.** Small differences appear only in System 2 few-shot, where only 15–30 System 2 rows are added, so a different tuning recipe moves the score by a few points. If faculty open `PV_XGBoost.ipynb` and see 0.980 for k=5 instead of the slide's 0.995, that is the reason.

---

## 2. Which dataset is used where?

All are **simulated** with `pv_simulator_v2.py`, with **realistic weather** (sunlight and temperature random, the same for every class). Labels: `Fault` 0 = normal, 1 = partial shading, 2 = dirt.

| File | Rows | Columns | Used by |
|---|---|---|---|
| `S1_sim500_realistic_4features.csv` | 500 | Voc, Isc, G, T (paper's sensor) | all notebooks: "System 1 initial" and training data for System 2 |
| `S1_sim500_realistic_IV.csv` | 500 | + Vmp, Imp, FF (paper's sensor) | v3 only (middle column of slide "step-by-step") |
| `S1_sim500_realistic_hires_IV.csv` | 500 | + Vmp, Imp, FF (high-res sensor) | all notebooks: "System 1 final" (main result) |
| `S2_sim200_realistic_4features.csv` | 200 | Voc, Isc, G, T (inverter route, like the paper) | all notebooks: System 2 zero-shot / few-shot |
| `S2_sim200_realistic_IV.csv` | 200 | + Vmp, Imp, FF (I-V tracer route) | v3 only (System 2 side result, I-V column) |

The per-model notebooks need only the 3 files marked "all notebooks".

---

## 3. How to run

### Google Colab
1. Open the notebook from `PV_FINAL_CLEAN/notebooks` (double-click → Open with Colab).
2. In the **first code cell**, change the `DATA_DIR` line to:
   ```python
   DATA_DIR = '/content/drive/MyDrive/PV_Project/PV_FINAL_CLEAN/notebooks/data'
   ```
   (The notebooks were written for the old `PV_final` folder. That old path still works too, because the same files are there.)
3. Runtime → Run all. Rough run times on Colab CPU:
   - DT, LR: under a minute.
   - RF, SVM: a few minutes.
   - XGBoost, ANN: about 10–15 minutes each.
   - v3 all-in-one: about 1–1.5 hours (about 35 min with `RUN_ANN = False`).

### On your laptop (VS Code / Jupyter)
- Download the `notebooks` folder (keep `data/` inside it).
- The per-model notebooks then work as they are, because they look for a folder called `data` next to them.
- For the v3 notebook, set `DATA_DIR = 'data'` in its first code cell.
- Packages: `pip install numpy pandas scikit-learn xgboost tensorflow` (tensorflow is only needed for the ANN).

### Making new datasets (only if you need new data)
In the `simulator` folder:
```
python make_datasets.py
```
This writes 5 new CSVs into a folder called `regenerated_not_presentation_data`. They have the same statistics as the final files but **different rows**, so they will not reproduce the slide numbers exactly. The final datasets are the ones in `notebooks/data`.

---

## 4. Old files you can ignore (they are NOT in this folder)

Nothing was deleted. These stay where they were; you just don't need them.

**In `PV_Project/PV_final/`:**

| File(s) | What it was |
|---|---|
| `S1_sim500_shortcut_4features.csv`, `S1_sim500_shortcut_IV.csv`, `S2_sim200_shortcut_4features.csv`, `S2_sim200_shortcut_IV.csv` | Weather-shortcut versions (weather copied per class). Dropped from the project. |
| `res_*.csv`, `res_confusion_closest_to_mean.json` (11 files) | Raw results of older runs (before v3). |
| `RESULTS.md` | Older full results, includes shortcut tables. Replaced by `RESULTS_presentation.md`. |
| `README.md` (5 Oct) | Old readme for v3. |
| `README.md` (12 Oct) | Readme for the per-model notebooks; its content is included here. |
| `data/` subfolder (12 Oct) | Duplicate copies of 3 CSVs (same files as in `notebooks/data`). |
| `final_*.csv`, `pv_simulator_v2.py`, `Data_300.csv`, `Data_60.csv`, notebooks | **These are the finals.** Identical copies are in this folder. |

**In `PV_Project/`:**

| File(s) | What it was |
|---|---|
| `pv_simulator.py` | Simulator **version 1**, which kept the weather shortcut. Replaced by `pv_simulator_v2.py`. |
| `data/` | Version-1 data, including mixed real + simulated sets (dropped). |
| `results_v2/` | Older results. |
| `__pycache__/` | Python cache, never needed. |

Once you are happy with this folder, you can safely move `PV_final` and the other old items into an `_old` folder, so nobody in the group opens the wrong file.
