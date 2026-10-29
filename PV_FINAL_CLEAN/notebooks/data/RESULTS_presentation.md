# PV fault detection — results for the presentation

All results are on **simulated** data made by the calibrated physics simulator. Every score is accuracy on rows the model never saw (80 % train / 20 % test, 5 runs, like the paper). Tuned settings and feature choices use the training rows only. Weather (sunlight, temperature) is independent of the fault, as in reality.

**Feature sets.** *4 features*: the paper's Voc, Isc, G, T. *I-V*: + Vmp, Imp, fill factor (extra measurements from the I-V curve). *Isc/G*: current per unit of sunlight (physics feature). *Physics ratios*: Isc/G + Vmp/Voc + Imp/Isc + Imp/G.

## Part A — System 1

### A1. Step-by-step improvement (accuracy, "without → with Isc/G")

ANN (tuned) chooses Isc/G itself on validation rows, so it has one number per column. Paper ANN value is its own ANN (0.983).

| Model | Realistic, 4 features | Realistic + I-V | Realistic + I-V, high-res sensor | Paper (real data) |
|---|---|---|---|---|
| ANN (tuned) | 0.794 | 0.936 | 0.970 | 0.983 |
| XGBoost (tuned) | 0.716 → 0.810 | 0.874 → 0.942 | 0.884 → 0.972 | — |
| SVM (tuned) | 0.808 → 0.816 | 0.940 → 0.930 | 0.972 → 0.972 | 0.977 |
| SVM (paper) | 0.778 → 0.786 | 0.916 → 0.938 | 0.956 → 0.966 | 0.977 |
| RF (paper) | 0.652 → 0.806 | 0.840 → 0.940 | 0.880 → 0.976 | 0.957 |
| DT (paper) | 0.654 → 0.742 | 0.750 → 0.906 | 0.832 → 0.950 | 0.930 |
| LR (paper) | 0.822 → 0.830 | 0.952 → 0.948 | 0.974 → 0.974 | 0.787 |

### A2. Final System 1 models (realistic + I-V + high-res sensor)

The ANN was offered only Isc/G as an extra feature (not the three other ratios); see notes.

| Model | Features (chosen on training data) | Accuracy | Precision | Recall | F1 | Std | Paper accuracy (real) |
|---|---|---|---|---|---|---|---|
| ANN (tuned) | I-V, Isc/G chosen per run | 0.970 | 0.972 | 0.970 | 0.970 | 0.021 | 0.983 |
| XGBoost (tuned) | I-V + Isc/G | 0.972 | 0.973 | 0.972 | 0.972 | 0.011 | — |
| SVM (tuned) | I-V + physics ratios | 0.972 | 0.975 | 0.972 | 0.972 | 0.022 | 0.977 |
| RF (paper) | I-V + physics ratios | 0.974 | 0.976 | 0.974 | 0.974 | 0.015 | 0.957 |
| DT (paper) | I-V + physics ratios | 0.958 | 0.959 | 0.958 | 0.958 | 0.024 | 0.930 |
| LR (paper) | I-V + physics ratios | 0.976 | 0.978 | 0.976 | 0.976 | 0.018 | 0.787 |

### A3. Confusion matrices (run closest to the mean; rows/columns = normal, shading, dirt)

- **ANN (tuned)** (run 2): [[33, 0, 1], [0, 33, 0], [2, 0, 31]]
- **XGBoost (tuned)** (run 0): [[33, 0, 1], [0, 33, 0], [2, 0, 31]]
- **SVM (tuned)** (run 0): [[34, 0, 0], [0, 33, 0], [2, 1, 30]]
- **RF (paper)** (run 0): [[33, 0, 1], [0, 33, 0], [2, 0, 31]]
- **DT (paper)** (run 0): [[33, 0, 1], [0, 33, 0], [2, 1, 30]]
- **LR (paper)** (run 0): [[34, 0, 0], [0, 33, 0], [2, 0, 31]]

### A4. ANOVA (randomized block: model × run)

| Metric | F | p | partial η² |
|---|---|---|---|
| Accuracy | 2.65 | 0.054 | 0.399 |
| Precision | 3.07 | 0.032 | 0.434 |
| Recall | 2.65 | 0.054 | 0.399 |
| F1 | 2.65 | 0.054 | 0.398 |

p ≈ 0.05: the final models are statistically about equally good (the paper found strong differences, p ≈ 5×10⁻¹¹).

### A5. Sensitivity to measurement error on the test voltages and currents (as in the paper's Fig. 11)


| Model | ±0 % | ±1 % | ±5 % | ±10 % | ±15 % | ±20 % |
|---|---|---|---|---|---|---|
| ANN (tuned) | 0.970 | 0.934 | 0.719 | 0.613 | 0.552 | 0.513 |
| XGBoost (tuned) | 0.972 | 0.911 | 0.619 | 0.524 | 0.505 | 0.499 |
| SVM (tuned) | 0.972 | 0.968 | 0.867 | 0.719 | 0.643 | 0.583 |
| RF (paper) | 0.974 | 0.964 | 0.671 | 0.571 | 0.527 | 0.518 |
| DT (paper) | 0.958 | 0.934 | 0.658 | 0.565 | 0.517 | 0.510 |
| LR (paper) | 0.976 | 0.971 | 0.831 | 0.687 | 0.628 | 0.584 |
| Paper ANN (Fig. 11) | 0.983 | 0.933 | 0.667 | 0.533 | 0.450 | 0.417 |

### A6. After noise-robust training (training rows + copies with ±1 % and ±3 % error)

Not run for the ANN and DT.

| Model | ±0 % | ±1 % | ±5 % | ±10 % |
|---|---|---|---|---|
| LR (paper) | 0.968 | 0.967 | 0.898 | 0.745 |
| RF (paper) | 0.966 | 0.968 | 0.885 | 0.683 |
| SVM (tuned) | 0.932 | 0.933 | 0.785 | 0.419 |
| XGBoost (tuned) | 0.954 | 0.949 | 0.844 | 0.660 |

## Part B — System 2: zero-shot and few-shot

Models are trained on simulated System 1 and tested on simulated System 2. *Zero-shot*: no System 2 data in training. *Few-shot k*: k System 2 rows per class added to training (weighted 5×), tested on the remaining System 2 rows (5 draws).

### B1. System 2 accuracy ("without → with Isc/G")

Paper (trained on real System 1, tested on its real System 2): best SVM 0.627, RF 0.540, ANN 0.527, LR 0.513, DT 0.467.

| Model | 4 feat. zero-shot | 4 feat. k=5 | 4 feat. k=10 | I-V zero-shot | I-V k=5 | I-V k=10 |
|---|---|---|---|---|---|---|
| ANN (tuned) | 0.696 | 0.909 | 0.966 | 0.472 | 0.906 | 0.956 |
| XGBoost (tuned) | 0.800 → 0.805 | 0.907 → 0.995 | 0.954 → 0.996 | 0.300 → 0.345 | 0.923 → 0.895 | 0.960 → 0.952 |
| SVM (tuned) | 0.805 → 0.800 | 0.965 → 0.966 | 0.991 → 0.994 | 0.420 → 0.410 | 0.901 → 0.872 | 0.932 → 0.951 |
| SVM (paper) | 0.660 → 0.895 | 0.962 → 0.965 | 0.976 → 0.993 | 0.335 → 0.335 | 0.908 → 0.892 | 0.969 → 0.967 |
| RF (paper) | 0.785 → 0.805 | 0.936 → 0.973 | 0.976 → 0.994 | 0.365 → 0.525 | 0.912 → 0.933 | 0.942 → 0.959 |
| DT (paper) | 0.695 → 0.775 | 0.785 → 0.945 | 0.914 → 0.944 | 0.335 → 0.335 | 0.941 → 0.870 | 0.954 → 0.944 |
| LR (paper) | 0.800 → 0.800 | 0.965 → 0.969 | 0.979 → 0.973 | 0.420 → 0.420 | 0.945 → 0.937 | 0.973 → 0.972 |

### B2. Side result: trained and tested on simulated System 2 (80/20, 5 runs)


| Model | 4 features | I-V |
|---|---|---|
| ANN (tuned) | 1.000 | 0.975 |
| XGBoost (tuned) | 0.935 → 0.995 | 0.980 → 0.980 |
| SVM (tuned) | 0.990 → 0.990 | 0.985 → 0.985 |
| SVM (paper) | 0.985 → 1.000 | 0.985 → 0.980 |
| RF (paper) | 0.955 → 0.995 | 0.990 → 0.990 |
| DT (paper) | 0.910 → 0.995 | 0.975 → 0.980 |
| LR (paper) | 0.995 → 1.000 | 0.995 → 0.995 |

### B3. Our simulated System 2 vs the paper's real System 2

| | Paper (real) | Ours (simulated) |
|---|---|---|
| Hardware | 7 × 200 W modules in series, PHB1500-NS inverter | same layout and module ratings |
| Module health | 2 modules with a hotspot cell, 2 with bad cell connections | all modules healthy |
| Data | 60 rows (30 distinct readings, each recorded twice) | 200 distinct rows, balanced |
| Weather | each fault recorded in its own session (~14.5 °C or ~28 °C) | sunlight/temperature random over the paper's range, same for every class |
| Shading | chairs in front of the modules | shadow over 6–10 neighbouring module sections, calibrated to the real readings |
| Dirt | splashed dirt | light film + blotches incl. a few heavy ones, calibrated to the real readings |
| Measurements | inverter MPP converted to Voc/Isc | same (4 features); plus an I-V-tracer version |
| Match to real data | — | normal and dirt close; our shading voltage higher (~0.74 vs ~0.61) |

## Notes and limitations
- Simulated System 2 is healthy and idealised, so its numbers are not a like-for-like comparison with the paper's real System 2 (0.627).
- The ANN was tuned over learning rate and Isc/G only; the three extra physics ratios were offered to the other models but not to the ANN.
- XGBoost is the model most sensitive to measurement error (A5); noise-robust training reduces this at a small cost on clean data (A6).
- System 1 light dirt that changes the current by less than the sensor noise is physically undetectable; this sets the ~97 % ceiling.