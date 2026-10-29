# Fault Detection in Photovoltaic Systems Using XGBoost and Explainable AI

Minor project, Department of Electrical and Electronics Engineering (minor in Computer Science and Engineering),
T.K.M. College of Engineering, Kollam, 2026.

**Team:** Muhammad Abdul Latheef (B23EEA46), Rahul R (B23EEA51), S Sreejith (B23EEA56), Siya S (B23EEA59)
**Guide:** Prof. Kajal K Nair

## Contents
| Folder | What it is |
|---|---|
| `PV_FINAL_CLEAN/` | Comparison study: six classifiers (ANN, SVM, RF, DT, LR, XGBoost) on normal / partial shading / dirt, the calibrated simulator, datasets and result tables. Start with `PV_FINAL_CLEAN/README_START_HERE.md`. |
| `XGBoost_XAI/` | Main work: one XGBoost model trained on a library of 90 generic simulated PV systems, 6 classes, 22 inputs, SHAP explanations. Zero-shot accuracy on unseen Systems 1/2/3: 0.967 / 0.980 / 0.983. |
| `website/` | Interactive website: create faults on the panels and see the I-V curve, diagnosis, SHAP reasons and the circuit response. Run `python run_demo.py`. |
| `matlab/` | Simulink/Simscape model of System 3 with the XGBoost model in the controller subsystem (MATLAB R2026b). |

## Comparison study results (accuracy on unseen readings, mean of 5 runs)
| Model | System 1 | System 2 zero-shot | System 2, 5 readings per class |
|---|---|---|---|
| XGBoost | 0.972 | 0.805 | **0.995** |
| ANN | 0.970 | 0.696 | 0.909 |
| SVM | 0.972 | 0.800 | 0.966 |
| RF | 0.974 | 0.805 | 0.973 |
| DT | 0.958 | 0.775 | 0.945 |
| LR | 0.976 | 0.800 | 0.969 |

Reference study: J. Zwirtes et al., "Fault detection in photovoltaic systems using a machine learning approach,"
IEEE Access, vol. 13, 2025. Its public data: https://github.com/JossiasZ/Photovoltaic-Data
