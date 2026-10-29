# XGBoost + Explainable AI: one model for PV systems it has never seen

* `PV_XGBoost_XAI_v6.ipynb` - the notebook (already run, outputs inside): tunes XGBoost on the library of 90 generic
  PV systems, tests on Systems 1-3 (zero-shot and after 5 readings per class), stress test, SHAP, saves the models.
* `train_xgboost.py` - the core training / evaluation / SHAP code as one script (the code in the report's appendix).
* `pv_simulator_v3.py` - cell-level PV circuit simulator (Systems 1-3, six classes, I-V tracer, the 90-system library,
  the 22 model inputs); `pv_simulator_v2.py` is the base single-diode simulator it builds on.
* `data_v6/` - test systems (`S1`, `S2`, `S3`) and the training library (`library_6class_5940.csv`, `library_systems.csv`).
* `models/` - `xgb_source.json` (library model) and the commissioned models; `figures/` - result plots.
* `verify_v6.py` - independent check (no test data in training, saved model = notebook result, fresh data, other seeds).

Results (accuracy): zero-shot 0.967 / 0.980 / 0.983, after 5 readings per class 0.974 / 0.992 / 0.991 (Systems 1 / 2 / 3).

Run: `pip install -r requirements.txt`, then open the notebook (Colab: upload this folder and run all, about 40-60 min).
