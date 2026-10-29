# Simulink/Simscape model of System 3 (MATLAB R2026b)

Needs Simulink, Simscape Electrical and the Statistics and Machine Learning Toolbox. Set this folder as the
Current Folder, then:

| Step | Command | What it does |
|---|---|---|
| 1 | `setup_pv` | imports the trained XGBoost model and checks 18 known readings against Python |
| 2 | `build_pv_slx(3)` | builds and opens `pv_system3.slx` |
| 3 | `check_healthy(3)` | healthy array: Simscape vs the Python simulator |
| 4 | `pv_check(3, ...)` | re-simulates a fault (paste from the website's "Copy MATLAB command") |
| 5 | `compare_with_website` | the same fault in the website and in Simulink |

Examples: `pv_check(3)`, `pv_check(3,'open',1)`, `pv_check(3,'short',[1 2 1; 1 4 3; 1 7 2])`, `pv_check(3,'corrosion',0.2)`.
