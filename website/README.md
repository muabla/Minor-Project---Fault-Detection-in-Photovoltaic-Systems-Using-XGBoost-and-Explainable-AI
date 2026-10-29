# Website demonstration (PV fault detection: XGBoost + SHAP)

Create faults on the panels of System 1, 2 or 3 and watch the monitor diagnose them: I-V curve, diagnosis,
SHAP reasons, and the circuit (contactor, indicators, display). The monitoring program (`pi/pi_monitor.py`) runs
with emulated sensors and pins (`pi/hal_virtual.py`).

1. Install Python 3.10+.
2. In this folder: `pip install -r requirements.txt`
3. `python run_demo.py` - the page opens at http://127.0.0.1:8000 (Ctrl+C to stop).

Opening `demo/index.html` directly does not work; the page needs the Python server.
