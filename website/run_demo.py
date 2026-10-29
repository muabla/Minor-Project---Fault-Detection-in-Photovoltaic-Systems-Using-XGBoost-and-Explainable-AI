"""Start the PV fault detection demo and open it in the browser.

    python run_demo.py            (then use the page that opens; press Ctrl+C here to stop)
"""
import importlib.util, os, sys, threading, time, webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "demo"))
sys.path.insert(0, HERE)
os.chdir(os.path.join(HERE, "demo"))

missing = [m for m in ("numpy", "pandas", "sklearn", "xgboost", "fastapi", "uvicorn") if importlib.util.find_spec(m) is None]
if missing:
    sys.exit("Missing packages: " + ", ".join(missing) + "\nRun:  pip install -r requirements.txt")
if not os.path.exists(os.path.join(HERE, "models", "xgb_source.json")):
    sys.exit("models/xgb_source.json not found. Run the notebook once (section E saves the models).")

import uvicorn
PORT = 8000
threading.Thread(target=lambda: (time.sleep(2), webbrowser.open(f"http://127.0.0.1:{PORT}")), daemon=True).start()
print(f"Demo running at http://127.0.0.1:{PORT}  (Ctrl+C to stop)")
uvicorn.run("server:app", host="127.0.0.1", port=PORT, log_level="warning")
