"""
PV fault detection demo -- backend (v6, software-in-the-loop).

What happens when you press "Measure & diagnose":
  1. the page sends the fault you drew (shadows, dust, cut wire, shorted diodes, corrosion) and the weather;
  2. the PLANT (pv_simulator_v3, the circuit simulation) is set to that state;
  3. the RASPBERRY PI PROGRAM (pi/pi_monitor.py) runs one monitoring cycle against emulated chips (pi/hal_virtual.py):
     it reads the ADS1115 and DS18B20, triggers the I-V tracer over GPIO/UART, computes the 22 inputs, runs XGBoost,
     and switches its GPIO pins (contactor, LEDs, buzzer) and writes the OLED;
  4. the page draws what the program did: pin states, OLED frame buffer, bus log, the I-V curve and the SHAP explanation.

Start it with:  python run_demo.py
"""
import json, os, sys, time
import numpy as np, pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "pi"))
import pv_simulator_v3 as P
import pi_monitor as PM
import hal_virtual as HV

MODELS, DATA = os.path.join(ROOT, "models"), os.environ.get("PV_DATA", os.path.join(ROOT, "data_v6"))
PARAMS = json.load(open(os.path.join(MODELS, "xgb_params.json")))
K = len(P.CLASSES); SYSTEM_IDS = (1, 2, 3)
LIB = pd.read_csv(os.path.join(DATA, [f for f in os.listdir(DATA) if f.startswith("library_6class")][0]))
X0, Y0 = P.inputs(LIB), LIB.Fault                    # training data of the library model (no test system in it)
FEATS = list(X0.columns)

NOTES = {0: "Normal operation.",
         1: "Alert only: shading is not dangerous; power is lost while it lasts. Remove or trim the obstacle if it is permanent.",
         2: "Alert only: schedule panel cleaning.",
         3: "Alarm: a string is not delivering power. Check fuses, connectors and cables.",
         4: "Trip: the DC contactor opens to isolate the array (fire risk). It stays open until RESET is pressed.",
         5: "Maintenance: high series resistance (corroded or loose connectors, burnt contacts, damaged cable) wastes power and heats up."}
WORDS = {"Voc/MaxVoc": "open-circuit voltage", "Isc/MaxIsc": "short-circuit current", "G/1000": "irradiance", "AT/50": "ambient temperature",
         "Vmp/MaxVoc": "voltage at max power", "Imp/MaxIsc": "current at max power", "FF": "fill factor", "Isc/G": "current per unit sunlight",
         "Vmp/Voc": "knee position (Vmp/Voc)", "Imp/Isc": "MPP current vs Isc", "Imp/G": "MPP current per unit sunlight",
         "Rs_n": "curve slope near Voc", "Rsh_n": "curve slope near Isc", "Steps": "steps in the I-V curve",
         "Voc_r": "voltage (Voc)", "Isc_r": "current (Isc)", "Vmp_r": "voltage at max power", "Imp_r": "current at max power",
         "FF_r": "fill factor", "P_r": "power", "Rs_r": "series resistance", "Strings": "number of parallel strings"}


def plain(feature, value, ref):
    """One sentence comparing this reading with a healthy array in the same sunlight and temperature."""
    if feature == "Steps":
        return f"The I-V curve has {value:.0f} power peak(s); a healthy array has {ref:.0f}." + (" Extra peaks mean bypass diodes have switched on." if value > ref else "")
    if feature == "Strings":
        return f"The array has {value:.0f} parallel string(s) (set at installation). In parallel arrays a short shows up as resistance, not as lost voltage."
    if feature in ("G/1000", "AT/50"):
        return f"Weather input ({WORDS[feature]}) — used only as the reference for what is expected."
    if abs(ref) < 1e-9:
        return f"{WORDS[feature][0].upper() + WORDS[feature][1:]}: {value:.3f}."
    pct = (value / ref - 1) * 100
    if feature in ("Rs_r", "Rs_n"):
        return f"Series resistance is {value / ref:.1f}× what a healthy array has." if value > 1.15 * ref else f"Series resistance is normal ({value / ref:.2f}× healthy)."
    if abs(pct) < 1.5:
        return f"{WORDS[feature][0].upper() + WORDS[feature][1:]} is normal (within {abs(pct):.1f} % of a healthy array)."
    return f"{WORDS[feature][0].upper() + WORDS[feature][1:]} is {abs(pct):.0f} % {'below' if pct < 0 else 'above'} what a healthy array gives in this sunlight and temperature."


# ---------------------------------------------------------------- one virtual Raspberry Pi per system
class State:
    def reset(self):
        self.hal, self.mon = {}, {}
        for s in SYSTEM_IDS:
            self.hal[s] = HV.VirtualHAL()
            self.mon[s] = PM.Monitor(self.hal[s], PM.system_config(s), os.path.join(MODELS, "xgb_source.json"))
        self.desc = {s: "Library model — this system was never in training (zero-shot)" for s in SYSTEM_IDS}
        self.adapted = {s: False for s in SYSTEM_IDS}
        self.ranges = {s: (X0.min(), X0.max()) for s in SYSTEM_IDS}


ST = State(); ST.reset()


def hw_state(s):
    h = ST.hal[s]
    return dict(pins={str(k): v for k, v in h.states.items()}, oled=h.i2c.oled_pixels(), log=list(h.log), tripped=ST.mon[s].tripped)


def layout_info(system, config):
    shape = {36: (4, 9), 60: (6, 10), 72: (6, 12)}
    return [[dict(name=m["name"], n_cells=m["n_cells"], n_bypass=m["n_bypass"], cols=shape[m["n_cells"]][0], rows=shape[m["n_cells"]][1],
                  voc=m.get("Voc_nameplate", m["Voc"]), isc=m["Isc"]) for m in b] for b in P.layout(system, config)]


def checks(system, req, X):
    """Warnings: fault smaller than the detection threshold, or a reading outside what the library covered."""
    w = []; E = P.ENV_SYS[system]
    if not (E["G"][0] <= req.G <= E["G"][1]) or not (E["Ta"][0] <= req.Ta <= E["Ta"][1]):
        w.append(f"Weather outside the tested range ({E['G'][0]:.0f}-{E['G'][1]:.0f} W/m2, {E['Ta'][0]:.0f}-{E['Ta'][1]:.0f} C).")
    flat = [m for b in P.layout(system, req.config) for m in b]; nb = flat[0]["n_bypass"]
    if req.shadow and nb:
        sh = [np.asarray(t, float) > 0.5 for t in req.shadow]
        total = nb * len(flat); need = max(2, int(np.ceil(0.05 * total)))
        subs = sum(int(m[g].any()) for m, f in zip(sh, flat) for g in np.array_split(np.arange(f["n_cells"]), nb))
        if 0 < subs < need:
            w.append(f"Small shadow: it covers {subs} of {total} substrings (detection threshold: {need}). Make it bigger to show shading reliably.")
    if req.dust is not None and 0 < req.dust < 3:
        w.append("Very light dust (under 3 % loss) is below the detection threshold; the model may call it normal.")
    if req.short and nb:
        n = len(req.short); per = max(len(b) for b in P.layout(system, req.config)) * nb; need = max(1, int(np.ceil(0.04 * per)))
        if n < need:
            w.append(f"Only {n} shorted bypass diode(s) of {per} in the string; the detection threshold is {need}.")
    if 0 < req.r_frac < 0.07:
        w.append("Mild extra resistance (under 7 %) is below the detection threshold.")
    lo, hi = ST.ranges[system]
    out = [f for f in P.RESID if X[f].iloc[0] < lo[f] - 0.02 or X[f].iloc[0] > hi[f] + 0.02]
    if out:
        w.append("Reading outside the range of the training data for: " + ", ".join(WORDS[f] for f in out[:3]) + ".")
    return w


def to_kwargs(req):
    kw = {}
    if req.trans:
        kw["maps"] = [np.round(np.asarray(t, float) * req.G / 5.0) * 5.0 for t in req.trans]
    if req.open_branch is not None:
        kw["open_branch"] = req.open_branch
    if req.short:
        kw["short"] = [(int(b), int(m), s if s == "all" else int(s)) for b, m, s in req.short]
    if req.r_frac > 0:
        kw["r_frac"], kw["r_branch"] = req.r_frac, req.r_branch
    return kw


# ---------------------------------------------------------------- API
app = FastAPI()


class MeasureReq(BaseModel):
    system: int = 1
    config: str = "series"
    G: float = 800.0
    Ta: float = 25.0
    trans: list | None = None
    open_branch: int | None = None
    short: list | None = None
    r_frac: float = 0.0
    r_branch: int = 0
    shadow: list | None = None
    dust: float | None = None
    reset: bool = False                       # RESET push button held during this cycle


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "index.html"))


@app.get("/api/layout")
def api_layout(system: int = 1, config: str = "series"):
    mv, mi = P.norm_limits(system, config)
    return dict(system=system, config=config, name=P.SYSTEMS[system]["name"], branches=layout_info(system, config),
                max_voc=mv, max_isc=mi, env=P.ENV_SYS[system])


@app.get("/api/status")
def api_status():
    return dict(desc=ST.desc, adapted=ST.adapted, classes=P.CLASSES)


@app.get("/api/hw")
def api_hw(system: int = 1):
    return hw_state(system)


@app.post("/api/measure")
def api_measure(req: MeasureReq):
    s = req.system; hal, mon = ST.hal[s], ST.mon[s]
    mon.cfg = PM.system_config(s, req.config)                       # System 1 can be wired in series or parallel
    hal.plant.scenario = dict(system=s, config=req.config, G=req.G, Ta=req.Ta, fault=to_kwargs(req))
    hal.log.clear(); hal.plant.measure()                             # the array's state at this moment
    hal.press_reset(req.reset)
    try:
        r = mon.run_cycle()                                          # <-- the Raspberry Pi program
    finally:
        hal.press_reset(False)
    m = hal.plant.last
    # the exact scenario the Python plant simulated (for the "Copy MATLAB command" button); light = cell irradiance / G
    kw = hal.plant.scenario["fault"]; flat = [mm for br in P.layout(s, req.config) for mm in br]
    light = [np.round(np.asarray(mp, float) / req.G, 3).tolist() for mp in kw["maps"]] if "maps" in kw else [[1.0] * mm["n_cells"] for mm in flat]
    scenario = dict(system=s, config=req.config, G=req.G, Ta=req.Ta, Tc=round(float(m['raw']['Tc']), 2), light=light,
                    short=[[int(b_) + 1, int(m_) + 1, 0 if g_ == "all" else int(g_) + 1] for b_, m_, g_ in kw.get("short", [])],
                    open=(kw["open_branch"] + 1) if "open_branch" in kw else 0,
                    corrosion=float(kw.get("r_frac", 0.0)), rbranch=int(kw.get("r_branch", 0)) + 1)
    ref = P.measure(s, req.G, req.Ta, np.random.default_rng(), config=req.config, curve=True)   # healthy array, same weather
    Xref = P.inputs(pd.DataFrame([{**ref["features"], **P.residual_row(s, req.config, ref["features"])}]))
    X = pd.DataFrame([r["inputs"]])[FEATS]
    order = sorted(r["shap"], key=lambda k: -abs(r["shap"][k]))[:6]
    shap_list = [dict(feature=f, words=WORDS[f], value=float(X[f].iloc[0]), contrib=float(r["shap"][f]),
                      plain=plain(f, float(X[f].iloc[0]), float(Xref[f].iloc[0]))) for f in order]
    return dict(features=m["features"], raw={k: float(v) for k, v in m["raw"].items()}, curve=m["curve"],
                pred=dict(cls=r["cls"], label=r["label"], probs=[float(p) for p in r["probs"]]), shap=shap_list,
                note=NOTES[r["cls"]], warnings=checks(s, req, X), model=ST.desc[s],
                inputs={k: float(v) for k, v in X.iloc[0].items()}, ref_curve=ref["curve"],
                ref_inputs={k: float(v) for k, v in Xref.iloc[0].items()}, hw=hw_state(s), scenario=scenario, feature_order=FEATS)


class SysReq(BaseModel):
    system: int = 1
    config: str = "series"
    G: float = 800.0
    k: int = 5


@app.post("/api/reset_button")
def api_reset_button(req: SysReq):
    """Press the RESET push button (GPIO26) during one cycle: the program re-closes the contactor if the fault is gone."""
    hal = ST.hal[req.system]; hal.press_reset(True)
    try:
        hal.log.clear(); hal.plant.measure(); r = ST.mon[req.system].run_cycle()
    finally:
        hal.press_reset(False)
    return dict(label=r["label"], hw=hw_state(req.system))


@app.post("/api/export_model")
def api_export(req: SysReq):
    name = f"live_model_system{req.system}.json"; ST.mon[req.system].model.save_model(os.path.join(MODELS, name))
    return dict(file=f"models/{name}")


@app.post("/api/hidden")
def api_hidden(req: SysReq):
    """A random fault drawn from the test system's own fault models, kept hidden by the page."""
    rng = np.random.default_rng(); fault = int(rng.integers(K))
    kw = P.sample_fault(req.system, fault, req.G, rng, req.config)
    out = dict(truth=fault, label=P.CLASSES[fault], open_branch=kw.get("open_branch"), r_frac=kw.get("r_frac", 0.0),
               r_branch=kw.get("r_branch", 0), short=[[b, m, s] for b, m, s in kw.get("short", [])], trans=None)
    if "maps" in kw:
        out["trans"] = [np.clip(m / req.G, 0, 1).round(3).tolist() for m in kw["maps"]]
    return out


def sample_set(system, n_per_class, rng):
    rows = []
    for f in range(K):
        for i in range(n_per_class):
            cfg = ("series", "parallel")[i % 2] if system == 1 else "series"
            G, Ta = rng.uniform(*P.ENV_SYS[system]["G"]), rng.uniform(*P.ENV_SYS[system]["Ta"])
            r = P.measure(system, G, Ta, rng, config=cfg, **P.sample_fault(system, f, G, rng, cfg))
            rows.append({**r["features"], **P.residual_row(system, cfg, r["features"]), "Fault": f})
    return pd.DataFrame(rows)


@app.post("/api/commission")
def api_commission(req: SysReq):
    """Commissioning: k labelled readings per class from this system + the library, retrain, check on fresh readings,
    then the Raspberry Pi program loads the new model file."""
    rng = np.random.default_rng(); t0 = time.time()
    shots = sample_set(req.system, req.k, rng); check = sample_set(req.system, 8, rng); t_read = time.time() - t0
    mon = ST.mon[req.system]
    before = float((mon.model.predict(P.inputs(check)) == check.Fault).mean())
    t1 = time.time()
    m = XGBClassifier(random_state=0, n_jobs=-1, **PARAMS).fit(pd.concat([X0, P.inputs(shots)]), pd.concat([Y0, shots.Fault]),
                                                               sample_weight=np.r_[np.ones(len(X0)), np.full(len(shots), 5.0)])
    t_train = time.time() - t1
    path = os.path.join(MODELS, f"live_model_system{req.system}.json"); m.save_model(path); mon.load_model(path)
    after = float((m.predict(P.inputs(check)) == check.Fault).mean())
    ST.adapted[req.system] = True
    ST.desc[req.system] = f"Commissioned: library + {req.k} readings per class from this system"
    allX = pd.concat([X0, P.inputs(shots)]); ST.ranges[req.system] = (allX.min(), allX.max())
    return dict(before=before, after=after, n_readings=len(shots), n_check=len(check), read_s=round(t_read, 1),
                train_s=round(t_train, 1), desc=ST.desc[req.system])


@app.post("/api/fallback")
def api_fallback(req: SysReq):
    ST.mon[req.system].load_model(os.path.join(MODELS, f"xgb_adapted_system{req.system}.json"))
    ST.adapted[req.system] = True; ST.desc[req.system] = "Pre-commissioned model (saved by the notebook, 5 readings per class)"
    return dict(desc=ST.desc[req.system])


@app.post("/api/reset")
def api_reset():
    ST.reset(); return dict(ok=True)
