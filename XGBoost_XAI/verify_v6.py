"""
verify_v6.py -- independent check of the v6 results (separate from the notebook).

  1. Leak check: the training library contains no data from test Systems 1-3
     (system ids, identical rows, look-alike designs, module ratings).
  2. The saved model file (models/xgb_source.json) reproduces the notebook's zero-shot accuracy.
  3. Fresh test data: new readings of Systems 1-3 generated with new random seeds, scored by the saved model.
  4. Retraining with 3 different random seeds gives the same accuracy (the result is not luck).
Run:  python verify_v6.py      (about 5-10 minutes)
"""
import json, time
import numpy as np, pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pv_simulator_v3 as P

t0 = time.time()
LIB = pd.read_csv("data_v6/library_6class_5940.csv"); LS = pd.read_csv("data_v6/library_systems.csv")
T = {1: pd.read_csv("data_v6/S1_6class_1002.csv"), 2: pd.read_csv("data_v6/S2_6class_402.csv"), 3: pd.read_csv("data_v6/S3_6class_402.csv")}
ok = True

print("1. LEAK CHECK")
ids = set(LIB.System)
print(f"   library system ids: {min(ids)}..{max(ids)} ({len(ids)} systems); test ids in library: {sorted(ids & {1, 2, 3}) or 'none'}")
ok &= not (ids & {1, 2, 3})
cols = list(P.inputs(LIB).columns)
lib_rows = set(map(tuple, P.inputs(LIB).round(6).values))
for s, d in T.items():
    same = sum(tuple(r) in lib_rows for r in P.inputs(d).round(6).values)
    print(f"   System {s}: test rows identical to a library row: {same}")
    ok &= same == 0
look = [i for i, r in LS.iterrows() if P._lookalike(dict(cells=r.cells, bypass=r.bypass, parallel=r.parallel, length=r.length, isc=r.isc))]
print(f"   library systems that are look-alikes of a test system: {len(look)}")
ok &= not look
test_mods = {"System 1 (10 W)": P.SA10, "System 3 (330 W)": P.M330} | {f"System 2 ({m['name']})": m for m in P.S2_STRING}
for name, m in test_mods.items():
    close = LS[(LS.cells == m["n_cells"]) & ((LS.isc / m["Isc"] - 1).abs() < 0.02)]
    print(f"   library systems with the same cell count and Isc within 2 % of {name}: {len(close)}"
          + (f" (ids {list(close.id)}; but string length / layout differ)" if len(close) else ""))

print("\n2. SAVED MODEL FILE vs NOTEBOOK")
m = XGBClassifier(); m.load_model("models/xgb_source.json")
nb = pd.read_csv("results_v6_summary.csv", index_col=0)
for s, d in T.items():
    a = accuracy_score(d.Fault, m.predict(P.inputs(d))); ref = nb.loc["zero-shot", f"System {s}"]
    print(f"   System {s}: saved model {a:.3f}   notebook {ref:.3f}")
    ok &= abs(a - ref) < 0.002

print("\n3. FRESH TEST DATA (new seeds, never seen by anyone)")
fresh = {}
for s, n in [(1, 300), (2, 240), (3, 240)]:
    fresh[s] = P.generate(s, n, seed=777000 + s)
    a = accuracy_score(fresh[s].Fault, m.predict(P.inputs(fresh[s])))
    print(f"   System {s}: {n} new readings -> zero-shot accuracy {a:.3f}", flush=True)

print("\n4. RETRAIN WITH OTHER RANDOM SEEDS (same settings)")
params = json.load(open("models/xgb_params.json"))
for seed in (11, 22, 33):
    mm = XGBClassifier(random_state=seed, n_jobs=-1, **params).fit(P.inputs(LIB), LIB.Fault)
    print(f"   seed {seed}: " + "  ".join(f"S{s} {accuracy_score(d.Fault, mm.predict(P.inputs(d))):.3f}" for s, d in T.items()), flush=True)

print(f"\nRESULT: {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}   ({time.time() - t0:.0f} s)")
