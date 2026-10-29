"""
pv_simulator_v3.py  --  PV systems 1-3 (+ generic training strings), 6 fault classes, I-V tracer + high-resolution sensor

Builds on pv_simulator_v2.py and reuses its cell-level physics unchanged
(single-diode cells -> substrings with bypass diodes -> modules -> strings), its calibrated
shading/dirt models, and its MPP search. What v3 adds:

  * Systems 3 and 4 (typical datasheet values, no real measurements exist for them)
  * three electrical faults: open circuit, short circuit, degradation (extra series resistance)
  * one measurement chain for every system: I-V tracer + high-resolution sensor
  * a single function, measure(), that the demo website also calls, so the demo produces
    readings exactly the way the training data was produced
  * expected-healthy ratios (measured / datasheet-model value at the same sunlight and temperature) and
    I-V curve-shape features (series-resistance slope, shunt slope, number of curve steps)
  * generic rooftop strings (random module types and string lengths, wider fault ranges) used only for
    training, so the model has seen bypass-diode physics before it meets Systems 2 and 3

Classes: 0 normal, 1 partial shading, 2 dirt, 3 open circuit, 4 short circuit, 5 degradation

Calibration rule (same as v2): fault parameter ranges describe plausible physical faults;
they are never tuned against any model's accuracy.
"""
import numpy as np
import pandas as pd
import pv_simulator_v2 as V2

CLASSES = ["normal", "partial shading", "dirt", "open circuit", "short circuit", "degradation"]
F4 = ["Voc/MaxVoc", "Isc/MaxIsc", "G/1000", "AT/50"]
IV = ["Vmp/MaxVoc", "Imp/MaxIsc", "FF"]

# ------------------------------------------------------------------ module specs
SA10 = V2.SA10                        # System 1 (paper)
S2_STRING = V2.SYSTEM2_STRING         # System 2 (paper, Table 2)
# System 3: typical 330 W, 60-cell module (residential rooftop class), 3 bypass diodes
M330 = dict(name="330W-60c", Voc=40.5, Isc=10.30, Vmp=33.6, Imp=9.82,
            n_cells=60, n_bypass=3, n_ideal=1.3, Rs_mod=0.06, Rsh_mod=300.0)   # Rs set so Pmp(STC) ~ 330 W
# System 4: typical 400 W, 72-cell module (commercial class), 3 bypass diodes
M400 = dict(name="400W-72c", Voc=49.5, Isc=10.40, Vmp=41.0, Imp=9.76,
            n_cells=72, n_bypass=3, n_ideal=1.3, Rs_mod=0.14, Rsh_mod=350.0)   # Rs set so Pmp(STC) ~ 400 W

SYSTEMS = {
    1: dict(name="System 1 - 2 x 10 W test modules (paper)", branches={"series": [[SA10, SA10]], "parallel": [[SA10], [SA10]]}),
    2: dict(name="System 2 - 7 x 200 W rooftop string (paper)", branches={"series": [list(S2_STRING)]}),
    3: dict(name="System 3 - 10 x 330 W residential string", branches={"series": [[M330] * 10]}),
    4: dict(name="System 4 - 2 parallel strings x 8 x 400 W", branches={"series": [[M400] * 8, [M400] * 8]}),
}
# Operating conditions: drawn uniformly over a realistic year-round range, the same for every class (so weather carries
# no information about the fault). The paper's measured ranges (System 1: 238-920 W/m2, 19.6-25.6 C; System 2: 709-831 W/m2,
# 14.3-28.8 C) were used in the October study and in v5; v6 uses the wide range for every test system.
ENV_WIDE = dict(G=(250.0, 1000.0), Ta=(15.0, 40.0))
ENV_PAPER = {1: dict(G=(238.0, 920.0), Ta=(19.62, 25.56)), 2: dict(G=(709.0, 831.0), Ta=(14.31, 28.77))}
ENV_SYS = {s: dict(ENV_WIDE) for s in (1, 2, 3, 4)}

# high-resolution I-V tracer: 8 readings averaged, 0.1 % V and 0.5 % I uncertainty (INA226-class)
SENSOR = dict(v_noise=0.001, i_noise=0.005, n_avg=8, g_noise=0.005)
# paper's System 1 meter (DSN VC288): 0.1 V / 0.01 A steps, +/-0.1 % V, +/-1 % I, single reading
SENSOR_PAPER = dict(v_noise=0.001, i_noise=0.01, n_avg=1, g_noise=0.01, v_res=0.1, i_res=0.01)
# measurement chain options (module-level so datasets and the demo use the same choice):
#   S1_SENSOR: "hires" or "paper"     ROUTE (Systems 2-4): "tracer" (I-V tracer) or "inverter" (paper: MPP -> Voc/Isc)
MEAS = dict(S1_SENSOR="hires", ROUTE="tracer")


def layout(system, config="series"):
    """Branches in parallel; each branch is a list of module specs in series."""
    b = SYSTEMS[system]["branches"]
    return b[config if config in b else "series"]


def norm_limits(system, config="series"):
    br = layout(system, config)
    if system == 1:
        max_voc = SA10["Voc_nameplate"] * len(br[0])
    else:
        max_voc = sum(s["Voc"] for s in br[0])
    max_isc = sum(min(s["Isc"] for s in b) for b in br)
    return max_voc, max_isc


# ------------------------------------------------------------------ circuit
def module_vi_ext(spec, cell_G, Tc, I_grid, shorted=()):
    """v2.module_vi, plus shorted substrings (failed-short bypass diode) or a shorted module ('all')."""
    p = V2._cell_params(spec, Tc)
    Iph_cells = spec["Isc"] * cell_G / 1000.0 * (1 + V2.ALPHA_ISC * (Tc - 25.0))
    Ns, nb = spec["n_cells"], spec["n_bypass"]
    groups = np.array_split(np.arange(Ns), max(nb, 1))
    V_mod = np.zeros_like(I_grid)
    if "all" in shorted:
        return V_mod
    for gi, g in enumerate(groups):
        if gi in shorted:
            continue                                   # substring shorted -> contributes 0 V
        V_sub = np.zeros_like(I_grid)
        vals, counts = np.unique(np.round(Iph_cells[g], 3), return_counts=True)
        for Iph, c in zip(vals, counts):
            V_sub += c * V2._cell_voltage(I_grid, Iph, p)
        if nb > 0:
            V_sub = np.maximum(V_sub, -V2.V_BYPASS)
        V_mod += V_sub
    return V_mod


def measure(system, G, Ta, rng, config="series", maps=None, open_branch=None, short=None,
            r_frac=0.0, r_branch=0, Tc_offset=None, curve=False, noiseless=False):
    """Simulate one I-V tracer measurement.
    maps        : per-cell irradiance (W/m2) for every module, flattened over branches (None = uniform G)
    open_branch : index of a branch whose wire is broken (single-branch systems -> whole array dead)
    short       : list of (branch, module, substring or 'all') that are short-circuited
    r_frac      : degradation, extra series resistance in branch r_branch, as a fraction of Vmp/Imp
    Returns features dict (normalised like the paper) and optionally the I-V curve."""
    br = layout(system, config)
    heat, tsd = (V2.HEAT[1], V2.TC_SD[1]) if system == 1 else (V2.HEAT[2], V2.TC_SD[2])
    Tc = Ta + G * heat + (rng.normal(0, tsd) if Tc_offset is None else Tc_offset)
    flat = [m for b in br for m in b]
    if maps is None:
        maps = [np.full(s["n_cells"], float(G)) for s in flat]
    short = short or []
    k = 0
    branch_curves, dead_short = [], False
    for bi, mods in enumerate(br):
        bmaps = maps[k:k + len(mods)]; k += len(mods)
        if open_branch is not None and bi == open_branch:
            continue
        sh = [(m, s) for (b, m, s) in short if b == bi]
        if len(sh) and all(any(mm == mi and s == "all" for (mm, s) in sh) for mi in range(len(mods))):
            dead_short = True                         # whole branch shorted -> wire across the terminals
        I_grid = np.linspace(0, 1.3 * max(s["Isc"] for s in mods), 700)
        V = np.zeros_like(I_grid)
        for mi, (spec, cg) in enumerate(zip(mods, bmaps)):
            V += module_vi_ext(spec, np.asarray(cg, float), Tc, I_grid, shorted={s for (mm, s) in sh if mm == mi})
        if r_frac > 0 and bi == r_branch:
            R = r_frac * sum(s["Vmp"] for s in mods) / min(s["Imp"] for s in mods)
            V = V - I_grid * R
        branch_curves.append((V, I_grid))

    max_voc, max_isc = norm_limits(system, config)
    if not branch_curves:                             # open circuit, nothing connected
        voc = isc = vmp = imp = 0.0; Vc = Ic = np.zeros(2)
    elif dead_short:                                  # terminals shorted: V = 0, current = sum of the others
        voc = 0.0
        isc = sum(V2.voc_isc_series(V, I)[1] for V, I in branch_curves if V[0] > 0)
        vmp = imp = 0.0; Vc, Ic = np.array([0.0, 0.0]), np.array([isc, 0.0])
    elif len(branch_curves) == 1:
        V, I = branch_curves[0]
        voc, isc = V2.voc_isc_series(V, I)
        vmp, imp = V2.mpp(V, I, rng, p_local=0.0)
        ok = V >= 0; Vc, Ic = V[ok][::-1], I[ok][::-1]
    else:                                             # parallel strings (blocking diodes: no reverse current)
        Vc = np.linspace(0, max(c[0][0] for c in branch_curves), 1500)
        Ic = np.zeros_like(Vc)
        for V, I in branch_curves:                    # each branch: its curve for V >= 0 plus its own Isc point
            isc_b = V2.voc_isc_series(V, I)[1]
            ok = V >= 0
            Vb, Ib = np.r_[V[ok], 0.0], np.r_[I[ok], isc_b]
            o = np.argsort(Vb, kind="stable")
            Ic += np.interp(Vc, Vb[o], Ib[o], left=isc_b, right=0.0)
        isc = Ic[0]
        voc = float(np.interp(0.0, Ic[::-1], Vc[::-1]))
        vmp, imp = V2.mpp(Vc, Ic, rng, p_local=0.0)

    sp = SENSOR_PAPER if (system == 1 and MEAS["S1_SENSOR"] == "paper") else SENSOR
    sv, si = sp["v_noise"] / np.sqrt(sp["n_avg"]), sp["i_noise"] / np.sqrt(sp["n_avg"])
    v_res = sp.get("v_res", 0.01)
    i_res = sp.get("i_res", 0.001 if system == 1 else 0.005)
    noisy = lambda x, s, res: V2.quantize(x * (1 + rng.normal(0, s)), res) if x > 0 else 0.0
    if noiseless:                                     # reference calculation (expected healthy values): no sensor model
        noisy = lambda x, s, res: float(x)
    if system != 1 and MEAS["ROUTE"] == "inverter":
        # paper's System 2 route: the inverter's MPPT point (may lock on a local peak under shading),
        # converted to Voc/Isc with the modules' fill-factor ratios (paper Sec. V-B)
        if branch_curves and not dead_short and voc > 0:
            vmp, imp = V2.mpp(Vc, Ic, rng)
        kv, ki = V2._fill_factor_ratios(flat)
        voc = vmp * kv * (1 + rng.normal(0, 0.005)) if vmp > 0 else 0.0
        isc = imp * ki * (1 + rng.normal(0, 0.01)) if imp > 0 else 0.0
        vmp, imp = noisy(vmp, 0.005, 0.01), noisy(imp, 0.01, 0.005)
    else:
        voc, vmp = noisy(voc, sv, v_res), noisy(vmp, sv, v_res)
        isc, imp = noisy(isc, si, i_res), noisy(imp, si, i_res)
    ff = vmp * imp / (voc * isc) if voc > 0 and isc > 0 else 0.0
    G_meas = G if noiseless else V2.quantize(G * (1 + rng.normal(0, sp["g_noise"])), 1.0)
    Ta_meas = Ta if noiseless else V2.quantize(Ta + rng.normal(0, 0.2), 0.0625)
    feats = {"Voc/MaxVoc": voc / max_voc, "Isc/MaxIsc": isc / max_isc, "G/1000": G_meas / 1000, "AT/50": Ta_meas / 50,
             "Vmp/MaxVoc": vmp / max_voc, "Imp/MaxIsc": imp / max_isc, "FF": ff}
    feats.update(curve_shape(Vc, Ic, rng, noiseless))
    out = dict(features=feats, raw=dict(Voc=voc, Isc=isc, Vmp=vmp, Imp=imp, Tc=Tc, G=G_meas, Ta=Ta_meas))
    if curve:
        idx = np.linspace(0, len(Vc) - 1, min(120, len(Vc))).astype(int)
        out["curve"] = dict(V=np.round(Vc[idx], 4).tolist(), I=np.round(Ic[idx], 5).tolist())
    return out


# ------------------------------------------------------------------ I-V curve shape (from the tracer sweep)
def curve_shape(Vc, Ic, rng, noiseless=False, n_pts=100):
    """Features read from the whole traced curve, as in I-V-curve fault-diagnosis papers:
       Rs_n   : slope -dV/dI near Voc, normalised by Voc/Isc  (series resistance: corroded connectors, degradation)
       Rsh_n  : slope -dI/dV near Isc, normalised by Isc/Voc  (shunt leakage / mismatch near short circuit)
       Steps  : number of power peaks on the P-V curve        (bypass diodes switching on: shading, some dirt)
    The tracer samples n_pts points, each with the high-resolution sensor's noise."""
    Vc, Ic = np.asarray(Vc, float), np.asarray(Ic, float)
    if len(Vc) < 5 or Vc.max() <= 0 or Ic.max() <= 0:
        return {"Rs_n": 0.0, "Rsh_n": 0.0, "Steps": 0.0}
    o = np.argsort(Vc); Vc, Ic = Vc[o], Ic[o]
    v = np.linspace(0, Vc.max(), n_pts); i = np.interp(v, Vc, Ic)
    if not noiseless:
        v = v * (1 + rng.normal(0, SENSOR["v_noise"] / np.sqrt(SENSOR["n_avg"]), n_pts))
        i = i * (1 + rng.normal(0, SENSOR["i_noise"] / np.sqrt(SENSOR["n_avg"]), n_pts))
    # two-point slopes read on the traced curve (each point carries the sensor noise)
    isc0, voc0 = Ic.max(), Vc.max()
    Vd = Vc[::-1]; Id = Ic[::-1]                                  # current ascending for V(I) interpolation
    nz = lambda x, sd: x if noiseless else x * (1 + rng.normal(0, sd))
    sv_, si_ = SENSOR["v_noise"] / np.sqrt(SENSOR["n_avg"]), SENSOR["i_noise"] / np.sqrt(SENSOR["n_avg"])
    va, vb = (nz(np.interp(f * isc0, Id, Vd), sv_) for f in (0.03, 0.20))
    rs = (va - vb) / (0.17 * isc0) * isc0 / voc0
    ia, ib = (nz(np.interp(f * voc0, Vc, Ic), si_) for f in (0.0, 0.25))
    rsh = (ia - ib) / (0.25 * voc0) * voc0 / isc0
    p = np.convolve(v * i, np.ones(3) / 3, mode="same")
    pk = [k for k in range(2, n_pts - 2) if p[k] >= p[k - 1] and p[k] > p[k + 1] and p[k] > 0.1 * p.max()]
    return {"Rs_n": float(rs), "Rsh_n": float(rsh), "Steps": float(len(pk))}


# ------------------------------------------------------------------ random faults (datasets)
P_ELEC = dict(short_subs=(2, 6), r_frac=(0.08, 0.35))
# Generic training strings use deliberately WIDER fault ranges than Systems 2-3 (so the test systems' exact fault
# distributions are never seen in training).
P2_WIDE = dict(shade_sub=(3, 12), shade_tau=(0.30, 0.70), dirt_uniform=(0.90, 1.00), dirt_frac=(0.0, 0.04),
               dirt_patch_tau=(0.5, 0.95), dirt_heavy_sub=(0, 2), dirt_heavy_tau=(0.10, 0.50))
P_ELEC_WIDE = dict(short_subs=(1, 8), r_frac=(0.05, 0.45))
GENERIC = set()


def add_generic_systems(n=12, seed=7):
    """Register n generic rooftop strings (random 60/72-cell modules, 5-14 in series, 3 bypass diodes each).
    Ids 100, 101, ... Deterministic for a given seed. Returns the list of ids."""
    rng = np.random.default_rng(seed); ids = []
    for i in range(n):
        cells = int(rng.choice([60, 72])); voc = rng.uniform(0.60, 0.69) * cells; isc = rng.uniform(5.0, 11.5)
        m = dict(name=f"generic-{cells}c", Voc=voc, Isc=isc, Vmp=voc * rng.uniform(0.80, 0.84), Imp=isc * rng.uniform(0.93, 0.96),
                 n_cells=cells, n_bypass=3, n_ideal=1.3, Rs_mod=rng.uniform(0.05, 0.45), Rsh_mod=rng.uniform(250, 500))
        sid = 100 + i; length = int(rng.integers(5, 15))
        SYSTEMS[sid] = dict(name=f"Generic string {i + 1} - {length} x {cells}-cell", branches={"series": [[m] * length]})
        ENV_SYS[sid] = dict(G=(400.0, 1000.0), Ta=(10.0, 35.0))
        GENERIC.add(sid); ids.append(sid)
    return ids


# ------------------------------------------------------------------ generic training library (v6)
# Random PV systems built from the general building blocks of crystalline-silicon string systems. Ranges come from what is
# sold and installed (small panels, 36/60/72-cell modules, homes and commercial roofs), NOT from Systems 1-4. Systems too
# similar to a test system are removed. Faults range from a detection threshold (a noticeable loss) up to large faults.
LIBRARY = {}                     # id -> description dict
ENV_LIB = dict(G=(200.0, 1050.0), Ta=(10.0, 45.0))
TEST_SIGNATURES = [dict(cells=36, bypass=0, parallel=1, length=2, isc=0.60),      # System 1 series
                   dict(cells=36, bypass=0, parallel=2, length=1, isc=0.60),      # System 1 parallel
                   dict(cells=72, bypass=3, parallel=1, length=7, isc=5.7),       # System 2
                   dict(cells=60, bypass=3, parallel=1, length=10, isc=10.3),     # System 3
                   dict(cells=72, bypass=3, parallel=2, length=8, isc=10.4)]      # System 4


def _lookalike(d):
    return any(d["cells"] == t["cells"] and d["bypass"] == t["bypass"] and d["parallel"] == t["parallel"]
               and abs(d["length"] - t["length"]) <= 2 and abs(d["isc"] / t["isc"] - 1) <= 0.25 for t in TEST_SIGNATURES)


def _random_module(rng, kind):
    if kind == "small":        # small panels (5-120 W), 36 cells, no bypass diodes
        cells, nb, isc, vcell = 36, 0, rng.uniform(0.3, 6.5), rng.uniform(0.57, 0.64)
    elif kind == "36c":        # 36-cell 50-150 W panels with 2 bypass diodes
        cells, nb, isc, vcell = 36, 2, rng.uniform(3.0, 8.5), rng.uniform(0.59, 0.64)
    else:                      # standard 60/72-cell panels with 3 bypass diodes
        cells, nb, isc, vcell = int(rng.choice([60, 72])), 3, rng.uniform(5.0, 11.5), rng.uniform(0.60, 0.69)
    voc = vcell * cells; z = voc / isc                       # resistances scale with the module's voltage/current ratio
    # module quality: from new, low-loss modules to older / cheaper ones (higher series, lower shunt resistance)
    return dict(name=f"{cells}c-{isc:.1f}A", Voc=voc, Isc=isc, Vmp=voc * rng.uniform(0.77, 0.84), Imp=isc * rng.uniform(0.91, 0.96),
                n_cells=cells, n_bypass=nb, n_ideal=rng.uniform(1.1, 1.4), Rs_mod=rng.uniform(0.010, 0.075) * z,
                Rsh_mod=float(np.exp(rng.uniform(np.log(25), np.log(150)))) * z)


def _variant(rng, m, k):
    """Another module model of the same cell count (replacement modules / mixed batches in one string)."""
    v = dict(m); v["Voc"] = m["Voc"] * rng.uniform(0.94, 1.06); v["Isc"] = m["Isc"] * rng.uniform(0.88, 1.12)
    v["Vmp"] = v["Voc"] * rng.uniform(0.77, 0.84); v["Imp"] = v["Isc"] * rng.uniform(0.91, 0.96)
    z = v["Voc"] / v["Isc"]; v["Rs_mod"] = rng.uniform(0.010, 0.075) * z; v["Rsh_mod"] = float(np.exp(rng.uniform(np.log(25), np.log(150)))) * z
    v["name"] = f"{m['name']}/{k}"; return v


def make_library(n=90, seed=11):
    """Register n random library systems (ids 1000, 1001, ...) and return their ids. Look-alikes of Systems 1-4 are skipped."""
    rng = np.random.default_rng(seed); ids = []; sid = 1000
    while len(ids) < n:
        kind = rng.choice(["small", "36c", "std"], p=[0.25, 0.15, 0.60])
        m = _random_module(rng, kind)
        length = int(rng.integers(1, 9)) if kind == "small" else int(rng.integers(2, 13)) if kind == "36c" else int(rng.integers(4, 21))
        par = int(rng.choice([1, 2, 3], p=[0.55, 0.3, 0.15]))
        d = dict(cells=m["n_cells"], bypass=m["n_bypass"], parallel=par, length=length, isc=m["Isc"])
        if _lookalike(d):
            continue
        mixed = length > 1 and rng.random() < 0.35            # 35 %: a string of 2-3 different module models
        if mixed:
            pool = [m] + [_variant(rng, m, k) for k in range(1, int(rng.integers(2, 4)))]
            string = [pool[int(i)] for i in rng.integers(0, len(pool), length)]; string[0] = m
        else:
            string = [m] * length
        d["mixed"] = bool(mixed)
        SYSTEMS[sid] = dict(name=f"Library {sid}: {par} x {length} x {m['name']} ({m['n_bypass']} bypass{', mixed modules' if mixed else ''})",
                            branches={"series": [list(string) for _ in range(par)]})
        ENV_SYS[sid] = dict(ENV_LIB); LIBRARY[sid] = dict(d, kind=str(kind)); ids.append(sid); sid += 1
    return ids


def sample_fault_library(system, fault, G, rng):
    """Faults for library systems: from a detection threshold (noticeable loss) up to large faults."""
    br = layout(system); flat = [m for b in br for m in b]; nb = flat[0]["n_bypass"]
    maps = [np.full(m["n_cells"], float(G)) for m in flat]
    if fault == 1:                                            # partial shading
        if nb == 0:                                           # no bypass diodes: a shadow on some cells of 1-2 modules
            for mi in rng.choice(len(maps), min(len(maps), int(rng.integers(1, 3))), replace=False):
                k = int(rng.integers(1, int(0.4 * maps[mi].size) + 1)); cells = rng.choice(maps[mi].size, k, replace=False)
                maps[mi][cells] *= np.clip(rng.uniform(0.3, 0.7) + rng.normal(0, 0.03, k), 0.1, 1)
        else:                                                 # a shadow over a run of neighbouring substrings
            subs = [(mi, g) for mi, m in enumerate(flat) for g in np.array_split(np.arange(m["n_cells"]), nb)]
            lo = max(2, int(np.ceil(0.05 * len(subs)))); hi = max(lo, int(np.ceil(0.5 * len(subs))))
            n = int(rng.integers(lo, hi + 1)); first = int(rng.integers(0, len(subs) - n + 1)); tau = rng.uniform(0.3, 0.7)
            for mi, g in subs[first:first + n]:
                cover = g[: int(rng.integers(max(1, len(g) // 3), len(g) + 1))]
                maps[mi][cover] *= np.clip(tau + rng.normal(0, 0.03, cover.size), 0.1, 1)
    elif fault == 2:                                          # dirt, two styles seen on real arrays:
        splashed = nb > 0 and rng.random() < 0.5              # (a) dust film (3-25 % loss) + patches (+ 0-2 heavy blotches)
        film = rng.uniform(0.95, 1.0) if splashed else rng.uniform(0.75, 0.97)   # (b) splashed: light film + 1-2 heavy blotches
        for m in maps:
            m *= np.clip(film + rng.normal(0, 0.02), 0.5, 1) * np.clip(1 + rng.normal(0, 0.015, m.size), 0.4, 1.0)
            cells = rng.choice(m.size, int(rng.uniform(0, 0.05) * m.size), replace=False); m[cells] *= rng.uniform(0.5, 0.95, cells.size)
        if nb > 0:
            subs = [(mi, g) for mi, m in enumerate(flat) for g in np.array_split(np.arange(m["n_cells"]), nb)]
            for j in rng.choice(len(subs), min(len(subs), int(rng.integers(1, 3) if splashed else rng.integers(0, 3))), replace=False):
                mi, g = subs[j]; cov = rng.choice(g, max(1, int(rng.uniform(0.2, 0.6) * g.size)), replace=False)
                maps[mi][cov] *= rng.uniform(0.1, 0.5)
    if fault in (1, 2):
        for m in maps:
            np.round(m / 5.0, out=m); m *= 5.0
        return dict(maps=maps)
    b = int(rng.integers(len(br)))
    if fault == 3:
        return dict(open_branch=b)
    if fault == 4:
        mods = br[b]
        if nb == 0 or len(mods) == 1 and nb == 0:
            return dict(short=[(b, int(rng.integers(len(mods))), "all")])
        subs = [(b, mi, g) for mi in range(len(mods)) for g in range(nb)]
        lo = max(1, int(np.ceil(0.04 * len(subs)))); hi = max(lo, int(np.ceil(0.3 * len(subs))))
        pick = rng.choice(len(subs), int(rng.integers(lo, hi + 1)), replace=False)
        return dict(short=[subs[i] for i in pick])
    if fault == 5:
        return dict(r_frac=float(rng.uniform(0.07, 0.5)), r_branch=b)
    return {}


def sample_fault(system, fault, G, rng, config="series"):
    """Random physical fault of class `fault` -> keyword arguments for measure()."""
    if system in LIBRARY:
        return sample_fault_library(system, fault, G, rng)
    br = layout(system, config)
    flat = [m for b in br for m in b]
    pe = P_ELEC_WIDE if system in GENERIC else P_ELEC
    kw = {}
    if fault in (1, 2):
        saved = dict(V2.P2)
        if system in GENERIC: V2.P2.update(P2_WIDE)
        try:
            kw["maps"] = V2.make_fault_maps(flat, G, fault, rng, system=1 if system == 1 else 2)
        finally:
            V2.P2.update(saved)
    elif fault == 3:
        kw["open_branch"] = int(rng.integers(len(br)))
    elif fault == 4:
        b = int(rng.integers(len(br)))
        mods = br[b]
        if mods[0]["n_bypass"] == 0:                  # System 1: no bypass diodes -> a whole module shorts
            kw["short"] = [(b, int(rng.integers(len(mods))), "all")]
        else:                                         # failed-short bypass diodes / shorted junction boxes
            subs = [(b, m, s) for m in range(len(mods)) for s in range(mods[m]["n_bypass"])]
            n = int(rng.integers(pe["short_subs"][0], pe["short_subs"][1] + 1))
            pick = rng.choice(len(subs), n, replace=False)
            kw["short"] = [subs[i] for i in pick]
    elif fault == 5:
        kw["r_frac"] = float(rng.uniform(*pe["r_frac"]))
        kw["r_branch"] = int(rng.integers(len(br)))
    return kw


def generate(system, n, seed, classes=range(6), env=None):
    """env: (G_min, G_max, Ta_min, Ta_max); default ENV_SYS[system]. classes cycle 0,1,2,... so the data is balanced."""
    env = env or (*ENV_SYS[system]["G"], *ENV_SYS[system]["Ta"])
    rng = np.random.default_rng(seed)
    classes = list(classes)
    rows = []
    for i in range(n):
        fault = classes[i % len(classes)]
        config = ("series" if (i // len(classes)) % 2 == 0 else "parallel") if system == 1 else "series"
        G, Ta = rng.uniform(env[0], env[1]), rng.uniform(env[2], env[3])
        r = measure(system, G, Ta, rng, config=config, **sample_fault(system, fault, G, rng, config))
        rows.append({**r["features"], **residual_row(system, config, r["features"]), "Fault": fault, "Config": config, "System": system})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ expected (healthy) performance
_EXP_CACHE = {}

def expected(system, config, G, Ta):
    """What a HEALTHY array of this system should measure at the measured irradiance and air temperature.
    Uses only the datasheet model of the system (the same single-diode model, no faults, no sensor noise, cell temperature
    from the standard heating rule). A real monitor computes this from the datasheet the same way.
    Returns dict Voc, Isc, Vmp, Imp, FF (normalised like the measured features)."""
    key = (system, config, round(float(G), 0), round(float(Ta), 1))
    if key not in _EXP_CACHE:
        r = measure(system, key[2], key[3], np.random.default_rng(0), config=config, Tc_offset=0.0, noiseless=True)["features"]
        _EXP_CACHE[key] = dict(Voc=r["Voc/MaxVoc"], Isc=r["Isc/MaxIsc"], Vmp=r["Vmp/MaxVoc"], Imp=r["Imp/MaxIsc"], FF=r["FF"],
                               Rs=r["Rs_n"])
    return _EXP_CACHE[key]


def residual_row(system, config, f):
    """Expected-healthy ratios for one measurement (dict of measured features)."""
    e = expected(system, config, f["G/1000"] * 1000, f["AT/50"] * 50)
    q = lambda a, b: a / b if b > 1e-9 else 0.0
    return {"Voc_r": q(f["Voc/MaxVoc"], e["Voc"]), "Isc_r": q(f["Isc/MaxIsc"], e["Isc"]), "Vmp_r": q(f["Vmp/MaxVoc"], e["Vmp"]),
            "Imp_r": q(f["Imp/MaxIsc"], e["Imp"]), "FF_r": q(f["FF"], e["FF"]),
            "P_r": q(f["Vmp/MaxVoc"] * f["Imp/MaxIsc"], e["Vmp"] * e["Imp"]), "Rs_r": q(f.get("Rs_n", 0.0), e["Rs"]),
            "Strings": float(len(layout(system, config)))}     # number of parallel strings (known from the installation)


def residuals(df, system, config=None):
    """Measured / expected-healthy ratios. About 1.0 for a healthy array on ANY system, so the same fault gives similar
    values on every system (this is what makes a model trained on one system usable on another)."""
    rows = []
    for i, r in df.iterrows():
        cfg = config or r.get("Config", "series")
        e = expected(system, cfg, r["G/1000"] * 1000, r["AT/50"] * 50)
        q = lambda a, b: a / b if b > 1e-9 else 0.0
        rows.append({"Voc_r": q(r["Voc/MaxVoc"], e["Voc"]), "Isc_r": q(r["Isc/MaxIsc"], e["Isc"]),
                     "Vmp_r": q(r["Vmp/MaxVoc"], e["Vmp"]), "Imp_r": q(r["Imp/MaxIsc"], e["Imp"]),
                     "FF_r": q(r["FF"], e["FF"]),
                     "P_r": q(r["Vmp/MaxVoc"] * r["Imp/MaxIsc"], e["Vmp"] * e["Imp"]),
                     **({"Rs_r": q(r["Rs_n"], e["Rs"])} if "Rs_n" in r else {})})
    return pd.DataFrame(rows, index=df.index)


# ------------------------------------------------------------------ model inputs
def features(df):
    """The 7 measured features + the physics features used by XGBoost (same names as the v3 notebooks)."""
    x = df[F4 + IV].copy().astype(float)
    safe = lambda a, b: np.where(b > 1e-9, a / np.where(b > 1e-9, b, 1.0), 0.0)
    x["Isc/G"] = safe(x["Isc/MaxIsc"], x["G/1000"])
    x["Vmp/Voc"] = safe(x["Vmp/MaxVoc"], x["Voc/MaxVoc"])
    x["Imp/Isc"] = safe(x["Imp/MaxIsc"], x["Isc/MaxIsc"])
    x["Imp/G"] = safe(x["Imp/MaxIsc"], x["G/1000"])
    return x


TEST_IDS = (1, 2, 3, 4)
CURVE = ["Rs_n", "Rsh_n", "Steps"]
RESID = ["Voc_r", "Isc_r", "Vmp_r", "Imp_r", "FF_r", "P_r", "Rs_r"]
TOPO = ["Strings"]


def inputs(df, topology=True):
    """Final model inputs (v6, 22): 7 measured features + 4 physics ratios + 3 curve-shape features + 7 expected-healthy ratios
    + the number of parallel strings. The dataset CSVs already contain the curve and ratio columns (computed when the reading
    was made). topology=False gives the 21 inputs of the earlier versions (v5, v6a, v6b)."""
    parts = [features(df), df[CURVE + RESID].astype(float)]
    if topology:
        if "Strings" in df:
            parts.append(df[TOPO].astype(float))
        else:                                                       # older CSVs of the test systems: add it from the layout
            assert set(df["System"]) <= set(TEST_IDS), "Strings column missing"
            parts.append(pd.DataFrame({"Strings": [float(len(layout(s, c))) for s, c in zip(df["System"], df.get("Config", ["series"] * len(df)))]}, index=df.index))
    return pd.concat(parts, axis=1)


if __name__ == "__main__":
    import argparse, os, shutil, time
    ap = argparse.ArgumentParser(description="Generate the v6 datasets: test Systems 1-4 and the generic training library")
    ap.add_argument("--out", default="data_v6")
    ap.add_argument("--seed", type=int, default=3026)
    ap.add_argument("--rows_per_library_system", type=int, default=66)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for s, n in [(1, 1002), (2, 402), (3, 402), (4, 402)]:          # TEST systems (never used for training)
        f = f"{a.out}/S{s}_6class_{n}.csv"
        if not os.path.exists(f):
            generate(s, n, a.seed + s).round(6).to_csv(f, index=False)
        print("System", s, "->", f, flush=True)
    ids = make_library(); t = time.time(); parts = []
    for k, sid in enumerate(ids):                                     # TRAINING library
        parts.append(generate(sid, a.rows_per_library_system, a.seed + sid))
        if k % 10 == 9: print(f"library {k + 1}/{len(ids)} systems, {time.time() - t:.0f} s", flush=True)
    lib = pd.concat(parts, ignore_index=True)
    lib.round(6).to_csv(f"{a.out}/library_6class_{len(lib)}.csv", index=False)
    pd.DataFrame([dict(id=i, name=SYSTEMS[i]["name"], **LIBRARY[i]) for i in ids]).to_csv(f"{a.out}/library_systems.csv", index=False)
    print("Library:", len(ids), "systems,", len(lib), "rows")
