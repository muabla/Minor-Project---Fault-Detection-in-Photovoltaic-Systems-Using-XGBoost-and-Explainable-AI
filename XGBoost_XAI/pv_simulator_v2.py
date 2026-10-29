"""
pv_simulator.py
Physics-based simulator for PV fault data (normal / partial shading / dirt).

Every sample is produced by solving a cell-level single-diode circuit:
  cells -> substrings (optional bypass diode) -> modules -> series/parallel array
Faults are injected physically:
  * partial shading : some cells of some modules receive reduced irradiance
  * dirt (soiling)  : optical transmittance loss over the whole module surface
The electrical outputs (Voc, Isc) then pass through a sensor model
(resolution + uncertainty from the base paper's instruments) and the paper's
normalization (Voc/MaxVoc, Isc/MaxIsc, G/1000, AT/50).

Calibration rule used for this project: physical/environmental parameter ranges
are chosen so simulated feature DISTRIBUTIONS look like the real measurements
(checked with a real-vs-sim classifier test). They are never tuned against
any ML model's accuracy.

Usage:
    python pv_simulator.py --real1 Data_300.csv --real2 Data_60.csv --out ./data
"""
import argparse
import numpy as np
import pandas as pd

K_B, Q = 1.380649e-23, 1.602176634e-19
EG = 1.12  # eV, silicon bandgap
FEATURES = ["Voc/MaxVoc", "Isc/MaxIsc", "G/1000", "AT/50"]

# ---------------------------------------------------------------- module specs
# System 1: Sinosola SA10-36P (paper, Sec. V-A)
SA10 = dict(name="SA10-36P", Voc=21.24, Voc_nameplate=22.5,  # Voc derated so the simulated fault-free class matches the measured one (with HEAT[1])
            Isc=0.60, Vmp=19.0, Imp=0.50,
            n_cells=36, n_bypass=0, n_ideal=1.3, Rs_mod=1.2, Rsh_mod=5000.0)
# System 2: three 200 W module types (paper, Table 2); 72-cell, 3 bypass diodes
S2_A = dict(name="200W-A", Voc=46.5, Isc=5.62, Vmp=37.2, Imp=5.38,
            n_cells=72, n_bypass=3, n_ideal=1.3, Rs_mod=0.45, Rsh_mod=400.0)
S2_B = dict(name="200W-B", Voc=43.2, Isc=6.35, Vmp=36.0, Imp=5.89,
            n_cells=72, n_bypass=3, n_ideal=1.3, Rs_mod=0.35, Rsh_mod=350.0)
S2_C = dict(name="200W-C", Voc=45.2, Isc=5.32, Vmp=37.0, Imp=5.41,
            n_cells=72, n_bypass=3, n_ideal=1.3, Rs_mod=0.45, Rsh_mod=400.0)
SYSTEM2_STRING = [S2_A] * 4 + [S2_B] * 2 + [S2_C]          # 7 modules in series

ALPHA_ISC = 0.0005   # 1/K, Isc temperature coefficient
NOCT = 45.0          # degC
# cell heating above ambient (degC per W/m2) and its scatter; NOCT rule = 25/800. Calibrated on the real FAULT-FREE class.
HEAT = {1: 0.010, 2: (NOCT - 20) / 800.0}   # S1 calibrated on the real fault-free class; small 10 W test modules run cooler than NOCT
TC_SD = {1: 2.5, 2: 4.0}
V_BYPASS = 0.5       # V, bypass diode forward drop
V_BREAKDOWN = 15.0   # V, cell reverse-breakdown clamp


# ------------------------------------------------------------ cell-level model
def _cell_params(spec, Tc):
    """Per-cell single-diode parameters at cell temperature Tc (degC)."""
    Tk, Tref = Tc + 273.15, 298.15
    n, Ns = spec["n_ideal"], spec["n_cells"]
    Vt = K_B * Tk / Q
    Vt_ref = K_B * Tref / Q
    voc_cell = spec["Voc"] / Ns
    I0_ref = spec["Isc"] / (np.exp(voc_cell / (n * Vt_ref)) - 1.0)
    I0 = I0_ref * (Tk / Tref) ** 3 * np.exp(EG / (n * K_B / Q) * (1 / Tref - 1 / Tk))
    return dict(n=n, Vt=Vt, I0=I0, Rs=spec["Rs_mod"] / Ns, Rsh=spec["Rsh_mod"] / Ns)


def _cell_voltage(I, Iph, p):
    """Voltage of one cell carrying current(s) I (vectorised bisection)."""
    nVt = p["n"] * p["Vt"]
    lo = np.full_like(I, -60.0)
    hi = np.full_like(I, 1.2)
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        f = Iph - p["I0"] * (np.exp(np.clip(mid / nVt, -50, 60)) - 1) - mid / p["Rsh"] - I
        pos = f > 0
        lo = np.where(pos, mid, lo)
        hi = np.where(pos, hi, mid)
    V = 0.5 * (lo + hi) - I * p["Rs"]
    return np.maximum(V, -V_BREAKDOWN)


def module_vi(spec, cell_G, Tc, I_grid):
    """Module voltage over a current grid. cell_G = effective irradiance per cell (W/m2)."""
    p = _cell_params(spec, Tc)
    Iph_cells = spec["Isc"] * cell_G / 1000.0 * (1 + ALPHA_ISC * (Tc - 25.0))
    Ns, nb = spec["n_cells"], spec["n_bypass"]
    groups = np.array_split(np.arange(Ns), max(nb, 1))
    V_mod = np.zeros_like(I_grid)
    for g in groups:
        V_sub = np.zeros_like(I_grid)
        # identical cells share a solution -> group by rounded photocurrent
        vals, counts = np.unique(np.round(Iph_cells[g], 3), return_counts=True)
        for Iph, c in zip(vals, counts):
            V_sub += c * _cell_voltage(I_grid, Iph, p)
        if nb > 0:
            V_sub = np.maximum(V_sub, -V_BYPASS)
        V_mod += V_sub
    return V_mod


def string_vi(specs, cell_G_list, Tc, I_grid):
    return sum(module_vi(s, g, Tc, I_grid) for s, g in zip(specs, cell_G_list))


def voc_isc_series(V, I_grid):
    voc = V[0]
    idx = np.where(V <= 0)[0]
    if len(idx) == 0:
        return voc, I_grid[-1]
    k = idx[0]
    isc = np.interp(0.0, [V[k], V[k - 1]], [I_grid[k], I_grid[k - 1]])
    return voc, isc


def voc_isc_parallel(V_list, I_grid):
    """Parallel modules: sum currents at common voltage."""
    V_axis = np.linspace(0, max(v[0] for v in V_list), 3000)
    I_tot = np.zeros_like(V_axis)
    for V in V_list:
        order = np.argsort(V)
        I_tot += np.interp(V_axis, V[order], I_grid[order], left=I_grid.max(), right=0.0)
    isc = I_tot[0]
    voc = np.interp(0.0, I_tot[::-1], V_axis[::-1])
    return voc, isc


def mpp(V, I_grid, rng, p_local=0.15):
    """Operating point chosen by the inverter MPPT. With multiple power peaks
    (typical under partial shading with bypass diodes) a real tracker can lock
    onto a local peak; we model that with probability p_local."""
    P = V * I_grid
    ok = V > 0
    P = np.where(ok, P, 0)
    peaks = [i for i in range(1, len(P) - 1) if P[i] >= P[i - 1] and P[i] > P[i + 1] and P[i] > 0.05 * P.max()]
    k = int(np.argmax(P))
    if len(peaks) > 1 and rng.random() < p_local:
        k = peaks[int(np.argmin(I_grid[peaks]))]     # highest-voltage peak
    return V[k], I_grid[k]


# ------------------------------------------------------------- fault injection
# Fault-model parameter ranges. Chosen ONLY by comparing simulated vs real class
# distributions (calibrate.py). Never chosen by looking at any model's accuracy.
P1 = dict(dirt_tau=(0.809, 1.000), shade_cells=(1, 8), shade_tau=(0.423, 0.579))   # calibrated (calibrate.py / cal_sep.py)
P2 = dict(shade_sub=(6, 10), shade_tau=(0.400, 0.570), dirt_uniform=(0.951, 1.000), dirt_frac=(0.000, 0.022), dirt_patch_tau=(0.577, 0.955), dirt_heavy_sub=(0, 1), dirt_heavy_tau=(0.131, 0.408))   # calibrated (dirt re-calibrated with heavy blotches)

def make_fault_maps(specs, G, fault, rng, system):
    """Per-cell effective irradiance for every module."""
    maps = [np.full(s["n_cells"], G, dtype=float) for s in specs]
    if system == 1:
        if fault == 2:   # small test modules, dust over the whole glass (paper Fig. 4a)
            tau_mean = rng.uniform(*P1["dirt_tau"])
            for m in maps:
                tau_mod = np.clip(tau_mean + rng.normal(0, 0.02), 0.5, 1.0)
                m *= np.clip(tau_mod + rng.normal(0, 0.015, m.size), 0.4, 1.0)
        elif fault == 1:  # object/hand shadow over a few cells of one module (paper Fig. 4b)
            mi = rng.integers(len(maps))
            k = rng.integers(P1["shade_cells"][0], P1["shade_cells"][1] + 1)
            cells = rng.choice(maps[mi].size, k, replace=False)
            maps[mi][cells] *= rng.uniform(*P1["shade_tau"], k)
    else:
        subs = [(mi, sub) for mi, s in enumerate(specs)
                for sub in np.array_split(np.arange(s["n_cells"]), max(s["n_bypass"], 1))]
        if fault == 1:
            # objects (chairs, paper Fig. 7b) cast one shadow over a run of adjacent
            # substrings; transmittance of a shadow = diffuse fraction of light
            n = rng.integers(P2["shade_sub"][0], P2["shade_sub"][1] + 1)
            first = rng.integers(0, len(subs) - n + 1)
            tau = rng.uniform(*P2["shade_tau"])
            for mi, sub in subs[first:first + n]:
                cover = sub[: rng.integers(len(sub) // 3, len(sub) + 1)]
                maps[mi][cover] *= np.clip(tau + rng.normal(0, 0.03, cover.size), 0.1, 1.0)
        elif fault == 2:
            # splashed, blotchy dirt (paper Fig. 7c): light film everywhere +
            # random opaque-ish patches on a fraction of the cells
            for m in maps:
                m *= rng.uniform(*P2["dirt_uniform"])
                frac = rng.uniform(*P2["dirt_frac"])
                cells = rng.choice(m.size, max(1, int(frac * m.size)), replace=False)
                m[cells] *= rng.uniform(*P2["dirt_patch_tau"], cells.size)
            # a few heavy blotches, each covering part of one substring: strong enough to make that
            # substring's bypass diode conduct, which lowers the operating voltage (seen in real dirt data)
            n_heavy = rng.integers(P2["dirt_heavy_sub"][0], P2["dirt_heavy_sub"][1] + 1)
            for j in rng.choice(len(subs), n_heavy, replace=False):
                mi, sub = subs[j]
                cover = rng.choice(sub, max(1, int(rng.uniform(0.2, 0.6) * sub.size)), replace=False)
                maps[mi][cover] *= rng.uniform(*P2["dirt_heavy_tau"])
    for m in maps:            # coarse irradiance steps -> far fewer distinct cell solutions
        np.round(m / 5.0, out=m); m *= 5.0
    return maps


# --------------------------------------------------------------- sensor models
def quantize(x, step):
    return np.round(x / step) * step


ENV_RANGES = {}   # filled by set_env_ranges(): per-system (G_min, G_max, T_min, T_max)

def set_env_ranges(system, real_df):
    """Operating conditions of the paper's measurement campaign = range of G and Ta in its data."""
    ENV_RANGES[system] = (real_df["G/1000"].min() * 1000, real_df["G/1000"].max() * 1000,
                          real_df["AT/50"].min() * 50, real_df["AT/50"].max() * 50)

def sample_env_ranges(system, rng):
    g0, g1, t0, t1 = ENV_RANGES[system]
    return rng.uniform(g0, g1), rng.uniform(t0, t1)

def sample_env(real_df, fault, rng, g_jitter=0.02, t_jitter=0.01):
    """Draw (G, Ta) from the real measurements of that class, with jitter
    (bootstrap-style: simulate at the conditions actually observed)."""
    sub = real_df[real_df["Fault"] == fault]
    row = sub.sample(1, random_state=int(rng.integers(1e9))).iloc[0]
    G = max(150.0, row["G/1000"] * 1000 * (1 + rng.normal(0, g_jitter)))
    Ta = row["AT/50"] * 50 * (1 + rng.normal(0, t_jitter))
    return G, Ta


# ------------------------------------------------------------ system simulators
IV_FEATURES = ["Vmp/MaxVoc", "Imp/MaxIsc", "FF"]   # extra I-V curve features (need an I-V tracer in practice)

# System 1 sensor profiles.
#  paper: DSN VC288 meter (0.1 V / 0.01 A resolution, +/-0.1 % V, +/-1 % I), PYR20 pyranometer (+/-1 %), single reading
#  hires: INA219-class sensor (0.01 V / 1 mA resolution, +/-0.5 % I), 8 readings averaged; pyranometer random noise
#         averaged down to 0.5 % (a fixed calibration offset is the same for every class, so it cannot help classify)
SENSORS_S1 = {"paper": dict(v_res=0.1, v_noise=0.001, i_res=0.01, i_noise=0.01, g_noise=0.01, n_avg=1),
              "hires": dict(v_res=0.01, v_noise=0.001, i_res=0.001, i_noise=0.005, g_noise=0.005, n_avg=8)}
SENSOR_S1 = "paper"

def parallel_curve(V_list, I_grid):
    """Parallel modules: total current at a common voltage."""
    V_axis = np.linspace(0, max(v[0] for v in V_list), 1500)
    I_tot = np.zeros_like(V_axis)
    for V in V_list:
        order = np.argsort(V)
        I_tot += np.interp(V_axis, V[order], I_grid[order], left=I_grid.max(), right=0.0)
    return V_axis, I_tot


def simulate_system1_sample(fault, G, Ta, config, rng):
    specs = [SA10, SA10]
    Tc = Ta + G * HEAT[1] + rng.normal(0, TC_SD[1])   # cell temperature
    maps = make_fault_maps(specs, G, fault, rng, system=1)
    I_grid = np.linspace(0, 1.3 * SA10["Isc"], 600)
    V_list = [module_vi(s, m, Tc, I_grid) for s, m in zip(specs, maps)]
    if config == "series":
        V = V_list[0] + V_list[1]
        voc, isc = voc_isc_series(V, I_grid)
        vmp, imp = mpp(V, I_grid, rng, p_local=0.0)            # I-V tracer finds the global peak
        max_voc, max_isc = 2 * SA10["Voc_nameplate"], SA10["Isc"]
    else:
        V_axis, I_tot = parallel_curve(V_list, I_grid)
        voc, isc = voc_isc_parallel(V_list, I_grid)
        vmp, imp = mpp(V_axis, I_tot, rng, p_local=0.0)
        max_voc, max_isc = SA10["Voc_nameplate"], 2 * SA10["Isc"]
    sp = SENSORS_S1[SENSOR_S1]
    sv, si = sp["v_noise"] / np.sqrt(sp["n_avg"]), sp["i_noise"] / np.sqrt(sp["n_avg"])
    voc = quantize(voc * (1 + rng.normal(0, sv)), sp["v_res"])
    isc = quantize(isc * (1 + rng.normal(0, si)), sp["i_res"])
    vmp = quantize(vmp * (1 + rng.normal(0, sv)), sp["v_res"])
    imp = quantize(imp * (1 + rng.normal(0, si)), sp["i_res"])
    ff = vmp * imp / (voc * isc) if voc > 0 and isc > 0 else 0.0
    G_meas = quantize(G * (1 + rng.normal(0, sp["g_noise"])), 1.0)       # pyranometer, 1 W/m2
    Ta_meas = quantize(Ta + rng.normal(0, 0.2), 0.0625)          # DS18B20 12-bit
    return [voc / max_voc, isc / max_isc, G_meas / 1000, Ta_meas / 50, vmp / max_voc, imp / max_isc, ff, fault]


def _fill_factor_ratios(specs):
    kv = np.mean([s["Voc"] / s["Vmp"] for s in specs])
    ki = np.mean([s["Isc"] / s["Imp"] for s in specs])
    return kv, ki


IV_TRACER_S2 = ["Voc_t/MaxVoc", "Isc_t/MaxIsc"]   # System 2 Voc/Isc measured directly by an I-V tracer

def simulate_system2_sample(fault, G, Ta, rng):
    """Returns the paper-style features (inverter MPP converted to Voc/Isc with the fill-factor
    ratios, paper Sec. V-B) and I-V-tracer features (true Voc, Isc, Vmp, Imp, FF of the string)."""
    specs = SYSTEM2_STRING
    Tc = Ta + G * HEAT[2] + rng.normal(0, TC_SD[2])
    maps = make_fault_maps(specs, G, fault, rng, system=2)
    I_grid = np.linspace(0, 1.25 * max(s["Isc"] for s in specs), 700)
    V = string_vi(specs, maps, Tc, I_grid)
    max_voc = sum(s["Voc"] for s in specs)
    max_isc = min(s["Isc"] for s in specs)     # series string is limited by weakest module
    # --- paper route: inverter MPPT point -> Voc, Isc
    vmp_inv, imp_inv = mpp(V, I_grid, rng)
    kv, ki = _fill_factor_ratios(specs)
    voc_est = vmp_inv * kv * (1 + rng.normal(0, 0.005))
    isc_est = imp_inv * ki * (1 + rng.normal(0, 0.01))
    # --- I-V tracer route: whole curve measured, global power peak
    voc_t, isc_t = voc_isc_series(V, I_grid)
    vmp_t, imp_t = mpp(V, I_grid, rng, p_local=0.0)
    voc_t, isc_t = voc_t * (1 + rng.normal(0, 0.005)), isc_t * (1 + rng.normal(0, 0.01))
    vmp_t, imp_t = vmp_t * (1 + rng.normal(0, 0.005)), imp_t * (1 + rng.normal(0, 0.01))
    ff_t = vmp_t * imp_t / (voc_t * isc_t)
    G_meas = quantize(G * (1 + rng.normal(0, 0.01)), 1.0)
    Ta_meas = quantize(Ta + rng.normal(0, 0.2), 0.0625)
    return [voc_est / max_voc, isc_est / max_isc, G_meas / 1000, Ta_meas / 50,
            vmp_t / max_voc, imp_t / max_isc, ff_t, voc_t / max_voc, isc_t / max_isc, fault]


def generate(real1, real2, n_sim1=200, n_sim2=140, seed=2026, env="bootstrap"):
    """env="bootstrap": conditions drawn from the real rows of the same class (Dataset 1).
       env="ranges"   : conditions drawn uniformly over the paper's measured G/Ta range (Dataset 2)."""
    rng = np.random.default_rng(seed)
    set_env_ranges(1, real1); set_env_ranges(2, real2)
    rows1 = []
    for i in range(n_sim1):
        fault = i % 3
        config = "series" if (i // 3) % 2 == 0 else "parallel"
        G, Ta = sample_env(real1, fault, rng) if env == "bootstrap" else sample_env_ranges(1, rng)
        rows1.append(simulate_system1_sample(fault, G, Ta, config, rng) + [config])
    rows2 = []
    for i in range(n_sim2):
        fault = i % 3
        G, Ta = sample_env(real2, fault, rng) if env == "bootstrap" else sample_env_ranges(2, rng)
        rows2.append(simulate_system2_sample(fault, G, Ta, rng))
    sim1 = pd.DataFrame(rows1, columns=FEATURES + IV_FEATURES + ["Fault", "Config"])
    sim2 = pd.DataFrame(rows2, columns=FEATURES + IV_FEATURES + IV_TRACER_S2 + ["Fault"])
    return sim1, sim2


if __name__ == "__main__":
    pass
