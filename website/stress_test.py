"""
stress_test.py -- demo-style faults of a chosen size (like a person making them in the demo), random weather.
Answers: "if I make this fault myself, does the model recognise it?" Used by the v6 notebook.
"""
import numpy as np, pandas as pd
import pv_simulator_v3 as P

THRESHOLD_NOTE = {   # detection threshold (a noticeable loss) used for the library's training faults
    "partial shading": ">= 2 substrings and >= 5 % of the string's substrings (no-bypass modules: >= 1 cell)",
    "dirt": ">= 3 % loss over the panels", "short circuit": ">= 4 % of the string's substrings (>= 1)",
    "degradation": ">= 7 % extra series resistance (of Vmp/Imp)"}


def _maps(s, G, cfg, fn, rng):
    flat = [m for b in P.layout(s, cfg) for m in b]; maps = [np.full(m["n_cells"], float(G)) for m in flat]
    fn(maps, flat, rng)
    for m in maps:
        np.round(m / 5.0, out=m); m *= 5.0
    return maps


def shade_subs(k, tau=0.5):
    def f(s, G, cfg, rng):
        def cf(maps, flat, rng):
            subs = [(mi, g) for mi, m in enumerate(flat) for g in np.array_split(np.arange(m["n_cells"]), max(m["n_bypass"], 1))]
            st = rng.integers(0, len(subs) - k + 1)
            for mi, g in subs[st:st + k]:
                maps[mi][g] *= tau
        return dict(maps=_maps(s, G, cfg, cf, rng))
    return f


def shade_cells(n, tau=0.5):
    def f(s, G, cfg, rng):
        def cf(maps, flat, rng):
            mi = rng.integers(len(maps)); maps[mi][rng.choice(maps[mi].size, n, replace=False)] *= tau
        return dict(maps=_maps(s, G, cfg, cf, rng))
    return f


def dust(loss):
    return lambda s, G, cfg, rng: dict(maps=_maps(s, G, cfg, lambda maps, flat, rng: [m.__imul__(1 - loss) for m in maps], rng))


def short(n):
    def f(s, G, cfg, rng):
        br = P.layout(s, cfg); b = int(rng.integers(len(br))); mods = br[b]
        if mods[0]["n_bypass"] == 0:
            return dict(short=[(b, int(rng.integers(len(mods))), "all")])
        subs = [(b, m, g) for m in range(len(mods)) for g in range(mods[m]["n_bypass"])]
        return dict(short=[subs[i] for i in rng.choice(len(subs), min(n, len(subs)), replace=False)])
    return f


corrosion = lambda r: (lambda s, G, cfg, rng: dict(r_frac=r, r_branch=int(rng.integers(len(P.layout(s, cfg))))))
cut_wire = lambda s, G, cfg, rng: dict(open_branch=int(rng.integers(len(P.layout(s, cfg)))))
healthy = lambda s, G, cfg, rng: {}

COMMON = [("healthy", 0, healthy), ("dust 2 %", 2, dust(0.02)), ("dust 5 %", 2, dust(0.05)), ("dust 10 %", 2, dust(0.10)),
          ("dust 20 %", 2, dust(0.20)), ("cut wire", 3, cut_wire), ("1 shorted bypass diode", 4, short(1)),
          ("2 shorted bypass diodes", 4, short(2)), ("4 shorted bypass diodes", 4, short(4)), ("8 shorted bypass diodes", 4, short(8)),
          ("corrosion 3 %", 5, corrosion(0.03)), ("corrosion 5 %", 5, corrosion(0.05)), ("corrosion 10 %", 5, corrosion(0.10)),
          ("corrosion 20 %", 5, corrosion(0.20)), ("corrosion 35 %", 5, corrosion(0.35)), ("corrosion 50 %", 5, corrosion(0.50))]
S1_SHADE = [("shadow on 1 cell", 1, shade_cells(1)), ("shadow on 2 cells", 1, shade_cells(2)), ("shadow on 4 cells", 1, shade_cells(4)),
            ("shadow on 8 cells", 1, shade_cells(8)), ("shadow on 16 cells", 1, shade_cells(16))]
BYPASS_SHADE = [("shadow on 1 substring", 1, shade_subs(1)), ("shadow on 2 substrings", 1, shade_subs(2)),
                ("shadow on 3 substrings", 1, shade_subs(3)), ("shadow on 6 substrings", 1, shade_subs(6)),
                ("shadow on 10 substrings", 1, shade_subs(10)), ("light shadow (80 % light), 4 substrings", 1, shade_subs(4, 0.8))]
# note: on System 1 (no bypass diodes) '1 shorted bypass diode' means one whole module shorted


def scenario_inputs(s, fn, n, rng):
    rows = []
    for i in range(n):
        cfg = ("series", "parallel")[i % 2] if s == 1 else "series"
        G, Ta = rng.uniform(*P.ENV_SYS[s]["G"]), rng.uniform(*P.ENV_SYS[s]["Ta"])
        r = P.measure(s, G, Ta, rng, config=cfg, **fn(s, G, cfg, rng))
        rows.append({**r["features"], **P.residual_row(s, cfg, r["features"])})
    return P.inputs(pd.DataFrame(rows))


def run(models, systems=(1, 2, 3, 4), n=30, seed=42):
    """models: {label: {system: fitted model}}. Returns a table: fraction of n readings classified correctly."""
    rng = np.random.default_rng(seed); out = []
    for s in systems:
        for name, truth, fn in COMMON + (S1_SHADE if s == 1 else BYPASS_SHADE):
            X = scenario_inputs(s, fn, n, rng); row = dict(system=s, fault=name, true_class=P.CLASSES[truth])
            for lab, ms in models.items():
                pred = ms[s].predict(X); row[lab] = (pred == truth).mean()
                wrong = pred[pred != truth]
                row[f"{lab}: main mistake"] = P.CLASSES[np.bincount(wrong).argmax()] if len(wrong) else ""
            out.append(row)
    return pd.DataFrame(out)
