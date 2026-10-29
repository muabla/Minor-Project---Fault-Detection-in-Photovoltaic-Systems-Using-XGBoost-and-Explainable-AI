"""
make_datasets.py  --  creates the five simulated datasets with pv_simulator_v2.py

pv_simulator_v2.py has no command-line entry point (its __main__ block is empty),
so this small helper imports it and calls generate() for you.

Usage (this file, pv_simulator_v2.py, Data_300.csv and Data_60.csv are in the same folder):
    python make_datasets.py                    # 500 rows System 1, 200 rows System 2, seed 2026
    python make_datasets.py --n1 500 --n2 200 --seed 7 --out ./my_new_data

IMPORTANT
* The datasets behind ALL reported results are the ones in  notebooks/data/ .
  You do NOT need to run this file to reproduce any result.
* Run it only if you want NEW simulated data. A new run gives different rows with the same
  statistics, so it will NOT match the presentation files row by row. That is why the default
  output folder is "regenerated_not_presentation_data" - so the final files never get overwritten.

Notes
* env="ranges": sunlight G and temperature T are drawn uniformly over the range the paper
  measured, the SAME way for every class -> weather carries no information about the fault.
  (env="bootstrap" copies each class's real weather and re-creates the paper's weather shortcut.)
* The real CSVs (Data_300.csv, Data_60.csv) are used ONLY to read the min/max of G and T.
"""
import argparse, os
import pandas as pd
import pv_simulator_v2 as P

F4 = ["Voc/MaxVoc", "Isc/MaxIsc", "G/1000", "AT/50"]
IV = ["Vmp/MaxVoc", "Imp/MaxIsc", "FF"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real1", default="Data_300.csv")
    ap.add_argument("--real2", default="Data_60.csv")
    ap.add_argument("--n1", type=int, default=500)
    ap.add_argument("--n2", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="regenerated_not_presentation_data")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    r1, r2 = pd.read_csv(a.real1), pd.read_csv(a.real2)

    # 1) paper's sensor (DSN VC288 model): System 1 + System 2
    P.SENSOR_S1 = "paper"
    s1, s2 = P.generate(r1, r2, n_sim1=a.n1, n_sim2=a.n2, seed=a.seed, env="ranges")
    s1[F4 + ["Fault"]].round(6).to_csv(f"{a.out}/S1_sim{a.n1}_realistic_4features.csv", index=False)
    s1[F4 + IV + ["Fault"]].round(6).to_csv(f"{a.out}/S1_sim{a.n1}_realistic_IV.csv", index=False)
    # System 2, paper route: inverter MPP converted to Voc/Isc with the fill-factor ratios
    s2[F4 + ["Fault"]].round(6).to_csv(f"{a.out}/S2_sim{a.n2}_realistic_4features.csv", index=False)
    # System 2, I-V tracer route: true Voc/Isc of the string + Vmp, Imp, FF
    s2iv = s2.drop(columns=["Voc/MaxVoc", "Isc/MaxIsc"]).rename(
        columns={"Voc_t/MaxVoc": "Voc/MaxVoc", "Isc_t/MaxIsc": "Isc/MaxIsc"})
    s2iv[F4 + IV + ["Fault"]].round(6).to_csv(f"{a.out}/S2_sim{a.n2}_realistic_IV.csv", index=False)

    # 2) high-resolution sensor (INA219-class, 8 readings averaged): System 1 only
    P.SENSOR_S1 = "hires"
    s1h, _ = P.generate(r1, r2, n_sim1=a.n1, n_sim2=0, seed=a.seed, env="ranges")
    s1h[F4 + IV + ["Fault"]].round(6).to_csv(f"{a.out}/S1_sim{a.n1}_realistic_hires_IV.csv", index=False)

    print("Saved 5 files to", os.path.abspath(a.out))
    print(s1h.groupby("Fault").size().rename("System 1 rows per class").to_string())

if __name__ == "__main__":
    main()
