"""
pi_monitor.py -- the Raspberry Pi 4 program of the PV fault monitor.

One monitoring cycle:
  1. read sunlight     : pyranometer -> amplifier -> ADS1115 AIN0 (I2C)
  2. read temperature  : DS18B20 (1-Wire)
  3. trace the I-V curve: raise TRACER_START, request a sweep over UART, receive Voc/Isc/MPP/curve-shape + the curve
  4. compute the 22 model inputs (normalise, physics ratios, measured / expected-healthy from the datasheet model, number of strings)
  5. XGBoost predicts the class; exact TreeSHAP says why
  6. act: LEDs, buzzer, DC contactor (GPIO) and the OLED (I2C)

Runs unchanged on a real Pi (hal_real.py: smbus2, w1-therm files, gpiozero, pyserial) and in the demo
(hal_virtual.py: emulated chips connected to the circuit simulation) -- software-in-the-loop.

Wiring (BCM numbering)
  GPIO2/3 (pin 3/5)   I2C SDA/SCL -> ADS1115 (0x48), SSD1306 OLED (0x3C)
  GPIO4   (pin 7)     1-Wire      -> DS18B20 (4.7 kOhm pull-up to 3.3 V)
  GPIO14/15 (pin 8/10) UART TX/RX  -> I-V tracer board
  GPIO27  (pin 13)    TRACER_START
  GPIO17  (pin 11)    CONTACTOR   -> MOSFET driver -> DC contactor coil (HIGH = closed)
  GPIO5 / 6 / 13      LED green / yellow / red (330 Ohm)
  GPIO19  (pin 35)    BUZZER      -> transistor driver
  GPIO26  (pin 37)    RESET       <- push button to GND (internal pull-up), re-closes the contactor after a trip
"""
import json, os, sys
import numpy as np, pandas as pd
from xgboost import XGBClassifier, DMatrix

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import pv_simulator_v3 as PVMODEL          # used here only as the DATASHEET MODEL (expected healthy values)
from drivers import ADS1115, DS18B20, SSD1306, IVTracer

PINS = dict(TRACER_START=27, CONTACTOR=17, LED_GREEN=5, LED_YELLOW=6, LED_RED=13, BUZZER=19, RESET=26)
PYRANOMETER_V_PER_WM2 = 0.002
CLASSES = PVMODEL.CLASSES
ACTIONS = {
    0: dict(oled=["SYSTEM OK", "NO FAULT"], led="green", buzzer=False, trip=False),
    1: dict(oled=["PARTIAL SHADING", "CHECK FOR OBJECTS", "CASTING SHADOWS"], led="yellow", buzzer=False, trip=False),
    2: dict(oled=["SOILING", "CLEANING REQUIRED"], led="yellow", buzzer=False, trip=False),
    3: dict(oled=["OPEN CIRCUIT", "STRING DISCONNECTED", "CHECK WIRING"], led="red", buzzer=True, trip=False),
    4: dict(oled=["SHORT CIRCUIT", "DC ISOLATED"], led="red", buzzer=True, trip=True),
    5: dict(oled=["HIGH RESISTANCE", "DEGRADATION", "MAINTENANCE DUE"], led="orange", buzzer=False, trip=False),
}


class Monitor:
    def __init__(self, hal, config, model_file):
        """hal: object with .i2c, .onewire, .serial, .pin(n, output=True) ; config: system datasheet/layout dict."""
        self.hal, self.cfg = hal, config
        self.adc = ADS1115(hal.i2c, 0x48, fsr=4.096)
        self.temp = DS18B20(hal.onewire)
        self.oled = SSD1306(hal.i2c, 0x3C)
        self.pins = {k: hal.pin(v, output=(k != "RESET")) for k, v in PINS.items()}
        self.tracer = IVTracer(hal.serial, self.pins["TRACER_START"])
        self.load_model(model_file)
        self.tripped = False
        self.pins["CONTACTOR"].on()                           # normal state: contactor closed, array feeds the inverter
        self.oled.show_lines(["PV MONITOR", "STARTING..."])

    def load_model(self, model_file):
        self.model = XGBClassifier(); self.model.load_model(model_file); self.model_file = model_file

    # --------------------------------------------------------------- one cycle
    def run_cycle(self):
        if self.tripped and self.pins["RESET"].is_active:       # after a trip the contactor stays open until RESET is pressed
            self.tripped = False
        G = self.adc.read_volts(0) / PYRANOMETER_V_PER_WM2
        Ta = self.temp.read_celsius()
        iv = self.tracer.sweep()
        X = self.inputs(iv, G, Ta)
        prob = self.model.predict_proba(X)[0]; cls = int(np.argmax(prob))
        contrib = self.model.get_booster().predict(DMatrix(X), pred_contribs=True)[0][cls][:-1]
        self.act(cls, [f"G {G:.0f} W/M2  T {Ta:.1f}C", f"P {iv['Vmp'] * iv['Imp']:.0f} W  CONF {prob[cls] * 100:.0f}%"])
        return dict(cls=cls, label=CLASSES[cls], probs=prob.tolist(), G=G, Ta=Ta, iv=iv, inputs=X.iloc[0].to_dict(),
                    shap=dict(zip(X.columns, contrib.tolist())), tripped=self.tripped)

    def inputs(self, iv, G, Ta):
        c = self.cfg
        f = {"Voc/MaxVoc": iv["Voc"] / c["max_voc"], "Isc/MaxIsc": iv["Isc"] / c["max_isc"], "G/1000": round(G) / 1000.0,
             "AT/50": Ta / 50.0, "Vmp/MaxVoc": iv["Vmp"] / c["max_voc"], "Imp/MaxIsc": iv["Imp"] / c["max_isc"],
             "Rs_n": iv["Rs_n"], "Rsh_n": iv["Rsh_n"], "Steps": iv["Steps"]}
        f["FF"] = iv["Vmp"] * iv["Imp"] / (iv["Voc"] * iv["Isc"]) if iv["Voc"] > 0 and iv["Isc"] > 0 else 0.0
        f.update(PVMODEL.residual_row(c["system_id"], c["config"], f))       # measured / expected-healthy (datasheet model)
        return PVMODEL.inputs(pd.DataFrame([f]))

    def act(self, cls, info=()):
        a = ACTIONS[cls]; p = self.pins
        if a["trip"]:
            self.tripped = True
        p["CONTACTOR"].off() if self.tripped else p["CONTACTOR"].on()
        p["LED_GREEN"].value = a["led"] == "green"
        p["LED_YELLOW"].value = a["led"] in ("yellow", "orange")
        p["LED_RED"].value = a["led"] in ("red", "orange")
        p["BUZZER"].value = a["buzzer"]
        # OLED (8 text rows): diagnosis on rows 0-2, readings on rows 4-5, contactor state on row 7
        rows = (list(a["oled"]) + [""] * 3)[:3] + [""] + (list(info) + ["", ""])[:2] + ["", "CONTACTOR OPEN" if self.tripped else "CONTACTOR CLOSED"]
        self.oled.show_lines(rows)


def system_config(system_id, config="series"):
    """Datasheet/layout information the monitor needs for one system (what an installer enters once)."""
    mv, mi = PVMODEL.norm_limits(system_id, config)
    return dict(system_id=system_id, config=config, max_voc=mv, max_isc=mi, name=PVMODEL.SYSTEMS[system_id]["name"])


if __name__ == "__main__":            # on the real Raspberry Pi
    import argparse, time
    from hal_real import RealHAL
    ap = argparse.ArgumentParser(); ap.add_argument("--system", type=int, default=3); ap.add_argument("--model", default="../models/xgb_source.json")
    ap.add_argument("--period", type=float, default=60.0); a = ap.parse_args()
    mon = Monitor(RealHAL(), system_config(a.system), a.model)
    while True:
        r = mon.run_cycle(); print(time.strftime("%H:%M:%S"), r["label"], f"{max(r['probs']):.2f}", flush=True); time.sleep(a.period)
