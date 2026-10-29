"""
hal_virtual.py -- emulated hardware for software-in-the-loop testing of pi_monitor.py.

The monitor's drivers talk to these objects exactly as they would talk to smbus2 / gpiozero / pyserial / the 1-Wire files.
Behind them is the PV plant: the circuit simulation (pv_simulator_v3) with the fault the user created.
  VirtualI2CBus  : ADS1115 (config/conversion registers, PGA, 16-bit) and SSD1306 (command decoding, 128x64 frame buffer)
  VirtualOneWire : DS18B20 w1_slave text (12-bit, 0.0625 C)
  VirtualSerial  : I-V tracer board (answers 'SWEEP' with the traced curve of the simulated array)
  VirtualPin     : GPIO outputs/inputs with a log of every change
Every bus transaction is logged so the demo can show what the program actually did.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pv_simulator_v3 as P


class Plant:
    """The simulated PV array + sensors. scenario = dict(system, config, G, Ta, fault=kwargs for P.measure)."""
    def __init__(self):
        self.scenario = dict(system=1, config="series", G=800.0, Ta=25.0, fault={}); self.last = None; self.rng = np.random.default_rng()

    def measure(self):
        s = self.scenario
        self.last = P.measure(s["system"], s["G"], s["Ta"], self.rng, config=s["config"], curve=True, **s["fault"])
        return self.last

    def sensors(self):
        if self.last is None:
            self.measure()
        return self.last["raw"]


class VirtualI2CBus:
    def __init__(self, plant, log):
        self.plant, self.log = plant, log
        self.ads_config, self.ads_conv = 0x8583, 0
        self.oled_cmds, self.oled_fb, self.oled_ptr = [], [0] * 1024, 0

    # --- smbus2-compatible API
    def write_i2c_block_data(self, addr, reg, data):
        if addr == 0x48:
            self.log.append(f"I2C W 0x48 reg {reg:#04x} = {data[0]:02X}{data[1]:02X}  (ADS1115 config)")
            cfg = (data[0] << 8) | data[1]; self.ads_config = cfg
            if cfg & 0x8000:                                          # OS bit: start a single conversion
                ch = ((cfg >> 12) & 0b111) - 0b100
                fsr = {0: 6.144, 1: 4.096, 2: 2.048, 3: 1.024, 4: 0.512, 5: 0.256}[(cfg >> 9) & 0b111]
                volts = self._analog(ch)
                self.ads_conv = int(max(-32768, min(32767, round(volts / fsr * 32768)))) & 0xFFFF
        elif addr == 0x3C and reg == 0x40:                            # OLED data bytes
            for b in data:
                self.oled_fb[self.oled_ptr % 1024] = b; self.oled_ptr += 1
        else:
            raise IOError(f"no device at {addr:#04x}")

    def read_i2c_block_data(self, addr, reg, n):
        if addr != 0x48:
            raise IOError(f"no device at {addr:#04x}")
        v = (self.ads_config | 0x8000) if reg == 0x01 else self.ads_conv  # conversion is instant -> OS bit reads 1
        if reg == 0x00:
            self.log.append(f"I2C R 0x48 reg 0x00 -> {v:04X}  (ADS1115 conversion)")
        return [(v >> 8) & 0xFF, v & 0xFF]

    def write_byte_data(self, addr, reg, value):
        if addr != 0x3C:
            raise IOError(f"no device at {addr:#04x}")
        self.oled_cmds.append(value)
        if len(self.oled_cmds) >= 6 and self.oled_cmds[-6] == 0x21 and self.oled_cmds[-3] == 0x22:
            self.oled_ptr = self.oled_cmds[-2] * 128 + self.oled_cmds[-5]   # column/page address -> start of the buffer
            self.log.append("I2C W 0x3C  SSD1306 address window set, frame buffer follows")

    def _analog(self, ch):
        raw = self.plant.sensors()
        if ch == 0:
            return 0.002 * raw["G"]                                   # pyranometer amplifier, 2 mV per W/m2
        return 0.0

    def oled_pixels(self):
        """128x64 pixel rows as strings of 0/1 (for drawing the OLED on the page)."""
        return ["".join("1" if (self.oled_fb[(y // 8) * 128 + x] >> (y % 8)) & 1 else "0" for x in range(128)) for y in range(64)]


class VirtualOneWire:
    def __init__(self, plant, log):
        self.plant, self.log = plant, log

    def read(self, path):
        t = self.plant.sensors()["Ta"]
        self.log.append(f"1-Wire read {path.split('/')[-2]}  t={int(t * 1000)}")
        return f"50 05 4b 46 7f ff 0c 10 1c : crc=1c YES\n50 05 4b 46 7f ff 0c 10 1c t={int(t * 1000)}\n"


class VirtualSerial:
    """The I-V tracer board: on 'SWEEP' it traces the simulated array and returns the result lines."""
    def __init__(self, plant, log):
        self.plant, self.log, self.out = plant, log, []

    def write(self, data):
        if data.strip() == b"SWEEP":
            m = self.plant.last or self.plant.measure()             # the array state of this cycle (sensors read the same moment)
            r, f = m["raw"], m["features"]; V, I = m["curve"]["V"], m["curve"]["I"]
            self.out = [f"IV,{r['Voc']:.4f},{r['Isc']:.5f},{r['Vmp']:.4f},{r['Imp']:.5f},{f['Rs_n']:.6f},{f['Rsh_n']:.6f},{f['Steps']:.0f},{len(V)}\n"]
            self.out += [f"{v:.4f},{i:.5f}\n" for v, i in zip(V, I)]
            self.log.append(f"UART TX 'SWEEP' -> tracer replied Voc={r['Voc']:.2f} V, Isc={r['Isc']:.3f} A, {len(V)} curve points")

    def readline(self):
        return self.out.pop(0).encode() if self.out else b""


class VirtualPin:
    def __init__(self, bcm, output, log, states):
        self.bcm, self.output, self.log, self.states = bcm, output, log, states
        states.setdefault(bcm, 0 if output else 1)                    # inputs idle HIGH (pull-up)

    def on(self): self._set(1)
    def off(self): self._set(0)

    @property
    def value(self): return self.states[self.bcm]

    @value.setter
    def value(self, v): self._set(1 if v else 0)

    @property
    def is_active(self): return self.states[self.bcm] == 0           # button to GND: pressed = LOW

    def _set(self, v):
        if self.states[self.bcm] != v:
            self.log.append(f"GPIO{self.bcm} -> {'HIGH' if v else 'LOW'}")
        self.states[self.bcm] = v


class VirtualHAL:
    def __init__(self):
        self.log, self.states = [], {}
        self.plant = Plant()
        self.i2c, self.onewire, self.serial = VirtualI2CBus(self.plant, self.log), VirtualOneWire(self.plant, self.log), VirtualSerial(self.plant, self.log)

    def pin(self, bcm, output=True):
        return VirtualPin(bcm, output, self.log, self.states)

    def press_reset(self, pressed):
        self.states[26] = 0 if pressed else 1
