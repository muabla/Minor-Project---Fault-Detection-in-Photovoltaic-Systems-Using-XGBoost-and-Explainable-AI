"""
drivers.py -- device drivers used by the Raspberry Pi monitor (pi_monitor.py).

Each driver talks to its chip through a "bus" object. On a real Raspberry Pi the buses are the real ones
(smbus2 for I2C, the Linux 1-Wire files, gpiozero pins, pyserial). In the demo the SAME drivers talk to emulated
chips (hal_virtual.py), so the register-level code below runs unchanged in both cases.

Hardware (see the wiring table in pi_monitor.py):
  ADS1115   16-bit ADC, I2C 0x48   A0 = pyranometer amplifier (2.0 mV per W/m2)
  DS18B20   temperature, 1-Wire    ambient air temperature, 12-bit (0.0625 C)
  SSD1306   128x64 OLED, I2C 0x3C  status text
  I-V tracer board                 capacitor sweep; started by GPIO, results over UART (115200 baud)
  GPIO      contactor driver, LEDs, buzzer, reset button
"""
import time

# ----------------------------------------------------------------------------- ADS1115 (TI datasheet SBAS444)
class ADS1115:
    REG_CONV, REG_CONFIG = 0x00, 0x01
    PGA = {6.144: 0b000, 4.096: 0b001, 2.048: 0b010, 1.024: 0b011, 0.512: 0b100, 0.256: 0b101}
    MUX_SINGLE = {0: 0b100, 1: 0b101, 2: 0b110, 3: 0b111}

    def __init__(self, bus, address=0x48, fsr=4.096):
        self.bus, self.addr, self.fsr = bus, address, fsr

    def read_volts(self, channel):
        """Single-shot conversion on AINx vs GND, 128 SPS. Returns volts."""
        cfg = (1 << 15) | (self.MUX_SINGLE[channel] << 12) | (self.PGA[self.fsr] << 9) | (1 << 8) | (0b100 << 5) | 0b11
        self.bus.write_i2c_block_data(self.addr, self.REG_CONFIG, [(cfg >> 8) & 0xFF, cfg & 0xFF])
        for _ in range(50):                                   # wait for OS bit = 1 (conversion done)
            hi, lo = self.bus.read_i2c_block_data(self.addr, self.REG_CONFIG, 2)
            if hi & 0x80:
                break
            time.sleep(0.002)
        hi, lo = self.bus.read_i2c_block_data(self.addr, self.REG_CONV, 2)
        raw = (hi << 8) | lo
        if raw & 0x8000:
            raw -= 1 << 16
        return raw * self.fsr / 32768.0


# ----------------------------------------------------------------------------- DS18B20 via the Linux w1-therm driver
class DS18B20:
    def __init__(self, onewire, device_id="28-00000a1b2c3d"):
        self.ow, self.dev = onewire, device_id

    def read_celsius(self):
        text = self.ow.read(f"/sys/bus/w1/devices/{self.dev}/w1_slave")
        lines = text.strip().splitlines()
        if not lines[0].endswith("YES"):
            raise IOError("DS18B20 CRC error")
        return int(lines[1].split("t=")[1]) / 1000.0


# ----------------------------------------------------------------------------- SSD1306 128x64 OLED, text mode
FONT5x7 = {  # classic 5x7 font, columns LSB at top; uppercase, digits, a few symbols
    " ": [0, 0, 0, 0, 0], "!": [0, 0, 95, 0, 0], "%": [35, 19, 8, 100, 98], "-": [8, 8, 8, 8, 8], ".": [0, 96, 96, 0, 0],
    "/": [32, 16, 8, 4, 2], ":": [0, 54, 54, 0, 0], "=": [20, 20, 20, 20, 20], "(": [0, 28, 34, 65, 0], ")": [0, 65, 34, 28, 0],
    "0": [62, 81, 73, 69, 62], "1": [0, 66, 127, 64, 0], "2": [66, 97, 81, 73, 70], "3": [33, 65, 69, 75, 49], "4": [24, 20, 18, 127, 16],
    "5": [39, 69, 69, 69, 57], "6": [60, 74, 73, 73, 48], "7": [1, 113, 9, 5, 3], "8": [54, 73, 73, 73, 54], "9": [6, 73, 73, 41, 30],
    "A": [126, 17, 17, 17, 126], "B": [127, 73, 73, 73, 54], "C": [62, 65, 65, 65, 34], "D": [127, 65, 65, 34, 28], "E": [127, 73, 73, 73, 65],
    "F": [127, 9, 9, 9, 1], "G": [62, 65, 73, 73, 122], "H": [127, 8, 8, 8, 127], "I": [0, 65, 127, 65, 0], "J": [32, 64, 65, 63, 1],
    "K": [127, 8, 20, 34, 65], "L": [127, 64, 64, 64, 64], "M": [127, 2, 12, 2, 127], "N": [127, 4, 8, 16, 127], "O": [62, 65, 65, 65, 62],
    "P": [127, 9, 9, 9, 6], "Q": [62, 65, 81, 33, 94], "R": [127, 9, 25, 41, 70], "S": [70, 73, 73, 73, 49], "T": [1, 1, 127, 1, 1],
    "U": [63, 64, 64, 64, 63], "V": [31, 32, 64, 32, 31], "W": [63, 64, 56, 64, 63], "X": [99, 20, 8, 20, 99], "Y": [7, 8, 112, 8, 7],
    "Z": [97, 81, 73, 69, 67],
}


class SSD1306:
    def __init__(self, bus, address=0x3C):
        self.bus, self.addr = bus, address
        for c in (0xAE, 0xD5, 0x80, 0xA8, 0x3F, 0xD3, 0x00, 0x40, 0x8D, 0x14, 0x20, 0x00, 0xA1, 0xC8, 0xDA, 0x12, 0x81, 0xCF, 0xD9, 0xF1, 0xDB, 0x40, 0xA4, 0xA6, 0xAF):
            self.bus.write_byte_data(self.addr, 0x00, c)        # control byte 0x00 = command

    def show_lines(self, lines):
        """Write up to 8 text lines (21 characters each) into the 128x64 frame buffer and send it."""
        fb = [0] * (128 * 8)                                    # 8 pages x 128 columns, 1 byte = 8 vertical pixels
        for row, text in enumerate(lines[:8]):
            for k, ch in enumerate(str(text).upper()[:21]):
                for j, col in enumerate(FONT5x7.get(ch, FONT5x7[" "])):
                    fb[row * 128 + k * 6 + j] = col
        for c in (0x21, 0, 127, 0x22, 0, 7):                    # column and page address range
            self.bus.write_byte_data(self.addr, 0x00, c)
        for i in range(0, len(fb), 16):                          # control byte 0x40 = data
            self.bus.write_i2c_block_data(self.addr, 0x40, fb[i:i + 16])


# ----------------------------------------------------------------------------- I-V tracer board (capacitive sweep)
class IVTracer:
    """Tracer board protocol (UART 115200 8N1). The Pi raises START, sends 'SWEEP', and the board replies
    'IV,<Voc>,<Isc>,<Vmp>,<Imp>,<Rs_n>,<Rsh_n>,<Steps>,<n>' followed by n lines '<V>,<I>' (the traced curve)."""
    def __init__(self, serial, start_pin):
        self.ser, self.start = serial, start_pin

    def sweep(self):
        self.start.on()
        self.ser.write(b"SWEEP\n")
        head = self.ser.readline().decode().strip().split(",")
        if head[0] != "IV":
            self.start.off(); raise IOError("tracer: bad reply " + ",".join(head))
        voc, isc, vmp, imp, rs, rsh, steps = map(float, head[1:8]); n = int(head[8])
        curve = [tuple(map(float, self.ser.readline().decode().split(","))) for _ in range(n)]
        self.start.off()
        return dict(Voc=voc, Isc=isc, Vmp=vmp, Imp=imp, Rs_n=rs, Rsh_n=rsh, Steps=steps, curve=curve)
