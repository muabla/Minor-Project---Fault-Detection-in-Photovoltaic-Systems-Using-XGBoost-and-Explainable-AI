"""
hal_real.py -- hardware access on a real Raspberry Pi 4 (Raspberry Pi OS).
Needs: pip install smbus2 gpiozero pyserial ; enable I2C, 1-Wire (dtoverlay=w1-gpio,gpiopin=4) and the UART in raspi-config.
Not tested on hardware in this project (no physical Pi); the demo uses hal_virtual.py with the same interface.
"""
from smbus2 import SMBus
from gpiozero import DigitalOutputDevice, Button
import serial


class _OneWire:
    def read(self, path):
        with open(path) as f:
            return f.read()


class RealHAL:
    def __init__(self, i2c_bus=1, uart="/dev/serial0"):
        self.i2c = SMBus(i2c_bus)
        self.onewire = _OneWire()
        self.serial = serial.Serial(uart, 115200, timeout=2)

    def pin(self, bcm, output=True):
        return DigitalOutputDevice(bcm) if output else Button(bcm, pull_up=True)
