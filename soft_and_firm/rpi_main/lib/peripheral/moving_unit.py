# Lib for communication witch moving unit
from serial import Serial

class Motors:
    def __init__(self,baudrate=115200):
        self.ser = Serial("/dev/ttyAMA2",baudrate,timeout=1)
    
    def set_speed(self,l_spd,r_spd):
        data = bytearray([l_spd+100, r_spd+100])
        self.ser.write(data)

    def close(self):
        self.ser.close()
