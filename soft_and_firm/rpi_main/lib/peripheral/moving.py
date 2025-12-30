# Lib for communication witch moving unit
from smbus import SMBus

class Moving:
    def __init__(self,addr=0x10):
        self.i2cbus=SMBus(1)
        self.addr=addr
        

    def set_motor_speed(self,left_speed,right_speed):
        self.i2cbus.write_byte_data(self.addr, left_speed+100, right_speed+100)