from lib.vision.img_process import ImgProcess
from lib.vision.camera import Camera
from lib.peripheral.moving import Moving
from lib.logging.logging import Logging

from time import sleep

log = Logging("LOG.txt")
improc=ImgProcess()

log.log_action("Initing camera...")
try:
    cam = Camera(3)
    log.log_success("Camera inited!")
except Exception as e:
    log.log_failure(f"Camera is not inited with error: {e}")
    exit()

log.log_action("Initing moving unit...")
try:
    move = Moving(0x10)
    log.log_success("Moving unit inited!")
except Exception as e:
    log.log_failure(f"Moving unit is not inited with error: {e}")
    exit()

log.log_success("ROBOT FULLY INITED\n")

delay = 0.005

for i in range(100):
    move.set_motor_speed(i,0)
    sleep(delay)

for i in range(100):
    move.set_motor_speed(100-i,0)
    sleep(delay)


for i in range(100):
    move.set_motor_speed(0,i)
    sleep(delay)

for i in range(100):
    move.set_motor_speed(0,100-i)
    sleep(delay)



for i in range(100):
    move.set_motor_speed(-i,0)
    sleep(delay)

for i in range(100):
    move.set_motor_speed(-(100-i),0)
    sleep(delay)

for i in range(100):
    move.set_motor_speed(0,-i)
    sleep(delay)

for i in range(100):
    move.set_motor_speed(0,-(100-i))
    sleep(delay)