from lib.peripheral.moving_unit import Motors
from time import sleep


motors = Motors()
motors.set_speed(0,0)

print("Rotating leviy motor туды")
for i in range(100):
    motors.set_speed(i,0)
    sleep(0.0075)
for i in range(100):
    motors.set_speed(100-i,0)
    sleep(0.0075)

print("Rotating leviy motor сюды")
for i in range(100):
    motors.set_speed(-i,0)
    sleep(0.0075)
for i in range(100):
    motors.set_speed(-(100-i),0)
    sleep(0.0075)

print("Rotating praviy motor туды")
for i in range(100):
    motors.set_speed(0,i)
    sleep(0.0075)
for i in range(100):
    motors.set_speed(0,100-i)
    sleep(0.0075)

print("Rotating praviy motor сюды")
for i in range(100):
    motors.set_speed(0,-i)
    sleep(0.0075)
for i in range(100):
    motors.set_speed(0,-(100-i))
    sleep(0.0075)

motors.set_speed(0,0)

motors.close()