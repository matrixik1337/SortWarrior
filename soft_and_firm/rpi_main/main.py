print("===== LSMC:SortWarrior PuckCollect 2026 =====")

print("Starting:")
from lib.vision.img_process import ImgProcess
improc=ImgProcess()

print("Initing camera... ", end="")
try:
    from lib.vision.camera import Camera
    cam = Camera(3)
    print("Success!")
except:
    print(f"ERROR: {Exception}")
    exit()

print("Initing moving unit...", end="")
try:
    from lib.peripheral.moving import Moving
    move = Moving()
    print("Success!")
except:
    print(f"ERROR: {Exception}")
    exit()

print("Started successfully!")
