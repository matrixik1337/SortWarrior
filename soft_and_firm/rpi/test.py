from lib.vision.camera import Camera
from lib.vision.img_process import ImgProcess
import cv2

print("Initialize camera... ", end="")
try:
    cam = Camera(3)
    print("Success!")
except Exception as e:
    print(f"ERROR: {e}!")

print("Initialize OpenCV... ", end="")
try:
    improc = ImgProcess()
    print("Success!")
except Exception as e:
    print(f"ERROR: {e}!")

while(1):
    x, y,frame = improc.track_nearest_puck(cam.get_array(),1)
    cv2.imshow("Frame",frame)
    if(cv2.waitKey(1)==ord("q")):
        break

cv2.destroyAllWindows()
cam.stop()