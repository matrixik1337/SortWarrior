from ultralytics import YOLO
import cv2
from picamera2 import Picamera2 
import os

class Camera:
    def __init__(self, mode=3):
        os.environ["LIBCAMERA_LOG_LEVELS"] = "4"
        self.cam = Picamera2()
        available_modes = self.cam.sensor_modes[mode]
        config = self.cam.create_preview_configuration(
            main = {
                'format': 'RGB888'
            },
            sensor={
                'output_size': available_modes['size'],
                'bit_depth': available_modes['bit_depth']
            }
        )
        self.cam.configure(config)
        self.cam.start()
    
    def stop(self):
        self.cam.stop()

    def get_array(self):
        return self.cam.capture_array()

path_to_model = input("Enter path to model: ")

cam = Camera(3)

model = YOLO(path_to_model,task="detect",verbose=False)
model.cpu()

cv2.namedWindow("result",cv2.WINDOW_NORMAL)

while cv2.waitKey(1)!=ord("q"):
    frame = cam.get_array()
    detections = model.track()
    result_frame = frame.copy()
    results = model.track(frame, stream=True)

    for result in results:
        class_names = result.names
        for box in result.boxes:
            if box.conf[0] > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cls = int(box.cls[0])
                class_name = class_names[cls]

                conf = float(box.conf[0])

                colour = (0,255,0)

                cv2.rectangle(result_frame, (x1, y1), (x2, y2), colour, 2)

                cv2.putText(result_frame, f"{class_name} {conf:.2f}",
                            (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, colour, 2)
    cv2.imshow("result",result_frame)