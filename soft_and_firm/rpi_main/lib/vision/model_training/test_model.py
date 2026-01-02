import cv2
import random
from ultralytics import YOLO
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Path to YOLO model")
parser.add_argument("-i", "--image", help="Path to test image")
args = parser.parse_args()

model = YOLO(args.model,verbose=False)
frame = cv2.imread(args.image)

results = model.track(frame,True)
for result in results:
        class_names = result.names
        for box in result.boxes:
            if box.conf[0] > 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                class_name = class_names[cls]
                conf = float(box.conf[0])
                colour = (0,255,0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(frame, f"{class_name} {conf:.2f}",
                            (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, colour, 2)
                
cv2.namedWindow("RESULT", cv2.WINDOW_NORMAL)
cv2.resize(frame,(500,500))
while(cv2.waitKey(100)!=ord("q")):
    cv2.imshow("RESULT",frame)