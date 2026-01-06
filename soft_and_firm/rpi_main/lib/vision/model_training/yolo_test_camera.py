from ultralytics import YOLO
import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model","-m",type=str)

args = parser.parse_args()

cam = cv2.VideoCapture(0)
model = YOLO(args.model,task="detect",verbose=False)

cv2.namedWindow("result",cv2.WINDOW_NORMAL)

while cv2.waitKey(1)!=ord("q"):
    ret, frame = cam.read()
    results = model.predict(
        source=frame,
        imgsz=736,
        conf=0.6,
        verbose=True,
        stream=True,
        device="cpu"
    )


    for result in results:
        class_names = result.names
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cls = int(box.cls[0])
            class_name = class_names[cls]

            conf = float(box.conf[0])

            colour = (0,255,0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

            cv2.putText(frame, f"{class_name} {conf:.2f}",
                        (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, colour, 2)
    cv2.imshow("result",frame)
