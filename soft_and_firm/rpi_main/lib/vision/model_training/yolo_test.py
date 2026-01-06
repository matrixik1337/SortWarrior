from ultralytics import YOLO
import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model","-m",type=str)
parser.add_argument("--image","-i",type=str)
args = parser.parse_args()

img = cv2.imread(args.image)
model = YOLO(args.model,task="detect",verbose=False)

results = model.predict(
    source=img,
    imgsz=736,
    conf=0.6,
    verbose=True,
    stream=False,
)

cv2.namedWindow("result",cv2.WINDOW_NORMAL)
for result in results:
    class_names = result.names
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cls = int(box.cls[0])
        class_name = class_names[cls]

        conf = float(box.conf[0])

        colour = (0,255,0)

        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)

        cv2.putText(img, f"{class_name} {conf:.2f}",
                    (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, colour, 2)


while cv2.waitKey(10)!=ord("q"):
    cv2.imshow("result",img)
