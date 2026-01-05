from ultralytics import YOLO
import cv2

path_to_model = input("Enter path to model: ")
path_to_image = input("Enter path to image: ")

frame = cv2.imread(path_to_image)
model = YOLO(path_to_model,task="detect",verbose=False)
model.cpu()
detections = model.track()

result_frame = frame.copy()



results = model.predict(
    source=frame,
    imgsz=736,
    augment=True,
    conf=0.6,
    verbose=False
)

for result in results:
    class_names = result.names
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cls = int(box.cls[0])
        class_name = class_names[cls]

        conf = float(box.conf[0])

        colour = (0,255,0)

        cv2.rectangle(result_frame, (x1, y1), (x2, y2), colour, 2)

        cv2.putText(result_frame, f"{class_name} {conf:.2f}",
                    (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, colour, 2)


cv2.namedWindow("result",cv2.WINDOW_NORMAL)
while cv2.waitKey(100)!=ord("q"):
    cv2.imshow("result",result_frame)