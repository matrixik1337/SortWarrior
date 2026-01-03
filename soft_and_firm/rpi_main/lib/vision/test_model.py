import cv2
import numpy as np
import onnxruntime as ort
import time

# Конфигурация
MODEL_PATH = "model.onnx"
IMG_SIZE = 320
CLASSES = ['blue_puck', 'red_puck', 'blue_base', 'red_base']
CONF_THRESH = 0
IMAGE_PATH = "test_img.png"

session = ort.InferenceSession(MODEL_PATH)

def init_model(model_path):
    global session, input_name
    import onnxruntime as ort
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

def detect_objects(frame):
    if session is None:
        init_model()
    
    h, w = frame.shape[:2]
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    
    input_data = img_resized.astype(np.float32) / 255.0
    input_data = np.transpose(input_data, (2, 0, 1))
    input_data = np.expand_dims(input_data, axis=0)
    
    outputs = session.run(None, {input_name: input_data})
    
    predictions = outputs[0][0]
    results = []
    
    for pred in predictions:
        if pred[4] > CONF_THRESH:
            class_id = np.argmax(pred[5:])
            if class_id < len(CLASSES):
                cx, cy, bw, bh = pred[:4]

                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)
                
                results.append({
                    "name": CLASSES[class_id],
                    "startx": x1, "starty": y1,
                    "endx": x2, "endy": y2,
                    "conf": float(pred[4])
                })
    
    return results

frame = cv2.imread(IMAGE_PATH)
result_frame = frame.copy()
init_model("model.onnx")


start_time = time.time()
detections = detect_objects(frame)
end_time = time.time()

processing_time = end_time-start_time
print(f"Frame processed in {processing_time}s")

for i in detections:
    if i["conf"]>CONF_THRESH:
        cv2.putText(result_frame,str(i["name"])+" "+str(100*i["conf"])+"%",(i["startx"],i["starty"]-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.rectangle(result_frame,(i["startx"],i["starty"]),(i["endx"],i["endy"]),1,1)

while cv2.waitKey(1) != ord("q"):
    cv2.imshow("result",result_frame)

