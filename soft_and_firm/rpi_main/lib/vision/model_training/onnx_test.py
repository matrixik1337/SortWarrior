import cv2
import numpy as np
import onnxruntime as ort

class YOLOONNXInference:
    def __init__(self, model_path, conf_thresh=0.5, iou_thresh=0.5):
        self.session = ort.InferenceSession(model_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        
        # Получаем информацию о входе
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.input_height, self.input_width = input_shape[2], input_shape[3]
    
    def preprocess(self, image):
        # Изменение размера и нормализация
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.input_width, self.input_height))
        image = image / 255.0
        image = image.transpose(2, 0, 1)  # HWC to CHW
        image = np.expand_dims(image, axis=0).astype(np.float32)
        return image
    
    def postprocess(self, outputs, orig_shape):
        """Обработка вывода модели (с NMS)"""
        predictions = outputs[0][0]  # [batch, num_detections, 85]
        
        # Фильтрация по confidence
        conf_mask = predictions[:, 4] > self.conf_thresh
        predictions = predictions[conf_mask]
        
        if len(predictions) == 0:
            return []
        
        # Декодирование bounding boxes
        boxes = []
        scores = []
        class_ids = []
        
        for pred in predictions:
            # Извлечение координат
            x, y, w, h = pred[0:4]
            
            # Конвертация в формат [x1, y1, x2, y2]
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2
            
            # Масштабирование к оригинальному размеру
            x1 = int(x1 * orig_shape[1] / self.input_width)
            y1 = int(y1 * orig_shape[0] / self.input_height)
            x2 = int(x2 * orig_shape[1] / self.input_width)
            y2 = int(y2 * orig_shape[0] / self.input_height)
            
            # Получение класса и confidence
            class_id = np.argmax(pred[5:])
            confidence = pred[4] * pred[5 + class_id]
            
            boxes.append([x1, y1, x2, y2])
            scores.append(confidence)
            class_ids.append(class_id)
        
        # NMS
        indices = cv2.dnn.NMSBoxes(
            boxes, scores, self.conf_thresh, self.iou_thresh
        )
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                results.append({
                    'box': boxes[i],
                    'confidence': scores[i],
                    'class_id': class_ids[i]
                })
        
        return results
    
    def infer(self, image):
        orig_shape = image.shape[:2]
        
        # Препроцессинг
        input_tensor = self.preprocess(image)
        
        # Инференс
        outputs = self.session.run(
            None, 
            {self.input_name: input_tensor}
        )
        
        # Постпроцессинг
        detections = self.postprocess(outputs, orig_shape)
        return detections

# Использование
detector = YOLOONNXInference('best.onnx')
image = cv2.imread('test_img.png')
results = detector.infer(image)

# Визуализация результатов
for det in results:
    x1, y1, x2, y2 = det['box']
    confidence = det['confidence']
    class_id = det['class_id']
    
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"Class {class_id}: {confidence:.2f}"
    cv2.putText(image, label, (x1, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imwrite('result.jpg', image)