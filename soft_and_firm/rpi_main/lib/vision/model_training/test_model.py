import cv2
import numpy as np
import onnxruntime as ort
import time
from typing import List, Tuple, Optional

class YOLOONNXInference:
    def __init__(self, model_path: str, class_names: List[str], 
                 img_size: int = 320, conf_threshold: float = 0.25, 
                 iou_threshold: float = 0.45):
        """
        Инициализация ONNX модели YOLO
        
        Args:
            model_path: путь к файлу .onnx
            class_names: список имен классов
            img_size: размер входного изображения
            conf_threshold: порог уверенности
            iou_threshold: порог для NMS
        """
        self.class_names = class_names
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Инициализация ONNX Runtime
        providers = ['CPUExecutionProvider']  # Для Raspberry Pi используем CPU
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Получаем информацию о входе и выходе модели
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f"Model loaded: {model_path}")
        print(f"Input name: {self.input_name}")
        print(f"Input shape: {self.session.get_inputs()[0].shape}")
        print(f"Output names: {self.output_names}")
        print(f"Number of classes: {len(class_names)}")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Подготовка изображения для модели
        """
        # Изменение размера с сохранением пропорций
        h, w = image.shape[:2]
        scale = min(self.img_size / h, self.img_size / w)
        
        new_h, new_w = int(h * scale), int(w * scale)
        image_resized = cv2.resize(image, (new_w, new_h))
        
        # Добавление паддинга
        pad_h = (self.img_size - new_h) // 2
        pad_w = (self.img_size - new_w) // 2
        
        image_padded = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        image_padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = image_resized
        
        # Нормализация и изменение формата
        image_norm = image_padded.astype(np.float32) / 255.0
        image_norm = image_norm.transpose(2, 0, 1)  # HWC -> CHW
        image_norm = np.expand_dims(image_norm, 0)  # Добавляем batch dimension
        
        return image_norm, (scale, pad_w, pad_h, w, h)
    
    def postprocess(self, outputs: List[np.ndarray], 
                    preprocess_info: Tuple) -> Tuple[List[np.ndarray], List[float], List[int]]:
        """
        Обработка выходов модели
        """
        scale, pad_w, pad_h, orig_w, orig_h = preprocess_info
        
        # YOLOv8 ONNX модель обычно возвращает один выход с формой (1, 84, 8400)
        # где 84 = 4 (bbox) + 80 (классы), но у нас свои классы
        predictions = outputs[0]  # (1, num_classes + 4, num_predictions)
        
        # Транспонируем для удобства: (1, num_predictions, num_classes + 4)
        predictions = predictions.transpose(0, 2, 1)
        
        boxes = []
        scores = []
        class_ids = []
        
        # Обрабатываем предсказания
        for pred in predictions[0]:
            # Последние num_classes значений - вероятности классов
            class_scores = pred[4:]
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]
            
            if confidence > self.conf_threshold:
                # Координаты бокса (x_center, y_center, width, height)
                cx, cy, w, h = pred[:4]
                
                # Конвертируем в формат (x1, y1, x2, y2)
                x1 = (cx - w/2 - pad_w) / scale
                y1 = (cy - h/2 - pad_h) / scale
                x2 = (cx + w/2 - pad_w) / scale
                y2 = (cy + h/2 - pad_h) / scale
                
                # Обрезаем по границам изображения
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(orig_w, int(x2))
                y2 = min(orig_h, int(y2))
                
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    scores.append(float(confidence))
                    class_ids.append(int(class_id))
        
        # Применяем Non-Maximum Suppression
        if len(boxes) > 0:
            boxes = np.array(boxes)
            scores = np.array(scores)
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 
                                      self.conf_threshold, self.iou_threshold)
            
            if len(indices) > 0:
                indices = indices.flatten()
                boxes = boxes[indices]
                scores = scores[indices]
                class_ids = [class_ids[i] for i in indices]
        
        return boxes, scores, class_ids
    
    def draw_detections(self, image: np.ndarray, boxes: List[np.ndarray], 
                       scores: List[float], class_ids: List[int]) -> np.ndarray:
        """
        Отрисовка детекций на изображении
        """
        result = image.copy()
        colors = np.random.randint(0, 255, size=(len(self.class_names), 3), dtype=np.uint8)
        
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box
            
            color = colors[class_id].tolist()
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            label = f"{self.class_names[class_id]}: {score:.2f}"
            
            # Фон для текста
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            cv2.rectangle(result, (x1, y1 - text_height - 10),
                         (x1 + text_width, y1), color, -1)
            
            # Текст
            cv2.putText(result, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result
    
    def inference(self, image: np.ndarray, draw: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Полный пайплайн инференса
        
        Returns:
            image_with_detections: изображение с отрисованными детекциями
            results: словарь с результатами
        """
        start_time = time.time()
        
        # Предобработка
        input_tensor, preprocess_info = self.preprocess(image)
        
        # Инференс
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # Постобработка
        boxes, scores, class_ids = self.postprocess(outputs, preprocess_info)
        
        inference_time = time.time() - start_time
        
        # Отрисовка
        if draw:
            result_image = self.draw_detections(image, boxes, scores, class_ids)
        else:
            result_image = image
        
        # Собираем результаты
        results = {
            'boxes': boxes,
            'scores': scores,
            'class_ids': class_ids,
            'class_names': [self.class_names[i] for i in class_ids],
            'inference_time': inference_time,
            'fps': 1.0 / inference_time if inference_time > 0 else 0
        }
        
        return result_image, results

# Тестовый скрипт для Raspberry Pi
def test_onnx_model():
    # Пути и параметры
    model_path = input("Enter path to your model: ")  # Ваша ONNX модель
    class_names = ["blue_puck", "red_puck", "blue_base", "red_base"]  # Ваши классы
    
    # Инициализация детектора
    detector = YOLOONNXInference(
        model_path=model_path,
        class_names=class_names,
        img_size=320,  # Должно совпадать с размером при обучении
        conf_threshold=0.25,
        iou_threshold=0.45
    )
    
    # Тест 1: Загрузка изображения
    print("\n" + "="*50)
    print("Test 1: Image inference")
    print("="*50)
    
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    result_image, results = detector.inference(test_image)
    
    print(f"Inference time: {results['inference_time']*1000:.2f} ms")
    print(f"FPS: {results['fps']:.1f}")
    print(f"Detected objects: {len(results['boxes'])}")
    
    for box, score, cls_name in zip(results['boxes'], results['scores'], results['class_names']):
        print(f"  - {cls_name}: {score:.2f} at {box}")
    
    # Тест 2: Бенчмарк производительности
    print("\n" + "="*50)
    print("Test 2: Performance benchmark")
    print("="*50)
    
    warmup_iterations = 10
    benchmark_iterations = 100
    
    print(f"Warming up ({warmup_iterations} iterations)...")
    for _ in range(warmup_iterations):
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        _, _ = detector.inference(test_image, draw=False)
    
    print(f"Running benchmark ({benchmark_iterations} iterations)...")
    total_time = 0
    for i in range(benchmark_iterations):
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        _, results = detector.inference(test_image, draw=False)
        total_time += results['inference_time']
        
        if (i + 1) % 20 == 0:
            print(f"  Completed {i + 1}/{benchmark_iterations}")
    
    avg_time = total_time / benchmark_iterations * 1000  # мс
    avg_fps = 1000 / avg_time
    
    print(f"\nBenchmark results:")
    print(f"  Average inference time: {avg_time:.2f} ms")
    print(f"  Average FPS: {avg_fps:.1f}")
    print(f"  Total time for {benchmark_iterations} iterations: {total_time:.2f}s")

if __name__ == "__main__":
    test_onnx_model()