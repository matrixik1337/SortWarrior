import cv2
import numpy as np
import onnxruntime as ort
import time
import os

# Конфигурация
MODEL_PATH = "model.onnx"
IMG_SIZE = 320
CLASSES = ['blue_puck', 'red_puck', 'blue_base', 'red_base']
CONF_THRESH = 0.25
IMAGE_PATH = "test_img.png"

# Проверка файлов
print(f"Model exists: {os.path.exists(MODEL_PATH)}")
print(f"Image exists: {os.path.exists(IMAGE_PATH)}")

def init_model(model_path):
    """Инициализация модели ONNX"""
    global session, input_name
    
    try:
        # Получаем доступные провайдеры
        available_providers = ort.get_available_providers()
        print(f"Available ONNX Runtime providers: {available_providers}")
        
        # Выбираем провайдеры (предпочитаем CPU для надежности)
        providers = ['CPUExecutionProvider']
        
        # Если хотите попробовать CUDA, можно добавить
        if 'CUDAExecutionProvider' in available_providers:
            print("CUDA is available, trying to use it...")
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        # Создаем сессию
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        session = ort.InferenceSession(
            model_path, 
            sess_options=session_options,
            providers=providers
        )
        
        input_name = session.get_inputs()[0].name
        print(f"Model initialized successfully!")
        print(f"Input name: {input_name}")
        print(f"Input shape: {session.get_inputs()[0].shape}")
        
        # Выводим информацию о выходах
        for i, output in enumerate(session.get_outputs()):
            print(f"Output {i}: name={output.name}, shape={output.shape}")
            
    except Exception as e:
        print(f"Error initializing model: {e}")
        # Пробуем загрузить только с CPU
        try:
            print("Trying with CPU only...")
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            input_name = session.get_inputs()[0].name
            print("Model loaded with CPU provider")
        except Exception as e2:
            print(f"Failed to load model: {e2}")
            session = None
            input_name = None

def detect_objects(frame):
    """Обнаружение объектов на кадре"""
    global session, input_name
    
    if session is None:
        print("Error: Model not initialized!")
        return []
    
    h, w = frame.shape[:2]
    print(f"Processing frame: {w}x{h}")
    
    # Препроцессинг
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    
    input_data = img_resized.astype(np.float32) / 255.0
    input_data = np.transpose(input_data, (2, 0, 1))
    input_data = np.expand_dims(input_data, axis=0)
    
    print(f"Input data shape: {input_data.shape}")
    
    # Inference
    try:
        outputs = session.run(None, {input_name: input_data})
        
        # Отладка: выводим информацию о выходах
        print(f"Number of outputs: {len(outputs)}")
        for i, out in enumerate(outputs):
            print(f"Output {i}: shape={out.shape}, dtype={out.dtype}")
            
        # В зависимости от модели, формат может быть разным
        # YOLO обычно возвращает [batch, num_predictions, 5+num_classes]
        # или [num_predictions, 6] где 6 = [x1, y1, x2, y2, conf, class_id]
        
        results = []
        
        # Способ 1: Если выход имеет форму [1, N, 5+classes]
        if len(outputs[0].shape) == 3 and outputs[0].shape[0] == 1:
            predictions = outputs[0][0]  # Берем первый батч
            print(f"Processing predictions with shape: {predictions.shape}")
            
            for pred in predictions:
                conf = pred[4]
                if conf > CONF_THRESH:
                    class_id = np.argmax(pred[5:])
                    if class_id < len(CLASSES):
                        cx, cy, bw, bh = pred[:4]
                        
                        # Конвертируем из относительных координат в абсолютные
                        x1 = int((cx - bw/2) * w)
                        y1 = int((cy - bh/2) * h)
                        x2 = int((cx + bw/2) * w)
                        y2 = int((cy + bh/2) * h)
                        
                        # Проверка границ
                        x1 = max(0, min(x1, w-1))
                        x2 = max(0, min(x2, w-1))
                        y1 = max(0, min(y1, h-1))
                        y2 = max(0, min(y2, h-1))
                        
                        if x2 > x1 and y2 > y1:  # Проверка валидности bbox
                            results.append({
                                "name": CLASSES[class_id],
                                "startx": x1, "starty": y1,
                                "endx": x2, "endy": y2,
                                "conf": float(conf)
                            })
        
        # Способ 2: Если выход имеет форму [N, 6]
        elif len(outputs[0].shape) == 2 and outputs[0].shape[1] >= 6:
            predictions = outputs[0]
            print(f"Processing predictions with shape: {predictions.shape}")
            
            for pred in predictions:
                x1, y1, x2, y2, conf, class_id = pred[:6]
                if conf > CONF_THRESH and int(class_id) < len(CLASSES):
                    # Конвертируем float координаты в int
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Проверка границ
                    x1 = max(0, min(x1, w-1))
                    x2 = max(0, min(x2, w-1))
                    y1 = max(0, min(y1, h-1))
                    y2 = max(0, min(y2, h-1))
                    
                    if x2 > x1 and y2 > y1:
                        results.append({
                            "name": CLASSES[int(class_id)],
                            "startx": x1, "starty": y1,
                            "endx": x2, "endy": y2,
                            "conf": float(conf)
                        })
        
        print(f"Found {len(results)} objects")
        return results
        
    except Exception as e:
        print(f"Inference error: {e}")
        import traceback
        traceback.print_exc()
        return []

# Основной код
if __name__ == "__main__":
    # Инициализация модели
    init_model(MODEL_PATH)
    
    if session is None:
        print("Failed to initialize model. Exiting...")
        exit(1)
    
    # Загрузка изображения
    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        print(f"Error: Cannot load image {IMAGE_PATH}")
        exit(1)
    
    print(f"Image loaded: {frame.shape}")
    
    # Детекция
    result_frame = frame.copy()
    
    start_time = time.time()
    detections = detect_objects(frame)
    end_time = time.time()
    
    processing_time = end_time - start_time
    print(f"\nProcessing time: {processing_time:.3f} seconds")
    print(f"Total objects detected: {len(detections)}")
    
    # Визуализация результатов
    colors = {
        'blue_puck': (255, 0, 0),     # Красный (BGR)
        'red_puck': (0, 0, 255),      # Синий
        'blue_base': (255, 165, 0),   # Оранжевый
        'red_base': (0, 255, 255)     # Желтый
    }
    
    for det in detections:
        print(f"- {det['name']}: confidence={det['conf']:.3f}, "
              f"bbox=[{det['startx']}, {det['starty']}, {det['endx']}, {det['endy']}]")
        
        color = colors.get(det['name'], (0, 255, 0))
        
        # Рисуем прямоугольник
        cv2.rectangle(result_frame,
                     (det["startx"], det["starty"]),
                     (det["endx"], det["endy"]),
                     color, 2)
        
        # Подпись
        label = f"{det['name']} {det['conf']:.2f}"
        cv2.putText(result_frame, label,
                   (det["startx"], max(20, det["starty"] - 5)),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, color, 2)
    
    # Отображение
    cv2.imshow("Detection Results", result_frame)
    
    # Сохраняем результат
    cv2.imwrite("detection_result.jpg", result_frame)
    print("Result saved as 'detection_result.jpg'")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()