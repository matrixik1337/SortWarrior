import xml.etree.ElementTree as ET
import os
import shutil
import glob
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

def extract_classes_from_names(xml_dir):
    """Извлекает уникальные классы из XML и создает ID на основе порядка появления"""
    classes_dict = {}  # class_name -> class_id
    xml_files = glob.glob(os.path.join(xml_dir, '*.xml'))
    
    if not xml_files:
        print(f"Warning: No XML files found in {xml_dir}")
        return {}
    
    print(f"Found {len(xml_files)} XML files")
    
    # Проходим по всем файлам для сбора уникальных имен классов
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                class_name_elem = obj.find('name')
                if class_name_elem is None:
                    continue
                    
                class_name = class_name_elem.text.strip()
                
                # Добавляем класс в словарь, если его еще нет
                if class_name not in classes_dict:
                    classes_dict[class_name] = len(classes_dict)
                        
        except Exception as e:
            print(f"Error parsing {xml_file}: {e}")
    
    # Создаем отсортированный список классов
    class_list = list(classes_dict.keys())
    
    if class_list:
        print(f"\nFound {len(class_list)} unique classes:")
        for i, class_name in enumerate(class_list):
            print(f"  {i}: {class_name}")
    else:
        print("Warning: No classes found in XML files!")
    
    return classes_dict, class_list

def convert_voc_to_yolo_by_name(xml_file, output_dir, classes_dict):
    """Конвертирует XML в YOLO формат с использованием имени класса"""
    
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        size = root.find('size')
        if size is None:
            print(f"Warning: No size info in {xml_file}, skipping")
            return False
        
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)
        
        if img_width <= 0 or img_height <= 0:
            print(f"Warning: Invalid image size in {xml_file}, skipping")
            return False
        
        base_name = os.path.splitext(os.path.basename(xml_file))[0]
        txt_filename = base_name + '.txt'
        txt_path = os.path.join(output_dir, txt_filename)
        
        with open(txt_path, 'w') as f:
            objects_written = 0
            
            for obj in root.findall('object'):
                class_name_elem = obj.find('name')
                if class_name_elem is None:
                    continue
                    
                class_name = class_name_elem.text.strip()
                
                # Получаем ID класса из словаря
                if class_name in classes_dict:
                    class_id = classes_dict[class_name]
                else:
                    print(f"Error: Unknown class '{class_name}' in {xml_file}")
                    continue
                
                bbox = obj.find('bndbox')
                if bbox is None:
                    continue
                
                try:
                    xmin = float(bbox.find('xmin').text)
                    ymin = float(bbox.find('ymin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymax = float(bbox.find('ymax').text)
                except (ValueError, AttributeError) as e:
                    print(f"Warning: Invalid bbox coordinates in {xml_file}, skipping object")
                    continue
                
                if xmax <= xmin or ymax <= ymin:
                    print(f"Warning: Invalid bbox in {xml_file}, skipping")
                    continue
                
                # Конвертируем в YOLO формат
                x_center = (xmin + xmax) / (2 * img_width)
                y_center = (ymin + ymax) / (2 * img_height)
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height
                
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                        0 <= width <= 1 and 0 <= height <= 1):
                    print(f"Warning: Bbox out of bounds in {xml_file}, skipping")
                    continue
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                objects_written += 1
        
        if objects_written == 0:
            print(f"Warning: No valid objects in {xml_file}")
            open(txt_path, 'w').close()
        
        return True
        
    except Exception as e:
        print(f"Error converting {xml_file}: {e}")
        return False

def create_dataset_structure_by_name(images_dir, xml_dir, model_name):
    """Создает структуру датасета на основе имен классов"""
    
    # Извлекаем классы из XML
    classes_dict, class_list = extract_classes_from_names(xml_dir)
    if not class_list:
        print("No classes found in XML files!")
        return None
    
    # Создаем временные директории
    temp_dir = f"temp_{model_name}"
    temp_labels = os.path.join(temp_dir, "labels")
    temp_images = os.path.join(temp_dir, "images")
    
    os.makedirs(temp_labels, exist_ok=True)
    os.makedirs(temp_images, exist_ok=True)
    
    # Конвертируем XML в YOLO формат
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    converted = 0
    
    print(f"\nConverting {len(xml_files)} XML files...")
    for xml_file in xml_files:
        xml_path = os.path.join(xml_dir, xml_file)
        if convert_voc_to_yolo_by_name(xml_path, temp_labels, classes_dict):
            converted += 1
    
    print(f"Converted {converted}/{len(xml_files)} XML files")
    
    # Копируем соответствующие изображения
    copied_images = 0
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    for label_file in os.listdir(temp_labels):
        base_name = os.path.splitext(label_file)[0]
        
        for ext in image_extensions:
            image_file = base_name + ext
            image_path = os.path.join(images_dir, image_file)
            
            if os.path.exists(image_path):
                shutil.copy(image_path, os.path.join(temp_images, image_file))
                copied_images += 1
                break
    
    print(f"Copied {copied_images} images")
    
    if converted == 0 or copied_images == 0:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return None
    
    return temp_dir, class_list, classes_dict

def split_and_prepare_dataset(temp_dir, class_list, model_name):
    """Разделяет данные и создает структуру для YOLO"""
    
    temp_images = os.path.join(temp_dir, "images")
    temp_labels = os.path.join(temp_dir, "labels")
    
    # Получаем список изображений
    image_files = [f for f in os.listdir(temp_images) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'))]
    
    if not image_files:
        print("No images found!")
        return None
    
    # Разделяем данные
    train_val, test = train_test_split(image_files, test_size=0.1, random_state=42)
    train, val = train_test_split(train_val, test_size=0.22, random_state=42)  # 0.2/0.9 ≈ 0.22
    
    # Создаем финальную структуру датасета
    dataset_path = os.path.join("datasets", model_name)
    
    def copy_files(file_list, split_name):
        img_dest = os.path.join(dataset_path, 'images', split_name)
        label_dest = os.path.join(dataset_path, 'labels', split_name)
        
        os.makedirs(img_dest, exist_ok=True)
        os.makedirs(label_dest, exist_ok=True)
        
        count = 0
        for img_file in file_list:
            # Копируем изображение
            src_img = os.path.join(temp_images, img_file)
            dst_img = os.path.join(img_dest, img_file)
            shutil.copy(src_img, dst_img)
            
            # Копируем аннотацию
            base_name = os.path.splitext(img_file)[0]
            label_file = base_name + '.txt'
            src_label = os.path.join(temp_labels, label_file)
            dst_label = os.path.join(label_dest, label_file)
            
            if os.path.exists(src_label):
                shutil.copy(src_label, dst_label)
            else:
                open(dst_label, 'w').close()
            
            count += 1
        
        return count
    
    # Копируем файлы для каждого сплита
    train_count = copy_files(train, 'train')
    val_count = copy_files(val, 'val')
    test_count = copy_files(test, 'test')
    
    print(f"\nDataset created:")
    print(f"  Train: {train_count} images")
    print(f"  Val: {val_count} images")
    print(f"  Test: {test_count} images")
    print(f"  Total: {train_count + val_count + test_count} images")
    
    # Создаем YAML файл
    yaml_content = f"""path: {os.path.abspath(dataset_path)}
train: images/train
val: images/val
test: images/test

nc: {len(class_list)}

names:
"""
    for i, class_name in enumerate(class_list):
        yaml_content += f"  {i}: '{class_name}'\n"
    
    yaml_file = os.path.join(dataset_path, f"{model_name}_data.yaml")
    with open(yaml_file, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    # Сохраняем список классов
    with open(os.path.join(dataset_path, "classes.txt"), 'w', encoding='utf-8') as f:
        f.write("# Class mapping (ID: Name)\n")
        for i, class_name in enumerate(class_list):
            f.write(f"{i}: {class_name}\n")
    
    print(f"\nClass mapping saved to: {os.path.join(dataset_path, 'classes.txt')}")
    print(f"YAML config: {yaml_file}")
    
    # Анализируем распределение классов
    analyze_class_distribution(dataset_path, class_list)
    
    return dataset_path, yaml_file

def analyze_class_distribution(dataset_path, class_list):
    """Анализирует распределение классов в датасете"""
    
    class_counts = {i: 0 for i in range(len(class_list))}
    
    for split in ['train', 'val', 'test']:
        labels_dir = os.path.join(dataset_path, 'labels', split)
        
        if not os.path.exists(labels_dir):
            continue
            
        for label_file in os.listdir(labels_dir):
            if label_file.endswith('.txt'):
                label_path = os.path.join(labels_dir, label_file)
                
                try:
                    with open(label_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                parts = line.split()
                                if len(parts) >= 5:
                                    class_id = int(parts[0])
                                    if class_id in class_counts:
                                        class_counts[class_id] += 1
                except Exception as e:
                    print(f"Error reading {label_file}: {e}")
    
    print(f"\nClass distribution:")
    print(f"{'ID':<5} {'Class Name':<20} {'Count':<10} {'%':<10}")
    print("-" * 50)
    
    total = sum(class_counts.values())
    for i, class_name in enumerate(class_list):
        count = class_counts[i]
        percentage = (count / total * 100) if total > 0 else 0
        print(f"{i:<5} {class_name:<20} {count:<10} {percentage:.1f}%")
    
    print("-" * 50)
    print(f"{'Total':<26} {total:<10}")

def train_yolo_model(yaml_file, class_list, model_name, model_size='n'):
    """Обучает модель YOLO"""
    
    # Параметры для разных размеров модели
    params = {
        'n': {'imgsz': 320, 'epochs': 100, 'batch': 16},
        's': {'imgsz': 416, 'epochs': 100, 'batch': 8},
        'm': {'imgsz': 512, 'epochs': 100, 'batch': 4},
        'l': {'imgsz': 640, 'epochs': 100, 'batch': 2},
    }
    
    if model_size not in params:
        model_size = 'n'
    
    p = params[model_size]
    
    print(f"\nTraining configuration:")
    print(f"  Model: YOLOv8{model_size.upper()}")
    print(f"  Image size: {p['imgsz']}")
    print(f"  Epochs: {p['epochs']}")
    print(f"  Batch size: {p['batch']}")
    print(f"  Classes: {len(class_list)}")
    
    # Настройка параметров обучения
    use_default = input("\nUse default parameters? (y/n): ").strip().lower()
    if use_default == 'n':
        p['imgsz'] = int(input(f"Enter image size [{p['imgsz']}]: ") or p['imgsz'])
        p['epochs'] = int(input(f"Enter epochs [{p['epochs']}]: ") or p['epochs'])
        p['batch'] = int(input(f"Enter batch size [{p['batch']}]: ") or p['batch'])
    
    print(f"\nStarting training...")
    
    try:
        # Создаем новую модель
        model = YOLO(f'yolov8{model_size}.yaml')
        
        # Обучаем
        results = model.train(
            data=yaml_file,
            imgsz=p['imgsz'],
            epochs=p['epochs'],
            batch=p['batch'],
            name=f"{model_name}_{model_size}",
            pretrained=False,
            patience=30,
            workers=2,
            verbose=True,
            save=True,
            val=True,
            plots=True,
            seed=42
        )
        
        # Экспортируем в ONNX
        weights_path = f"runs/detect/{model_name}_{model_size}/weights/best.pt"
        if os.path.exists(weights_path):
            trained_model = YOLO(weights_path)
            export_path = trained_model.export(format='onnx', simplify=True, opset=12)
            print(f"\nModel exported to: {export_path}")
            
            # Также сохраняем в TorchScript
            try:
                torchscript_path = trained_model.export(format='torchscript')
                print(f"Also exported to TorchScript: {torchscript_path}")
            except:
                pass
        
        print(f"\nTraining completed successfully!")
        print(f"Best model saved to: {weights_path}")
        
        return results
        
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("===== SIMPLE YOLO TRAINING PIPELINE =====")
    print("Converts XML annotations to YOLO format and trains model\n")
    
    # Ввод параметров
    xml_dir = input("Enter XML annotations directory: ").strip()
    img_dir = input("Enter images directory: ").strip()
    
    if not os.path.exists(xml_dir):
        print(f"Error: XML directory not found: {xml_dir}")
        return
    
    if not os.path.exists(img_dir):
        print(f"Error: Images directory not found: {img_dir}")
        return
    
    model_name = input("Enter model name: ").strip()
    
    # Выбор размера модели
    print("\nSelect model size:")
    print("n - Nano (fastest)")
    print("s - Small (balanced)")
    print("m - Medium (better accuracy)")
    print("l - Large (best accuracy)")
    model_size = input("Your choice [n]: ").strip().lower()
    
    if model_size not in ['n', 's', 'm', 'l']:
        model_size = 'n'
    
    print("\n" + "="*60)
    print("Step 1: Extracting classes from XML...")
    
    # Создание датасета
    result = create_dataset_structure_by_name(img_dir, xml_dir, model_name)
    if result is None:
        print("Failed to create dataset!")
        return
    
    temp_dir, class_list, classes_dict = result
    
    print("\n" + "="*60)
    print("Step 2: Creating dataset structure...")
    
    dataset_result = split_and_prepare_dataset(temp_dir, class_list, model_name)
    if dataset_result is None:
        print("Failed to prepare dataset!")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return
    
    dataset_path, yaml_file = dataset_result
    
    print("\n" + "="*60)
    print("Step 3: Ready for training")
    
    confirm = input("\nStart training? (y/n): ").strip().lower()
    if confirm == 'y':
        train_yolo_model(yaml_file, class_list, model_name, model_size)
    else:
        print("Training cancelled.")
        print(f"\nDataset prepared at: {dataset_path}")
        print(f"You can train later using: yolo train data={yaml_file} model=yolov8{model_size}.yaml")
    
    # Очистка временных файлов
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temporary files: {temp_dir}")
    
    print("\n" + "="*60)
    print("Pipeline completed!")
    print(f"Dataset saved at: {dataset_path}")

if __name__ == "__main__":
    main()