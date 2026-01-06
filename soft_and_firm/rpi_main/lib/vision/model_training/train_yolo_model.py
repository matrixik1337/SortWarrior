import xml.etree.ElementTree as ET
import os
import shutil
import glob
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import argparse

def extract_classes_from_names(xml_dir):
    classes_dict = {}
    xml_files = glob.glob(os.path.join(xml_dir, '*.xml'))
    
    if not xml_files:
        return {}
    
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for obj in root.findall('object'):
                class_name_elem = obj.find('name')
                if class_name_elem is None:
                    continue
                class_name = class_name_elem.text.strip()
                if class_name not in classes_dict:
                    classes_dict[class_name] = len(classes_dict)
        except:
            continue
    
    class_list = list(classes_dict.keys())
    return classes_dict, class_list

def convert_voc_to_yolo_by_name(xml_file, output_dir, classes_dict):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        size = root.find('size')
        if size is None:
            return False
        
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)
        
        if img_width <= 0 or img_height <= 0:
            return False
        
        base_name = os.path.splitext(os.path.basename(xml_file))[0]
        txt_path = os.path.join(output_dir, base_name + '.txt')
        
        with open(txt_path, 'w') as f:
            for obj in root.findall('object'):
                class_name_elem = obj.find('name')
                if class_name_elem is None:
                    continue
                
                class_name = class_name_elem.text.strip()
                if class_name not in classes_dict:
                    continue
                
                class_id = classes_dict[class_name]
                bbox = obj.find('bndbox')
                if bbox is None:
                    continue
                
                try:
                    xmin = float(bbox.find('xmin').text)
                    ymin = float(bbox.find('ymin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymax = float(bbox.find('ymax').text)
                except:
                    continue
                
                if xmax <= xmin or ymax <= ymin:
                    continue
                
                x_center = (xmin + xmax) / (2 * img_width)
                y_center = (ymin + ymax) / (2 * img_height)
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height
                
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                    continue
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        return True
    except:
        return False

def create_dataset_structure(images_dir, xml_dir, model_name):
    classes_dict, class_list = extract_classes_from_names(xml_dir)
    if not class_list:
        return None
    
    temp_dir = f"temp_{model_name}"
    temp_labels = os.path.join(temp_dir, "labels")
    temp_images = os.path.join(temp_dir, "images")
    
    os.makedirs(temp_labels, exist_ok=True)
    os.makedirs(temp_images, exist_ok=True)
    
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    converted = 0
    
    for xml_file in xml_files:
        xml_path = os.path.join(xml_dir, xml_file)
        if convert_voc_to_yolo_by_name(xml_path, temp_labels, classes_dict):
            converted += 1
    
    if converted == 0:
        return None
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    for label_file in os.listdir(temp_labels):
        base_name = os.path.splitext(label_file)[0]
        for ext in image_extensions:
            image_file = base_name + ext
            image_path = os.path.join(images_dir, image_file)
            if os.path.exists(image_path):
                shutil.copy(image_path, os.path.join(temp_images, image_file))
                break
    
    return temp_dir, class_list, classes_dict

def split_and_prepare_dataset(temp_dir, class_list, model_name):
    temp_images = os.path.join(temp_dir, "images")
    temp_labels = os.path.join(temp_dir, "labels")
    
    image_files = [f for f in os.listdir(temp_images) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'))]
    
    if not image_files:
        return None
    
    train_val, test = train_test_split(image_files, test_size=0.1, random_state=42)
    train, val = train_test_split(train_val, test_size=0.22, random_state=42)
    
    dataset_path = os.path.join("datasets", model_name)
    
    def copy_files(file_list, split_name):
        img_dest = os.path.join(dataset_path, 'images', split_name)
        label_dest = os.path.join(dataset_path, 'labels', split_name)
        os.makedirs(img_dest, exist_ok=True)
        os.makedirs(label_dest, exist_ok=True)
        
        for img_file in file_list:
            src_img = os.path.join(temp_images, img_file)
            dst_img = os.path.join(img_dest, img_file)
            shutil.copy(src_img, dst_img)
            
            base_name = os.path.splitext(img_file)[0]
            label_file = base_name + '.txt'
            src_label = os.path.join(temp_labels, label_file)
            dst_label = os.path.join(label_dest, label_file)
            
            if os.path.exists(src_label):
                shutil.copy(src_label, dst_label)
            else:
                open(dst_label, 'w').close()
    
    copy_files(train, 'train')
    copy_files(val, 'val')
    copy_files(test, 'test')
    
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
    
    return dataset_path, yaml_file

def train_yolo_model(yaml_file, class_list, model_name, use_pretrained, model_size='n', device_to_use="cuda"):
    params = {
        'n': {'imgsz': 320, 'epochs': 5000, 'batch': 8},
        's': {'imgsz': 416, 'epochs': 4000, 'batch': 4},
        'm': {'imgsz': 512, 'epochs': 3000, 'batch': 2},
        'l': {'imgsz': 640, 'epochs': 2500, 'batch': 1},
    }
    
    p = params[model_size]
        
    try:
        if use_pretrained==True:
            model = YOLO("yolo11n.pt")
        else:
            model = YOLO("yolo11n.yaml")


        results = model.train(
            data=yaml_file,
            imgsz=p['imgsz'],
            epochs=p['epochs'],
            batch=p['batch'],
            name=model_name,
            pretrained=use_pretrained,
            
            # АУГМЕНТАЦИИ
            
            hsv_h=0.0,        # Отключаем изменение оттенка (Hue) - модель не учит цвет
            hsv_s=0.8,        # Сильные изменения насыщенности (0.0-1.0)
            hsv_v=0.6,        # Сильные изменения яркости
            
            degrees=180.0,    # Полный диапазон поворотов (-180 до +180 градусов)
            shear=30.0,       # Сильный наклон/сдвиг
            
            scale=0.9,        # Сильное масштабирование (0.9 = от 10% до 190% размера)

            perspective=0.001,  # Перспективные искажения (0.0-0.001)
            
            fliplr=0.5,       # Горизонтальное отражение 50%
            flipud=0.2,       # Вертикальное отражение 20%
            
            translate=0.2,    # Сдвиг до 20% от размера изображения
            
            mosaic=1.0,       # Всегда использовать мозаику
            mixup=0.3,        # Mixup аугментация 30%
            copy_paste=0.1,   # Копи-паст аугментация
            
            erasing=0.4,      # Random erasing для лучшей обобщающей способности
            
            
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=20.0,
            patience=500,
            
            label_smoothing=0.2,
            dropout=0.2,
            
            seed=42,
            deterministic=True,
            workers=4,
            device=device_to_use,
            
            augment=True,        # Включить все аугментации
            rect=False,
            cache=True,
        )
        return results
    except Exception as e:
        print(f"Error: {e}")
        return None

arg = argparse.ArgumentParser()

arg.add_argument("--annotations","-a",type=str,help="Path to XML annotations directory")
arg.add_argument("--images","-i",type=str,help="Path to images directory")
arg.add_argument("--name","-n",type=str,help="Name of your model")
arg.add_argument("--size","-s",type=str,default="n",help="Size of your model (n/s/m/l)")
arg.add_argument("--device","-d",type=str,default="cuda",help="Device to train on")
arg.add_argument("--pretrained","-p",type=bool,default=False,help="Train pretrained model")


args = arg.parse_args()

def main():
    img_dir = args.images
    xml_dir = args.annotations
    
    if not os.path.exists(xml_dir) or not os.path.exists(img_dir):
        return
    
    model_name = args.name
    model_size = args.size
    
    result = create_dataset_structure(img_dir, xml_dir, model_name)
    if result is None:
        return
    
    temp_dir, class_list, classes_dict = result
    dataset_result = split_and_prepare_dataset(temp_dir, class_list, model_name)
    
    if dataset_result is None:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return
    
    dataset_path, yaml_file = dataset_result
    
    confirm = input("Train? (y/n): ").strip().lower()
    if confirm == 'y':
        train_yolo_model(yaml_file, class_list, model_name, args.pretrained, model_size, args.device)
    
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)

if __name__ == "__main__":
    main()