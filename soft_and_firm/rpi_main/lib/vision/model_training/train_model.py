import xml.etree.ElementTree as ET
import os
import shutil
import glob
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

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

def train_yolo_model(yaml_file, class_list, model_name, model_size='n'):
    params = {
        'n': {'imgsz': 320, 'epochs': 500, 'batch': 32},
        's': {'imgsz': 416, 'epochs': 500, 'batch': 16},
        'm': {'imgsz': 512, 'epochs': 500, 'batch': 8},
        'l': {'imgsz': 640, 'epochs': 500, 'batch': 4},
    }
    
    if model_size not in params:
        model_size = 'n'
    
    p = params[model_size]
    
    try:
        with open("arch.yaml","w") as f:
            f.write(f"""
            # Ultralytics YOLO11 object detection model with P3/8 - P5/32 outputs
            # Model docs: https://docs.ultralytics.com/models/yolo11
            # Task docs: https://docs.ultralytics.com/tasks/detect

            # Parameters
            nc: {len(class_list)} # number of classes
            scales: # model compound scaling constants, i.e. 'model=yolo11n.yaml' will call yolo11.yaml with scale 'n'
            # [depth, width, max_channels]
            n: [0.50, 0.25, 1024] # summary: 181 layers, 2624080 parameters, 2624064 gradients, 6.6 GFLOPs
            s: [0.50, 0.50, 1024] # summary: 181 layers, 9458752 parameters, 9458736 gradients, 21.7 GFLOPs
            m: [0.50, 1.00, 512] # summary: 231 layers, 20114688 parameters, 20114672 gradients, 68.5 GFLOPs
            l: [1.00, 1.00, 512] # summary: 357 layers, 25372160 parameters, 25372144 gradients, 87.6 GFLOPs
            x: [1.00, 1.50, 512] # summary: 357 layers, 56966176 parameters, 56966160 gradients, 196.0 GFLOPs

            # YOLO11n backbone
            backbone:
            # [from, repeats, module, args]
            - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
            - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
            - [-1, 2, C3k2, [256, False, 0.25]]
            - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
            - [-1, 2, C3k2, [512, False, 0.25]]
            - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
            - [-1, 2, C3k2, [512, True]]
            - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
            - [-1, 2, C3k2, [1024, True]]
            - [-1, 1, SPPF, [1024, 5]] # 9
            - [-1, 2, C2PSA, [1024]] # 10

            # YOLO11n head
            head:
            - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
            - [[-1, 6], 1, Concat, [1]] # cat backbone P4
            - [-1, 2, C3k2, [512, False]] # 13

            - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
            - [[-1, 4], 1, Concat, [1]] # cat backbone P3
            - [-1, 2, C3k2, [256, False]] # 16 (P3/8-small)

            - [-1, 1, Conv, [256, 3, 2]]
            - [[-1, 13], 1, Concat, [1]] # cat head P4
            - [-1, 2, C3k2, [512, False]] # 19 (P4/16-medium)

            - [-1, 1, Conv, [512, 3, 2]]
            - [[-1, 10], 1, Concat, [1]] # cat head P5
            - [-1, 2, C3k2, [1024, True]] # 22 (P5/32-large)

            - [[16, 19, 22], 1, Detect, [nc]] # Detect(P3, P4, P5)
            """)

        model = YOLO("arch.yaml")
        results = model.train(
            data=yaml_file,
            imgsz=p['imgsz'],
            epochs=p['epochs'],
            batch=p['batch'],
            name=f"{model_name}_{model_size}",
            pretrained=False,
            lr0=0.01,
            lrf=0.1,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=10.0,
            patience=100,
            optimizer='AdamW',
            seed=42,
            deterministic=True,
            workers=8
        )
        
        weights_path = f"runs/detect/{model_name}_{model_size}/weights/best.pt"
        if os.path.exists(weights_path):
            trained_model = YOLO(weights_path)
            trained_model.export(format='onnx', simplify=True, opset=12)
        
        return results
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    xml_dir = input("XML directory: ").strip()
    img_dir = input("Images directory: ").strip()
    
    if not os.path.exists(xml_dir) or not os.path.exists(img_dir):
        return
    
    model_name = input("Model name: ").strip()
    model_size = input("Model size [n]: ").strip().lower()
    
    if model_size not in ['n', 's', 'm', 'l']:
        model_size = 'n'
    
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
        train_yolo_model(yaml_file, class_list, model_name, model_size)
    
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()