import xml.etree.ElementTree as ET
import os
import shutil
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

def convert_voc_to_yolo(xml_file, output_dir, class_names):
    os.makedirs(f"{output_dir}",exist_ok=True)

    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Get image dimensions
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)
    
    txt_filename = os.path.splitext(os.path.basename(xml_file))[0] + '.txt'
    txt_path = os.path.join(output_dir, txt_filename)
    
    with open(txt_path, 'w') as f:
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            class_id = class_names.index(class_name)
            
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            x_center = (xmin + xmax) / (2 * img_width)
            y_center = (ymin + ymax) / (2 * img_height)
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def split_dataset(name, image_dir, label_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    images.sort()
    
    train_val, test = train_test_split(images, test_size=test_ratio, random_state=42)
    train, val = train_test_split(train_val, test_size=val_ratio/(train_ratio+val_ratio), random_state=42)
    
    def copy_files(file_list, split_name):
        os.makedirs(f'datasets/{name}/images/{split_name}', exist_ok=True)
        os.makedirs(f'datasets/{name}/labels/{split_name}', exist_ok=True)
        
        for img_file in file_list:
            shutil.copy(os.path.join(image_dir, img_file), f'datasets/{name}/images/{split_name}/')
            
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_src = os.path.join(label_dir, label_file)
            if os.path.exists(label_src):
                shutil.copy(label_src, f'datasets/{name}/labels/{split_name}/')
            else:
            
                open(f'datasets/{name}/labels/{split_name}/{label_file}', 'w').close()
    
    copy_files(train, 'train')
    copy_files(val, 'val')
    copy_files(test, 'test')
    
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")


model_name = input("Enter model name: ")
img_dir = input("Enter images path: ")

class_names = []
num_of_classes = int(input("Enter num of classes: "))
for i in range(num_of_classes):
    class_names.append(input(f"Enter class name number {i+1}: "))

xml_dir = input("Enter annotations directory: ")
labels_dir = "labels"
for xml_file in os.listdir(xml_dir):
    if xml_file.endswith('.xml'):
        convert_voc_to_yolo(os.path.join(xml_dir, xml_file), labels_dir, class_names)

print("Splitting images...")
split_dataset(model_name,img_dir,labels_dir)

model_arch = f"""
path: {model_name}/
train: 'images/train'
val: 'images/val'
names:
"""

for i in range(len(class_names)):
    model_arch += f"  {i}: '{class_names[i]}'\n"


os.chdir(f"datasets/{model_name}")

print("Creating model architecture...")
with open(f"{model_name}_arch.yaml","w") as file:
    file.write(model_arch)
    file.close()

print("Initialize model...")

model = YOLO("../../yolov8n.pt")

print("TRAINING...")
results = model.train(
   data=f'{model_name}_arch.yaml',
   imgsz=640,
   epochs=50,
   batch=16,
   name=model_name,

)



print("DONE!")