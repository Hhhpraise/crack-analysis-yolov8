import os
import random
import shutil
from glob import glob
from sklearn.model_selection import train_test_split

# Configuration
dataset_path = "C:/Users/HP/PycharmProjects/pythonProject/yolo/dataset"
output_path = "C:/Users/HP/PycharmProjects/pythonProject/yolo"

# Class mapping (binary classification - crack vs no crack)
class_mapping = {
    'CD': 1, 'UD': 0,  # Deck
    'CP': 1, 'UP': 0,  # Pavement
    'CW': 1, 'UW': 0   # Wall
}

def create_directory_structure():
    """Create YOLOv8 directory structure for detection"""
    folders = [
        'images/train', 'images/val', 'images/test',
        'labels/train', 'labels/val', 'labels/test'
    ]
    for folder in folders:
        os.makedirs(os.path.join(output_path, folder), exist_ok=True)

def process_dataset():
    """Process dataset for detection task"""
    image_paths = []
    for category in ['D', 'P', 'W']:
        for label in ['C', 'U']:
            subfolder = f"{label}{category[0]}"
            image_paths.extend(glob(os.path.join(dataset_path, category, subfolder, '*.jpg')))

    # Split dataset (70% train, 15% val, 15% test)
    train_val, test = train_test_split(image_paths, test_size=0.15, random_state=42)
    train, val = train_test_split(train_val, test_size=0.1765, random_state=42)

    # Process each split
    for split, paths in [('train', train), ('val', val), ('test', test)]:
        for img_path in paths:
            parts = img_path.split(os.sep)
            category = parts[-3]
            label_folder = parts[-2]
            original_name = os.path.splitext(parts[-1])[0]
            new_name = f"{category}_{label_folder}_{original_name}"

            # Copy image
            shutil.copy(img_path, os.path.join(output_path, 'images', split, f"{new_name}.jpg"))

            # Create YOLO label file (single class per image)
            class_id = class_mapping[label_folder]
            with open(os.path.join(output_path, 'labels', split, f"{new_name}.txt"), 'w') as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0")  # Centered box covering whole image

def create_yaml_file():
    """Create dataset YAML for detection"""
    yaml_content = f"""path: {os.path.abspath(output_path)}
train: images/train
val: images/val
test: images/test

# Class names
nc: 2
names: ['no_crack', 'crack']
"""
    with open(os.path.join(output_path, 'crack_detection.yaml'), 'w') as f:
        f.write(yaml_content)

if __name__ == "__main__":
    print("Creating directory structure...")
    create_directory_structure()

    print("Processing dataset...")
    process_dataset()

    print("Creating YAML file...")
    create_yaml_file()

    print(f"Detection dataset organized in {output_path}")