import os
import shutil

# Define source paths for downloaded data within project_directory
object_data_path = 'Object'  # Folder containing object-level data files
scene_data_path = 'Scene'    # Folder containing scene-level data files

# Define target directories in your project structure
sketch_dir = 'data/sketches'
positive_dir = 'data/positives'
negative_dir = 'data/negatives'

# Create directories if they don’t exist
os.makedirs(sketch_dir, exist_ok=True)
os.makedirs(positive_dir, exist_ok=True)
os.makedirs(negative_dir, exist_ok=True)

# Function to organize files in object data
def organize_object_data(data_path):
    sketch_path = os.path.join(data_path, 'Sketch')
    gt_path = os.path.join(data_path, 'GT')
    edge_path = os.path.join(data_path, 'Edge')  # Optional, if you need edge maps

    # Move sketches
    if os.path.exists(sketch_path):
        for file in os.listdir(sketch_path):
            shutil.move(os.path.join(sketch_path, file), os.path.join(sketch_dir, f"object_{file}"))

    # Move ground truth (positive images)
    if os.path.exists(gt_path):
        for file in os.listdir(gt_path):
            shutil.move(os.path.join(gt_path, file), os.path.join(positive_dir, f"object_{file}"))

    # Move edge maps (if needed)
    if os.path.exists(edge_path):
        for file in os.listdir(edge_path):
            shutil.move(os.path.join(edge_path, file), os.path.join(positive_dir, f"edge_{file}"))  # Optional target

# Organize object-level data
organize_object_data(object_data_path)

# Organize scene-level data similarly, adapting to your dataset structure
def organize_scene_data(data_path):
    sketch_path = os.path.join(data_path, 'Sketch')
    gt_path = os.path.join(data_path, 'GT')

    # Move scene sketches
    if os.path.exists(sketch_path):
        for file in os.listdir(sketch_path):
            shutil.move(os.path.join(sketch_path, file), os.path.join(sketch_dir, f"scene_{file}"))

    # Move scene ground truth (positive images)
    if os.path.exists(gt_path):
        for file in os.listdir(gt_path):
            shutil.move(os.path.join(gt_path, file), os.path.join(positive_dir, f"scene_{file}"))

# Organize scene-level data
organize_scene_data(scene_data_path)

# Optionally, add random images as negatives if needed


