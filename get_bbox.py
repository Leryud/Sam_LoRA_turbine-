import PIL
from PIL import Image
import numpy as np
from src.utils import get_bounding_box
import json
from pathlib import Path
import shutil
import os

# Create directory structure
dataset_path = Path("./dataset")
train_path = dataset_path / "train"
test_path = dataset_path / "test"
val_path = dataset_path / "val"

train_path.mkdir(exist_ok=True)
test_path.mkdir(exist_ok=True)
val_path.mkdir(exist_ok=True)

(train_path / "masks").mkdir(exist_ok=True)
(test_path / "masks").mkdir(exist_ok=True)
(val_path / "masks").mkdir(exist_ok=True)

def process_filenames(file_path, split):
    """
    Process filenames from a file and return a JSON struct.

    Args:
        file_path (str): Path to the file containing filenames.
        split (str): Data split (train, test, or val).

    Returns:
        list: A list of JSON structs containing image path, mask path, and bounding box.
    """
    result = []
    with open(file_path, 'r') as f:
        for line in f:
            filename = line.strip()
            image_path = f"./dataset/v1/{filename}"
            mask_path = f"./dataset/v1/masks/{filename.split('.')[0]}_class_turbine.png"

            # Copy image and mask to respective folders
            new_image_path = f"./dataset/{split}/{filename}"
            new_mask_path = f"./dataset/{split}/masks/{filename.split('.')[0]}_class_turbine.png"
            shutil.copy(image_path, new_image_path)
            shutil.copy(mask_path, new_mask_path)

            mask = np.array(Image.open(mask_path))
            bbox = get_bounding_box(mask)
            bbox = [int(i) for i in bbox]
            result.append({
                "image_path": new_image_path,
                "mask_path": new_mask_path,
                "bbox": bbox
            })
    return result

annotations = dict()
annotations["train"] = process_filenames("dataset/train.txt", "train")
annotations["test"] = process_filenames("dataset/test.txt", "test")
annotations["val"] = process_filenames("dataset/val.txt", "val")

with open('annotations.json', 'w', encoding='utf-8') as f:
    json.dump(annotations, f, indent=4)
