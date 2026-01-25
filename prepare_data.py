import os
import random
import shutil
from pathlib import Path
from PIL import Image

random.seed(42)

SOURCE_DIR = Path("data/PetImages")
OUTPUT_DIR = Path("data")

TRAIN_DIR = OUTPUT_DIR / "train"
VAL_DIR = OUTPUT_DIR / "val"

SPLIT_RATIO = 0.8  # 80% train, 20% val


def is_valid_image(path):
    try:
        img = Image.open(path)
        img.verify()
        return True
    except:
        return False


def prepare_class(class_name):
    print(f"Processing class: {class_name}")
    src_folder = SOURCE_DIR / class_name

    images = [p for p in src_folder.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    images = [p for p in images if is_valid_image(p)]

    random.shuffle(images)
    split_idx = int(len(images) * SPLIT_RATIO)

    train_images = images[:split_idx]
    val_images = images[split_idx:]

    (TRAIN_DIR / class_name).mkdir(parents=True, exist_ok=True)
    (VAL_DIR / class_name).mkdir(parents=True, exist_ok=True)

    for img_path in train_images:
        shutil.copy(img_path, TRAIN_DIR / class_name / img_path.name)

    for img_path in val_images:
        shutil.copy(img_path, VAL_DIR / class_name / img_path.name)

    print(f"{class_name}: {len(train_images)} train, {len(val_images)} val")


def main():
    for class_name in ["Cat", "Dog"]:
        prepare_class(class_name)

    print("\nDataset ready!")
    print("Train folder:", TRAIN_DIR)
    print("Val folder:", VAL_DIR)


if __name__ == "__main__":
    main()
