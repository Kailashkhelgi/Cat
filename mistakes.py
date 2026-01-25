"""
mistakes.py

Find and save a set of misclassified validation images (error analysis).
Outputs:
  mistakes/
    cat_as_dog_0.9123__originalfilename.jpg
    dog_as_cat_0.8341__originalfilename.jpg
Also prints counts.

Run:
  python mistakes.py
"""

from pathlib import Path
import shutil

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image


# -----------------------------
# 1) Config
# -----------------------------
CHECKPOINT_PATH = "resnet18_petimages.pth"  # <-- change if your .pth name differs
VAL_DIR = Path("data/val")
OUT_DIR = Path("mistakes")

BATCH_SIZE = 64
MAX_SAVED = 30  # save up to N mistakes to keep folder manageable

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# -----------------------------
# 2) Load model checkpoint
# -----------------------------
def load_model(checkpoint_path: str):
    """
    Load checkpoint, rebuild the model architecture, load weights.
    Returns:
      model, class_names, img_size
    """
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    class_names = ckpt["class_names"]
    img_size = ckpt.get("img_size", 224)

    # Build the same architecture (resnet18) and replace final layer
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(class_names))

    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()

    return model, class_names, img_size


# -----------------------------
# 3) Dataset loader WITH file paths
# -----------------------------
class ImageFolderWithPaths(datasets.ImageFolder):
    """
    torchvision.datasets.ImageFolder but returns (image, label, path)
    so we can copy the original files when we find mistakes.
    """
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, label, path


def main():
    print("Device:", DEVICE)

    model, class_names, img_size = load_model(CHECKPOINT_PATH)

    # Validation transform must match training/validation normalization
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    ds = ImageFolderWithPaths(VAL_DIR, transform=val_transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Create output folder (clean it if it exists)
    OUT_DIR.mkdir(exist_ok=True)

    saved = 0
    total = 0
    wrong = 0

    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            # Loop over batch items to find mistakes
            for i in range(len(paths)):
                total += 1
                y_true = labels[i].item()
                y_pred = preds[i].item()

                if y_true != y_pred:
                    wrong += 1

                    # Confidence of the predicted class
                    conf = probs[i, y_pred].item()

                    true_name = class_names[y_true]
                    pred_name = class_names[y_pred]

                    src_path = Path(paths[i])

                    # Build a readable filename:
                    # e.g. cat_as_dog_0.9123__1234.jpg
                    out_name = f"{true_name.lower()}_as_{pred_name.lower()}_{conf:.4f}__{src_path.name}"
                    out_path = OUT_DIR / out_name

                    # Copy original image file into mistakes folder
                    shutil.copy(src_path, out_path)

                    saved += 1
                    if saved >= MAX_SAVED:
                        break

            if saved >= MAX_SAVED:
                break

    print(f"Total checked: {total}")
    print(f"Wrong predictions: {wrong}")
    print(f"Saved mistakes: {saved}")
    print(f"Mistakes folder: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
