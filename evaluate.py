"""
evaluate.py

Evaluate a trained PyTorch image classifier checkpoint:
- Loads the saved checkpoint (class names + model weights)
- Runs inference on validation dataset
- Prints:
    - accuracy
    - confusion matrix
    - classification report (precision/recall/f1)
- Saves confusion matrix to a text file for your portfolio

Run:
  python evaluate.py
"""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import confusion_matrix, classification_report


# -----------------------------
# 1) Config
# -----------------------------
CHECKPOINT_PATH = "resnet18_petimages.pth"  # change if your file name differs
DATA_DIR = Path("data")
VAL_DIR = DATA_DIR / "val"
BATCH_SIZE = 64

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# 2) Helpers
# -----------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_checkpoint(path: str):
    """Load checkpoint dictionary (weights + metadata)."""
    ckpt = torch.load(path, map_location=DEVICE)
    return ckpt


def build_model(num_classes: int) -> nn.Module:
    """
    Rebuild the same architecture used in training.
    We set weights=None because we will load our trained weights from checkpoint.
    """
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def get_val_loader(img_size: int) -> tuple[DataLoader, list[str]]:
    """Create validation DataLoader and return it with class names."""
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_ds = datasets.ImageFolder(VAL_DIR, transform=val_transform)

    # IMPORTANT: use val_ds.classes as the ground truth class order
    loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return loader, val_ds.classes


# -----------------------------
# 3) Main evaluation
# -----------------------------
def main():
    print("Device:", DEVICE)

    ckpt = load_checkpoint(CHECKPOINT_PATH)
    class_names = ckpt["class_names"]
    img_size = ckpt.get("img_size", 224)

    val_loader, val_folder_classes = get_val_loader(img_size)

    # Sanity check: class order must match between training and ImageFolder now
    if list(val_folder_classes) != list(class_names):
        print("WARNING: Class order mismatch!")
        print("Checkpoint class_names:", class_names)
        print("Validation folder classes:", val_folder_classes)
        print("This can happen if folder names/order differ. Fix by ensuring same folders/classes.")
        return

    model = build_model(num_classes=len(class_names))
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()

    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"\nValidation accuracy: {acc:.4f} ({correct}/{total})")

    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion matrix:")
    print(cm)

    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print("\nClassification report:")
    print(report)

    # Save results to a file (nice for portfolio + reproducibility)
    out_path = Path("eval_results.txt")
    out_path.write_text(
        f"Checkpoint: {CHECKPOINT_PATH}\n"
        f"Validation accuracy: {acc:.4f} ({correct}/{total})\n\n"
        f"Confusion matrix:\n{cm}\n\n"
        f"Classification report:\n{report}\n"
    )
    print(f"\nSaved evaluation to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
