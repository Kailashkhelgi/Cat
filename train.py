"""
train.py

Image Classification with Transfer Learning (Cats vs Dogs) using ResNet18 + GPU (if available).

What this script does:
1) Loads images from:
      data/train/Cat, data/train/Dog
      data/val/Cat,   data/val/Dog
2) Uses a pretrained ResNet18 backbone (ImageNet weights).
3) Fine-tunes ONLY:
      - the final classification layer (model.fc)
      - the last ResNet block (model.layer4)
   Everything else stays frozen.
4) Trains for a few epochs and prints validation accuracy each epoch.
5) Saves a checkpoint that includes:
      - model weights
      - class names
      - image size

Run (Windows):
  .venv\\Scripts\\activate
  python train.py
"""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm


# -----------------------------
# 1) Configuration (edit here)
# -----------------------------
DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"

BATCH_SIZE = 32
NUM_EPOCHS = 3          # Fine-tuning usually needs fewer epochs
IMG_SIZE = 224          # ResNet models expect 224x224 images

# Baseline LR was 1e-3 when training only the head.
# Fine-tuning should use a smaller LR to avoid wrecking pretrained features.
FINE_TUNE_LR = 1e-4

MODEL_OUT = "resnet18_petimages_finetuned.pth"

# Use GPU if available, otherwise CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# 2) Image preprocessing
# -----------------------------
# ResNet18 pretrained on ImageNet expects these normalization stats.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Training transforms: include lightweight augmentation to reduce overfitting.
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Validation transforms: deterministic only (no randomness).
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# -----------------------------
# 3) Load datasets + dataloaders
# -----------------------------
# ImageFolder expects: root/class_name/image.jpg
train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_ds = datasets.ImageFolder(VAL_DIR, transform=val_transform)

# num_workers=0 is the safest default on Windows
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

class_names = train_ds.classes
num_classes = len(class_names)

print("Classes detected:", class_names)
print("Train images:", len(train_ds))
print("Val images:", len(val_ds))
print("Device:", DEVICE)
print("Using GPU:", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU")


# -----------------------------
# 4) Build model (Transfer Learning + Fine-tuning)
# -----------------------------
# Load pretrained ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze ALL parameters first (so we start from a stable pretrained backbone)
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the LAST ResNet block ("layer4") for partial fine-tuning.
# This lets the network adapt higher-level features to your dataset.
for param in model.layer4.parameters():
    param.requires_grad = True

# Replace the final classification head (fc layer) for our number of classes.
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)

# By default, new layers have requires_grad=True, but we keep it explicit for clarity.
for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(DEVICE)

# CrossEntropyLoss is standard for multi-class classification.
criterion = nn.CrossEntropyLoss()

# Optimize only the parameters that are unfrozen: fc + layer4.
optimizer = torch.optim.Adam(
    list(model.fc.parameters()) + list(model.layer4.parameters()),
    lr=FINE_TUNE_LR
)


# -----------------------------
# 5) Evaluation function
# -----------------------------
def evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, float]:
    """
    Evaluate model on a dataloader.

    Returns:
      avg_loss (float): average loss over the dataset
      accuracy (float): correct_predictions / total_predictions
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# -----------------------------
# 6) Training loop
# -----------------------------
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}"):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(images)

        # Compute loss
        loss = criterion(logits, labels)

        # Backpropagation
        loss.backward()

        # Update weights (only for unfrozen layers)
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    train_loss = running_loss / len(train_ds)
    val_loss, val_acc = evaluate(model, val_loader)

    print(f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")


# -----------------------------
# 7) Save checkpoint
# -----------------------------
torch.save(
    {
        "model_state": model.state_dict(),
        "class_names": class_names,
        "img_size": IMG_SIZE,
        "arch": "resnet18",
        "fine_tuned_layers": ["layer4", "fc"],
    },
    MODEL_OUT,
)

print(f"\nSaved model to: {MODEL_OUT}")
