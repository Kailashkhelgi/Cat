"""
app.py

Simple FastAPI server for Cat vs Dog classifier.
Allows image upload and returns predicted class + confidence.

Run:
  uvicorn app:app --reload
Then open:
  http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

import torch
from torch import nn
from torchvision import transforms, models

# -----------------------------
# Config
# -----------------------------
CHECKPOINT_PATH = "resnet18_petimages.pth"  # change if your filename differs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# -----------------------------
# Load model once at startup
# -----------------------------
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
class_names = ckpt["class_names"]
img_size = ckpt.get("img_size", 224)

model = models.resnet18(weights=None)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, len(class_names))
model.load_state_dict(ckpt["model_state"])
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

app = FastAPI(title="Cat vs Dog Classifier API")


@app.get("/")
def root():
    return {"message": "Cat vs Dog classifier is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read uploaded image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Preprocess
    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    pred_idx = torch.argmax(probs).item()
    confidence = probs[pred_idx].item()

    return {
        "prediction": class_names[pred_idx],
        "confidence": round(confidence, 4)
    }
