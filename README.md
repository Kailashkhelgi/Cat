![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)


# 🐾 Cat vs Dog Image Classifier — End-to-End ML Engineering Project

An end-to-end **Machine Learning Engineering** project that builds, evaluates, and deploys a **Deep Learning Image Classifier** using **Transfer Learning (ResNet18)**.

The system classifies images of **Cats vs Dogs**, includes full evaluation and error analysis, and is deployed with a **FastAPI inference service** and an interactive **Streamlit web UI**.

---

## 🚀 Project Highlights

✔ Transfer Learning using **ResNet18 (pretrained on ImageNet)**  
✔ Fine-tuning last convolution block for domain adaptation  
✔ GPU training with CUDA  
✔ Clean data pipeline with train/validation split  
✔ Evaluation with confusion matrix & classification report  
✔ Error analysis using misclassified image inspection  
✔ REST API deployment using **FastAPI**  
✔ Interactive browser UI built with **Streamlit**

---

## 🧠 Model Architecture

| Component | Details |
|----------|---------|
| Backbone | ResNet18 (ImageNet pretrained) |
| Fine-tuned Layers | `layer4` + `fc` |
| Input Size | 224 × 224 |
| Framework | PyTorch |
| Loss Function | CrossEntropyLoss |
| Optimizer | Adam |

---

## 📊 Model Performance

| Metric | Score |
|-------|-------|
| Validation Accuracy | **98.0%** |
| Precision | 0.98 |
| Recall | 0.98 |
| F1 Score | 0.98 |

Confusion matrix and classification reports are generated using `scikit-learn`.

---

## 🗂 Dataset

**PetImages (Cats vs Dogs)** [dataset.]( https://www.microsoft.com/en-us/download/details.aspx?id=54765&utm_source=chatgpt.com )

Data processing steps:
- Removed corrupted images
- Verified valid formats
- Split into **80% train / 20% validation**
- Applied normalization and augmentation

---

## 🏗 Project Structure

```text

├── train.py # Training + fine-tuning script
├── evaluate.py # Evaluation metrics & confusion matrix
├── mistakes.py # Saves misclassified examples for analysis
├── app.py # FastAPI inference server
├── ui.py # Streamlit web interface
├── requirements.txt # Project dependencies
├── README.md # Project documentation
└── data/ # Dataset (not included in repo)
```


---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/cat-dog-classifier.git
cd cat-dog-classifier
```
---
### 2️⃣ Create Virtual Environment

```sh
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```
---
### 🏋️ Train the Model
```
python train.py
```
### 📈 Evaluate the Model
```sh
python evaluate.py
```
---
### 🔍 Error Analysis
```
python mistakes.py
```
This saves misclassified images for inspection in a mistakes/ folder.

---

### 🚀 Run the Inference API
```sh
uvicorn app:app --reload
```
API docs available at: 👉 http://127.0.0.1:8000/docs

---
### 🌐 Run the Web UI
```sh
streamlit run ui.py
```
Then open: 👉 http://localhost:8501

Upload an image and get predictions in real time.
---
## 🛠 Tech Stack

1. Python
2. PyTorch
3. Torchvision
4. FastAPI
5. Streamlit
6. Scikit-learn
---
## 💼 ML Engineering Skills Demonstrated

This project showcases practical ML engineering abilities:

1. Building reproducible training pipelines
2. Transfer learning & fine-tuning
3. Model evaluation and error analysis
4. Model checkpointing and loading
5. REST API model serving
6. Interactive UI for inference
---

### Screenshot

<img width="1913" height="877" alt="image" src="https://github.com/user-attachments/assets/d6f098d8-0c52-4c8b-98e8-4ed9077431d7" />

<img width="1912" height="875" alt="image" src="https://github.com/user-attachments/assets/e1c03c9c-a425-4633-86a8-7b1a076f90cf" />

---


___Thank You___







