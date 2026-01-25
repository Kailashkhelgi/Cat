"""
ui.py

Streamlit UI for the Cat vs Dog FastAPI model.

How it works:
- You upload an image in the browser.
- The UI sends it to FastAPI POST /predict
- Shows predicted label + confidence.

Run:
  1) Start API in one terminal:
       uvicorn app:app --reload
  2) Start UI in another terminal:
       streamlit run ui.py
"""

import io
import requests
import streamlit as st
from PIL import Image

# -----------------------------
# Config
# -----------------------------
API_URL = "http://127.0.0.1:8000/predict"


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐾", layout="centered")
st.title("🐾 Cat vs Dog Classifier")
st.write("Upload an image and get a prediction from your FastAPI model.")

st.info("Make sure your API is running: `uvicorn app:app --reload`")


# -----------------------------
# Image uploader
# -----------------------------
uploaded_file = st.file_uploader("Choose an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the image
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    st.image(image, caption="Uploaded image", use_container_width=True)

    # Predict button (so it doesn't auto-call repeatedly)
    if st.button("Predict"):
        try:
            # Send file to FastAPI as multipart/form-data
            files = {"file": (uploaded_file.name, image_bytes, uploaded_file.type)}
            response = requests.post(API_URL, files=files, timeout=30)

            # Handle API response
            if response.status_code == 200:
                result = response.json()
                pred = result.get("prediction", "N/A")
                conf = result.get("confidence", "N/A")

                st.success(f"Prediction: **{pred}**")
                st.write(f"Confidence: **{conf}**")
            else:
                st.error(f"API error {response.status_code}: {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Is FastAPI running at http://127.0.0.1:8000 ?")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
