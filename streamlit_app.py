import streamlit as st
import torch
import numpy as np
from PIL import Image
import albumentations
import joblib
import os

from model import LcdModel
from config import IMAGE_HEIGHT, IMAGE_WIDTH, DEVICE
from train import decode_predictions, remove_duplicates

# Page configuration
st.set_page_config(page_title="OpticOps | Meter Reader", layout="centered")

# Sidebar
st.sidebar.title("OpticOps")
st.sidebar.markdown("#### Smart OCR for Digital Meter Reading")
st.sidebar.info(
    "This application uses a trained deep learning model to extract numeric readings from meter images. "
    "It's part of the OpticOps project which aims to automate utility data collection using AI."
)

st.sidebar.markdown("### How to Use")
st.sidebar.markdown("""
1. Upload a clear image of a digital meter (.jpg/.png)  
2. The system will preprocess and run OCR  
3. You’ll receive the extracted reading below  
4. Optionally, download the result as a text file  
""")

# Load model
@st.cache_resource
def load_model(num_chars):
    model = LcdModel(num_chars=num_chars)
    model.load_state_dict(torch.load("models/lcd_model_best.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# Load label encoder
@st.cache_resource
def load_label_encoder():
    return joblib.load("models/label_encoder.pkl")

# Image preprocessing
def preprocess_image(image):
    image = np.array(image.convert("RGB"))
    aug = albumentations.Compose([
        albumentations.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, always_apply=True),
        albumentations.Normalize(max_pixel_value=255.0, always_apply=True)
    ])
    image = aug(image=image)["image"]
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    return torch.tensor(image).unsqueeze(0)

# Main app
st.title("OpticOps: Meter Reading OCR")

uploaded_file = st.file_uploader("Upload a meter image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load encoder and model
    label_enc = load_label_encoder()
    model = load_model(num_chars=len(label_enc.classes_))

    # Preprocess
    image_tensor = preprocess_image(image).to(DEVICE)

    # Predict
    with torch.no_grad():
        preds = model(image_tensor)
        pred_text = decode_predictions(preds, label_enc)[0]

    # Postprocess
    cleaned_text = remove_duplicates(pred_text)

    # Show result
    st.markdown("### Extracted Meter Reading")
    st.code(cleaned_text, language='text')

    # Download button
    st.download_button(
        label="Download Reading as Text File",
        data=cleaned_text,
        file_name="meter_reading.txt",
        mime="text/plain"
    )