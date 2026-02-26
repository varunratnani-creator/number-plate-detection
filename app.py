import torch
torch.set_num_threads(1)

from ultralytics import YOLO
import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="Number Plate Detection", layout="centered")

st.title("🚗 Number Plate Detection App")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload a car image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image)

    with st.spinner("Detecting number plate..."):
        results = model(img_array, imgsz=320)

    result_img = results[0].plot()

    st.image(result_img, caption="Detection Result", use_column_width=True)
