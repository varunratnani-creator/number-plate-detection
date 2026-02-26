from ultralytics import YOLO
import streamlit as st
from PIL import Image
import numpy as np
import os

# Prevent OpenCV from using GUI backend
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

st.title("🚗 Number Plate Detection App")

model = YOLO("best.pt")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image)

    results = model(img_array)

    result_img = results[0].plot()

    st.image(result_img, caption="Detected Image", use_column_width=True)
