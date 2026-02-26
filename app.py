from ultralytics import YOLO
import streamlit as st
from PIL import Image
import numpy as np

st.title("🚗 Number Plate Detection App")

model = YOLO("best.pt")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    img_array = np.array(image)

    results = model.predict(img_array, imgsz=640, conf=0.25)

    result_img = results[0].plot()

    st.image(result_img, caption="Detected Image")
