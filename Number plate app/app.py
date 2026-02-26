import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image

model = YOLO("best.pt")

st.title("🚗 Number Plate Detection App")
st.write("Upload an image to detect vehicle number plates.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    st.image(image, caption="Original Image", use_column_width=True)

    results = model.predict(img_array, conf=0.25)

    annotated_image = results[0].plot()

    st.image(annotated_image, caption="Detected Image", use_column_width=True)