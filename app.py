import streamlit as st
import cv2
import numpy as np
from utils.detection_utils import detect_haar, detect_ssd

st.title("Facemask Detection (Haar vs SSD)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Haar Cascade")
        haar_boxes = detect_haar(image)
        img_copy = image.copy()
        for box in haar_boxes:
            cv2.rectangle(img_copy, (box[0], box[1]), (box[2], box[3]), (0,255,0), 2)
        st.image(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB), channels="RGB")

    with col2:
        st.subheader("SSD")
        ssd_boxes = detect_ssd(image)
        img_copy = image.copy()
        for box in ssd_boxes:
            cv2.rectangle(img_copy, (box[0], box[1]), (box[2], box[3]), (255,0,0), 2)
        st.image(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB), channels="RGB")