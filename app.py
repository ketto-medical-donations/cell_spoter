import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("RBC Cell Detection")
st.write("Upload a high-resolution RBC smear image (up to ~50 cells per field).")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    output = img.copy()

    # --------------------------
    # 1. Convert to grayscale
    # --------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --------------------------
    # 2. Adaptive threshold
    # --------------------------
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51, 3
    )

    # --------------------------
    # 3. Build mask of violet cells (to EXCLUDE them)
    # --------------------------
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_violet = np.array([115, 60, 40])  
    upper_violet = np.array([160, 255, 255])

    violet_mask = cv2.inRange(hsv, lower_violet, upper_violet)

    kernel = np.ones((5,5), np.uint8)
    violet_mask = cv2.morphologyEx(violet_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # --------------------------
    # 4. Remove violet objects 
    # --------------------------
    clean = th.copy()
    clean[violet_mask > 0] = 0

    # --------------------------
    # 5. Remove small noise
    # --------------------------
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)

    # --------------------------
    # 6. Find non-violet contours
    # --------------------------
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --------------------------
    # 7. Draw GREEN outlines 
    # --------------------------
    for cnt in contours:
        if cv2.contourArea(cnt) > 80:
            cv2.drawContours(output, [cnt], -1, (0,255,0), 2)

    # --------------------------
    # 8. Display results
    # --------------------------
    st.subheader("Processed Output")

    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    st.image(output_rgb, channels="RGB", caption="Detected RBC (violet excluded)")

    st.success("Done! Your image has been processed.")
