import streamlit as st
import cv2
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops

st.set_page_config(page_title="PSR Collagen Quantification – Paper-ready", layout="wide")
st.title("📊 PSR Polarized Collagen Quantifier (Feine Fasern + Längenfilter)")

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Analyse-Modus")
mode = st.sidebar.radio(
    "Modus",
    ["Semiauto", "Auto+Offset"],
    index=0  # Semiauto als Default
)

offset = st.sidebar.slider("Otsu Offset", -40, 40, -10)
manual_thresh = st.sidebar.slider("Manueller Threshold (V)", 0, 255, 110)

st.sidebar.header("Sättigungsfilter")
sat_min = st.sidebar.slider("Min. Saturation (S)", 0, 20, 5)

st.sidebar.header("Hue-Bereiche (OpenCV HSV)")
red_max = st.sidebar.slider("Rot max", 5, 15, 10)
orange_low = st.sidebar.slider("Orange low", 8, 20, 12)
orange_high = st.sidebar.slider("Orange high", 20, 40, 30)
green_low = st.sidebar.slider("Grün low", 30, 60, 40)
green_high = st.sidebar.slider("Grün high", 60, 120, 90)

st.sidebar.header("Objektfilter")
min_length = st.sidebar.slider("Minimale Faserlänge (px)", 1, 100, 10)
min_area   = st.sidebar.slider("Minimale Fläche (px²)", 1, 20, 5)

st.sidebar.header("PSR Bilder hochladen")
uploaded = st.sidebar.file_uploader(
    "Dateien auswählen",
    type=["tif", "tiff", "png", "jpg"],
    accept_multiple_files=True
)

# -------------------------
# Checkbox einmalig für alle Bilder
# -------------------------
use_manual_thresh = st.sidebar.checkbox("Manuellen Threshold verwenden", value=False)

# -------------------------
# Analyse-Funktion
# -------------------------
def analyze_image(file, use_manual_thresh=False):
    data = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    v_uint8 = v.astype(np.uint8)

    # -------------------------
    # Thresholding
    # -------------------------
    if mode == "Semiauto":
        otsu_val, _ = cv2.threshold(v_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_val = np.clip(otsu_val + offset, 0, 255)
        if use_manual_thresh:
            thresh_val = manual_thresh
        mask_thresh = (v_uint8 > thresh_val).astype(np.uint8) * 255
    else:  # Auto+Offset
        otsu_val, _ = cv2.threshold(v_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_val = np.clip(otsu_val + offset, 0, 255)
        mask_thresh = (v_uint8 > thresh_val).astype(np.uint8) * 255

    # Adaptive Threshold für feine Fasern
    if st.sidebar.checkbox("Adaptive Threshold aktivieren", value=True):
        adaptive_thresh = cv2.adaptiveThreshold(
            v_uint8, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -5
        )
        combined_mask = cv2.bitwise_or(mask_thresh, adaptive_thresh)
    else:
        combined_mask = mask_thresh

    # Sättigungsfilter
    sat_mask = (s > sat_min).astype(np.uint8) * 255
    collagen_mask = cv2.bitwise_and(combined_mask, sat_mask)

    # Morphologische Reinigung
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    collagen_mask = cv2.morphologyEx(collagen_mask, cv2.MORPH_OPEN, kernel)
    collagen_mask = cv2.morphologyEx(collagen_mask, cv2.MORPH_CLOSE, kernel)

    # -------------------------
    # Filter nach Länge / Fläche
    # -------------------------
    labels = label(collagen_mask)
    filtered_mask = np.zeros_like(collagen_mask)
    for region in regionprops(labels):
        if region.major_axis_length >= min_length and region.area >= min_area:
            filtered_mask[labels == region.label] = 255
    collagen_mask = filtered_mask
    cm = collagen_mask.astype(bool)

    # -----------
