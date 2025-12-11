import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Polarized Collagen Quantification", layout="wide")
st.title("📊 Polarized Collagen Batch‑Quantifier (Manual Threshold + Hue-Filter)")

# -------------------------
# Sidebar Settings
# -------------------------
st.sidebar.header("Brightness‑Threshold (manuell)")
manual_thresh = st.sidebar.slider("Threshold auf Brightness", 0, 255, 120)

st.sidebar.header("Hue‑Filter Einstellungen")
hue_thick_low = st.sidebar.slider("Dicke Fasern: Hue LOW", 0, 30, 25)
hue_thick_high = st.sidebar.slider("Dicke Fasern: Hue HIGH", 150, 180, 160)
hue_thin_low = st.sidebar.slider("Dünne Fasern: Hue LOW", 30, 90, 40)
hue_thin_high = st.sidebar.slider("Dünne Fasern: Hue HIGH", 60, 120, 90)

uploaded = st.sidebar.file_uploader("Bilder hochladen (TIFF/JPG/PNG)", accept_multiple_files=True)

results = []

# -------------------------
# Analyse-Funktion
# -------------------------
def analyze_image(file_bytes, fname):
    data = np.asarray(bytearray(file_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    brightness = hsv[:,:,2]

    # Manuelle Threshold-Maske
    _, mask = cv2.threshold(brightness, manual_thresh, 255, cv2.THRESH_BINARY)

    # Reinigende Morphologie
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Hue basierte Faserklassifizierung
    h = hsv[:,:,0]

    thick = ((h < hue_thick_low) | (h > hue_thick_high))
    thin  = ((h > hue_thin_low) & (h < hue_thin_high))

    thick_mask = np.logical_and(thick, mask_clean > 0)
    thin_mask  = np.logical_and(thin,  mask_clean > 0)

    collagen_area = int(np.sum(mask_clean > 0))
    total_area = mask_clean.size
    frac = collagen_area / total_area * 100

    mean_intensity = float(np.mean(brightness[mask_clean > 0]))

    ratio = float((np.sum(thick_mask) + 1e-6) / (np.sum(thin_mask) + 1e-6))

    # Prepare images for debugging output
    overlay = img_rgb.copy()
    overlay[thick_mask] = [255, 0, 0]
    overlay[thin_mask] = [0, 255, 0]

    return {
        "image": fname,
        "collagen_pixels": collagen_area,
        "area_percent": frac,
        "mean_intensity": mean_intensity,
        "thick_pixels": int(np.sum(thick_mask)),
        "thin_pixels": int(np.sum(thin_mask)),
        "ratio_thick_thin": ratio,
        "mask_clean": mask_clean,
        "overlay": overlay
    }

# -------------------------
# Main Execution
# -------------------------
if uploaded:
    for f in uploaded:
        f.seek(0)
        results.append(analyze_image(f, f.name))

    df = pd.DataFrame(results)[["image","collagen_pixels","area_percent","mean_intensity","thick_pixels","thin_pixels","ratio_thick_thin"]]
    st.subheader("📄 Ergebnisse")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 CSV herunterladen", csv, "collagen_results.csv", "text/csv")

    st.subheader("🔍 Debug‑Ansicht pro Bild")
    for r in results:
        st.markdown(f"### {r['image']}")
        st.image(r['mask_clean'], caption="Kollagen-Maske (manueller Threshold)")
        st.image(r['overlay'], caption="Overlay: Rot = dick • Grün = dünn")
else:
    st.info("Bitte Bilder hochladen, um die Analyse zu starten.")
