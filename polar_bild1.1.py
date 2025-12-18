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
st.sidebar.header("Threshold-Modus")
mode = st.sidebar.radio("Modus", ["Auto+Offset", "Manuell"])
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
apply_cutoff = st.sidebar.checkbox("Längen-Cutoff aktivieren", value=True)
min_length = st.sidebar.slider("Minimale Faserlänge (px)", 1, 100, 10)
min_area   = st.sidebar.slider("Minimale Fläche (px²)", 1, 20, 5)

uploaded = st.sidebar.file_uploader(
    "PSR Bilder hochladen",
    type=["tif", "tiff", "png", "jpg"],
    accept_multiple_files=True
)

# -------------------------
# Analyse-Funktion
# -------------------------
def analyze_image(file):
    data = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    v_uint8 = v.astype(np.uint8)

    # -------------------------
    # Thresholding
    # -------------------------
    bright_fg = st.sidebar.checkbox("Kollagen heller als Hintergrund (V)", value=True)
    if mode == "Manuell":
        mask_thresh = (v_uint8 > manual_thresh).astype(np.uint8) * 255
    else:
        otsu_val, _ = cv2.threshold(v_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_val = np.clip(otsu_val + offset, 0, 255)
        mask_thresh = (v_uint8 > thresh_val).astype(np.uint8) * 255 if bright_fg else (v_uint8 < thresh_val).astype(np.uint8) * 255

    # Adaptive Threshold für feine Fasern
    adaptive_thresh = cv2.adaptiveThreshold(
        v_uint8, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -5
    )
    combined_mask = cv2.bitwise_or(mask_thresh, adaptive_thresh)

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
        if (not apply_cutoff or region.major_axis_length >= min_length) and region.area >= min_area:
            filtered_mask[labels == region.label] = 255
    collagen_mask = filtered_mask
    cm = collagen_mask.astype(bool)

    # -------------------------
    # Hue-Klassifikation
    # -------------------------
    red_mask = (((h >= 0) & (h <= red_max)) | ((h >= 170) & (h <= 179))) & cm
    orange_mask = ((h >= orange_low) & (h <= orange_high)) & cm
    green_mask = ((h >= green_low) & (h <= green_high)) & cm

    # -------------------------
    # Quantifizierung Rot/Grün Relation
    # Orange wird als Remis betrachtet
    # -------------------------
    red_px = np.sum(red_mask)
    green_px = np.sum(green_mask)
    total_classified = red_px + green_px
    red_rel = 100 * red_px / (total_classified + 1e-6)
    green_rel = 100 * green_px / (total_classified + 1e-6)

    # Gesamtmaske (inklusive Mischfasern)
    total_area = np.sum(cm)

    # -------------------------
    # Overlay
    # -------------------------
    overlay = img_rgb.copy()
    overlay[red_mask] = [255, 0, 0]
    overlay[orange_mask] = [255, 165, 0]
    overlay[green_mask] = [0, 255, 0]

    return {
        "Image": file.name,
        "Total Collagen Area (px)": total_area,
        "Collagen I (red %)": red_rel,
        "Collagen III (green %)": green_rel,
        "overlay": overlay,
        "mask": collagen_mask
    }

# -------------------------
# Main
# -------------------------
results = []

if uploaded:
    for f in uploaded:
        f.seek(0)
        results.append(analyze_image(f))

    df = pd.DataFrame(results).drop(columns=["overlay", "mask"])
    st.subheader("📄 Ergebnisse (Rot/Grün Relation)")
    st.dataframe(df)

    st.download_button(
        "📥 CSV herunterladen",
        df.to_csv(index=False).encode("utf-8"),
        "psr_collagen_results.csv"
    )

    st.subheader("🔍 Qualitätskontrolle")
    for r in results:
        st.markdown(f"### {r['Image']}")
        st.image(r["mask"], caption="Gesamt-Kollagen-Maske inkl. feiner Fasern")
        st.image(
            r["overlay"],
            caption="Overlay: Rot (I) · Orange (I+III, Remis) · Grün (III)"
        )
else:
    st.info("Bitte PSR-Bilder hochladen.")
