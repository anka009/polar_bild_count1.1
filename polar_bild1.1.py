import streamlit as st
import cv2
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from PIL import Image

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
    # Bild einlesen
    data = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    # BGR → RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # HSV
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    v_uint8 = v.astype(np.uint8)

    # Originalbild
    original_pil = Image.fromarray(img_rgb)

    # -------------------------
    # Thresholding
    # -------------------------
    if mode == "Manuell":
        mask_thresh = (v_uint8 > manual_thresh).astype(np.uint8) * 255
    else:
        otsu_val, _ = cv2.threshold(v_uint8, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_val = np.clip(otsu_val + offset, 0, 255)
        mask_thresh = (v_uint8 > thresh_val).astype(np.uint8) * 255

    # Adaptive Threshold
    adaptive_thresh = cv2.adaptiveThreshold(
        v_uint8, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 15, -5
    )
    combined_mask = cv2.bitwise_or(mask_thresh, adaptive_thresh)

    # Sättigungsfilter
    sat_mask = (s > sat_min).astype(np.uint8) * 255
    collagen_mask = cv2.bitwise_and(combined_mask, sat_mask)

    # Morphologische Reinigung
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
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

    # -------------------------
    # Hue-Klassifikation
    # -------------------------
    red_mask = (
        (((h >= 0) & (h <= red_max)) |
         ((h >= 170) & (h <= 179))) & cm
    )
    orange_mask = ((h >= orange_low) & (h <= orange_high)) & cm
    green_mask = ((h >= green_low) & (h <= green_high)) & cm

    # -------------------------
    # Quantifizierung (gewichtete Relation)
    # -------------------------
    red_px = np.sum(red_mask)
    orange_px = np.sum(orange_mask)
    green_px = np.sum(green_mask)

    # Gewichtete Flächen
    eff_red = red_px + 0.5 * orange_px
    eff_green = green_px + 0.5 * orange_px

    # Verhältnis
    if eff_green > 0:
        red_green_ratio = eff_red / eff_green
    else:
        red_green_ratio = None

    # Gesamtfläche
    total_area = np.sum(cm)
    
    # Klassifizierte Fläche (nur Rot/Orange/Grün) 
    classified_area = red_px + orange_px + green_px 
    # Nicht klassifizierbare Kollagenpixel 
    unclassified_area = total_area - classified_area
    
    # -------------------------
    # Overlay
    # -------------------------
    overlay = img_rgb.copy()
    overlay[red_mask] = [255, 0, 0]
    overlay[orange_mask] = [255, 165, 0]
    overlay[green_mask] = [0, 255, 0]

    return {
        "Image": file.name,
        "original": original_pil,
        "mask": collagen_mask,
        "overlay": overlay,
        "Red (px)": red_px,
        "Orange (px)": orange_px,
        "Green (px)": green_px,
        "Eff_Red": eff_red,
        "Eff_Green": eff_green,
        "Red/Green Ratio": red_green_ratio,
        "Total Collagen Area (px)": total_area
        "Classified Collagen (px)": classified_area,
        "Unclassified Collagen (px)": unclassified_area 
    }

# -------------------------
# Main
# -------------------------
results = []

if uploaded:
    for f in uploaded:
        f.seek(0)
        results.append(analyze_image(f))

    # DataFrame erstellen
    df = pd.DataFrame(results)

    # Alle Bild-Spalten entfernen (PIL Images → ArrowError)
    image_columns = ["overlay", "mask", "original"]
    for col in image_columns:
        if col in df.columns:
            df = df.drop(columns=[col])

    st.subheader("📄 Ergebnisse (Rot/Grün Relation)")
    st.dataframe(df)

    st.download_button(
        "📥 CSV herunterladen",
        df.to_csv(index=False).encode("utf-8"),
        "psr_collagen_results.csv"
    )

    # 👉 NEUER BLOCK: Anzeigeoptionen + dynamische Bildanzeige
    st.subheader("🔧 Anzeigeoptionen")

    show_overlay = st.checkbox("Overlay anzeigen", value=True)
    show_mask = st.checkbox("Maske anzeigen", value=True)
    show_original = st.checkbox("Originalbild anzeigen", value=True)

    st.subheader("🔍 Qualitätskontrolle")

    for r in results:
        st.markdown(f"### {r['Image']}")

        # Reihenfolge festlegen: overlay → maske → original
        ordered_keys = []
        if show_overlay:
            ordered_keys.append("overlay")
        if show_mask:
            ordered_keys.append("mask")
        if show_original:
            ordered_keys.append("original")

        # Dynamisch Spalten erzeugen
        cols = st.columns(len(ordered_keys))

        # Bilder in Spalten platzieren
        for col, key in zip(cols, ordered_keys):
            with col:
                st.image(
                    r[key],
                    caption=key.capitalize(),
                    use_column_width=True
                )

else:
    st.info("Bitte PSR-Bilder hochladen.")
