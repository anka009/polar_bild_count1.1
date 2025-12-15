import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Polarized Collagen Quantification", layout="wide")
st.title("📊 Polarized Collagen Quantifier (Kollagen I, III, I+III)")

# -------------------------
# Sidebar Settings
# -------------------------
st.sidebar.header("Threshold-Einstellungen")
mode = st.sidebar.radio(
    "Threshold-Modus",
    ["Auto+Offset", "Manuell"],
    key="threshold_mode_radio"   # eindeutiger Schlüssel
)

otsu_placeholder = st.sidebar.empty()
offset = st.sidebar.slider("Feinjustierung (Offset ±)", -50, 50, 0, key="offset_slider")
manual_thresh = st.sidebar.slider("Manueller Threshold", 0, 255, 120, key="manual_thresh_slider")

# Hue-Regler für Kollagen I (Rot)
st.sidebar.header("Hue‑Filter Kollagen I (Rot)")
hue_red_low   = st.sidebar.slider("Rot: Hue LOW", 0, 30, 25, key="hue_red_low_slider")
hue_red_high  = st.sidebar.slider("Rot: Hue HIGH", 150, 180, 160, key="hue_red_high_slider")

# Hue-Regler für Kollagen III (Grün)
st.sidebar.header("Hue‑Filter Kollagen III (Grün)")
hue_green_low = st.sidebar.slider("Grün: Hue LOW", 30, 90, 40, key="hue_green_low_slider")
hue_green_high= st.sidebar.slider("Grün: Hue HIGH", 60, 120, 90, key="hue_green_high_slider")

# Hue-Regler für Mischfasern (I+III)
st.sidebar.header("Hue‑Filter Kollagen I+III (Mischfasern)")
hue_mix_low   = st.sidebar.slider("Misch: Hue LOW", 0, 180, 20, key="hue_mix_low_slider")
hue_mix_high  = st.sidebar.slider("Misch: Hue HIGH", 0, 180, 160, key="hue_mix_high_slider")

# File Upload
uploaded = st.sidebar.file_uploader(
    "Bilder hochladen (TIFF/JPG/PNG)",
    accept_multiple_files=True,
    key="file_uploader_widget"
)

results = []

# -------------------------
# Analyse-Funktion
# -------------------------
def analyze_image(file_bytes, fname,
                  mode, offset, manual_thresh,
                  hue_red_low, hue_red_high,
                  hue_green_low, hue_green_high,
                  hue_mix_low, hue_mix_high,
                  otsu_placeholder):
    data = np.asarray(bytearray(file_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    brightness = hsv[:,:,2]

    # Auto-Otsu
    otsu_val, _ = cv2.threshold(brightness, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_placeholder.markdown(f"**Auto-Otsu vorgeschlagen: {int(otsu_val)}**")

    manual_val = int(np.clip(otsu_val + offset, 0, 255))

    # Threshold anwenden je nach Modus
    if mode == "Manuell":
        _, mask = cv2.threshold(brightness, manual_thresh, 255, cv2.THRESH_BINARY)
    else:
        _, mask = cv2.threshold(brightness, manual_val, 255, cv2.THRESH_BINARY)

    # Morphologische Reinigung
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Farbklassifizierung
    h = hsv[:,:,0]
    red_mask   = np.logical_and((h < hue_red_low) | (h > hue_red_high), mask_clean > 0)
    green_mask = np.logical_and((h > hue_green_low) & (h < hue_green_high), mask_clean > 0)
    mix_mask   = np.logical_and((h > hue_mix_low) & (h < hue_mix_high), mask_clean > 0)

    # Exklusive Masken
    red_mask_exclusive   = np.logical_and(red_mask, ~mix_mask)
    green_mask_exclusive = np.logical_and(green_mask, ~mix_mask)

    # Pixelanzahl
    red_pixels   = int(np.sum(red_mask_exclusive))
    green_pixels = int(np.sum(green_mask_exclusive))
    mix_pixels   = int(np.sum(mix_mask))
    total_pixels = red_pixels + green_pixels + mix_pixels

    # Intensitäten
    red_intensity   = float(np.sum(brightness[red_mask_exclusive]))
    green_intensity = float(np.sum(brightness[green_mask_exclusive]))
    mix_intensity   = float(np.sum(brightness[mix_mask]))
    total_intensity = red_intensity + green_intensity + mix_intensity

    # Intensität pro Pixel
    red_intensity_per_pixel   = red_intensity / (red_pixels + 1e-6)
    green_intensity_per_pixel = green_intensity / (green_pixels + 1e-6)
    mix_intensity_per_pixel   = mix_intensity / (mix_pixels + 1e-6)
    total_intensity_per_pixel = total_intensity / (total_pixels + 1e-6)

    # Overlay
    overlay = img_rgb.copy()
    overlay[red_mask_exclusive]   = [255, 0, 0]     # Rot = Kollagen I
    overlay[green_mask_exclusive] = [0, 255, 0]     # Grün = Kollagen III
    overlay[mix_mask]             = [255, 165, 0]   # Orange = Mischfasern I+III

    return {
        "image": fname,
        "collagen_pixels": total_pixels,
        "collagen_intensity": total_intensity,
        "collagen_intensity_per_pixel": total_intensity_per_pixel,
        "collagen1_pixels": red_pixels,
        "collagen1_intensity": red_intensity,
        "collagen1_intensity_per_pixel": red_intensity_per_pixel,
        "collagen3_pixels": green_pixels,
        "collagen3_intensity": green_intensity,
        "collagen3_intensity_per_pixel": green_intensity_per_pixel,
        "collagen13_pixels": mix_pixels,
        "collagen13_intensity": mix_intensity,
        "collagen13_intensity_per_pixel": mix_intensity_per_pixel,
        "mask_clean": mask_clean,
        "red_mask": red_mask_exclusive,
        "green_mask": green_mask_exclusive,
        "mix_mask": mix_mask,
        "overlay": overlay
    }

# -------------------------
# Main Execution
# -------------------------
if uploaded:
    for f in uploaded:
        f.seek(0)
        results.append(analyze_image(
            f, f.name,
            mode, offset, manual_thresh,
            hue_red_low, hue_red_high,
            hue_green_low, hue_green_high,
            hue_mix_low, hue_mix_high,
            otsu_placeholder
        ))

    # Ergebnisse-Tabelle
    df = pd.DataFrame(results)[[
        "image",
        "collagen_pixels","collagen_intensity","collagen_intensity_per_pixel",
        "collagen1_pixels","collagen1_intensity","collagen1_intensity_per_pixel",
        "collagen3_pixels","collagen3_intensity","collagen3_intensity_per_pixel",
        "collagen13_pixels","collagen13_intensity","collagen13_intensity_per_pixel"
    ]]
    st.subheader("📄 Ergebnisse")
    st.dataframe(df)

    # CSV-Download
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 CSV herunterladen", csv,
                       "collagen_results.csv", "text/csv")

    # Debug-Ansicht
    st.subheader("🔍 Debug‑Ansicht pro Bild")
    for r in results:
        st.markdown(f"### {r['image']}")
        st.image(r['mask_clean'], caption="Gesamt Kollagen-Maske")
        st.image(r['overlay'], caption="Overlay: Rot • Grün • Orange (Mischfasern)")
        st.image(r['red_mask'], caption="Kollagen I (Rot)")
        st.image(r['green_mask'], caption="Kollagen III (Grün)")
        st.image(r['mix_mask'], caption="Kollagen I+III (Mischfasern)")
else:
    st.info("Bitte Bilder hochladen, um die Analyse zu starten.")

st.set_page_config(page_title="Polarized Collagen Quantification", layout="wide")
st.title("📊 Polarized Collagen Batch‑Quantifier (Rot/Grün/Mischfasern)")

# -------------------------
# Sidebar Settings
# -------------------------
st.sidebar.header("Threshold-Einstellungen")
mode = st.sidebar.radio(
    "Threshold-Modus",
    ["Auto+Offset", "Manuell"],
    key="threshold_mode"
)

otsu_placeholder = st.sidebar.empty()
offset = st.sidebar.slider("Feinjustierung (Offset ±)", -50, 50, 0, key="offset_slider")
manual_thresh = st.sidebar.slider("Manueller Threshold", 0, 255, 120, key="manual_slider")

st.sidebar.header("Hue‑Filter Einstellungen")
hue_red_low   = st.sidebar.slider("Rot: Hue LOW", 0, 30, 25, key="red_low")
hue_red_high  = st.sidebar.slider("Rot: Hue HIGH", 150, 180, 160, key="red_high")
hue_green_low = st.sidebar.slider("Grün: Hue LOW", 30, 90, 40, key="green_low")
hue_green_high= st.sidebar.slider("Grün: Hue HIGH", 60, 120, 90, key="green_high")

uploaded = st.sidebar.file_uploader(
    "Bilder hochladen (TIFF/JPG/PNG)",
    accept_multiple_files=True,
    key="file_uploader"
)

results = []

# -------------------------
# Analyse-Funktion
# -------------------------
def analyze_image(file_bytes, fname,
                  mode, offset, manual_thresh,
                  hue_red_low, hue_red_high,
                  hue_green_low, hue_green_high,
                  otsu_placeholder):
    data = np.asarray(bytearray(file_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    brightness = hsv[:,:,2]

    # Auto-Otsu
    otsu_val, _ = cv2.threshold(brightness, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_placeholder.markdown(f"**Auto-Otsu vorgeschlagen: {int(otsu_val)}**")

    manual_val = int(np.clip(otsu_val + offset, 0, 255))

    # Threshold anwenden je nach Modus
    if mode == "Manuell":
        _, mask = cv2.threshold(brightness, manual_thresh, 255, cv2.THRESH_BINARY)
    else:
        _, mask = cv2.threshold(brightness, manual_val, 255, cv2.THRESH_BINARY)

    # Morphologische Reinigung
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Farbklassifizierung
    h = hsv[:,:,0]
    red_mask   = np.logical_and((h < hue_red_low) | (h > hue_red_high), mask_clean > 0)
    green_mask = np.logical_and((h > hue_green_low) & (h < hue_green_high), mask_clean > 0)

    # Mischfasern = Überschneidung
    mixed_mask = np.logical_and(red_mask, green_mask)

    # Exklusive Masken
    red_mask_exclusive   = np.logical_and(red_mask, ~mixed_mask)
    green_mask_exclusive = np.logical_and(green_mask, ~mixed_mask)

    # Zählungen
    red_pixels   = int(np.sum(red_mask_exclusive))
    green_pixels = int(np.sum(green_mask_exclusive))
    mixed_pixels = int(np.sum(mixed_mask))

    ratio_red_green = float((red_pixels + 1e-6) / (green_pixels + 1e-6))

    # Overlay mit drei Farben
    overlay = img_rgb.copy()
    overlay[red_mask_exclusive]   = [255, 0, 0]     # Rot
    overlay[green_mask_exclusive] = [0, 255, 0]     # Grün
    overlay[mixed_mask]           = [255, 255, 0]   # Gelb für Mischfasern

    return {
        "image": fname,
        "collagen_pixels": int(np.sum(mask_clean > 0)),
        "area_percent": (np.sum(mask_clean > 0) / mask_clean.size) * 100,
        "mean_intensity": float(np.mean(brightness[mask_clean > 0])),
        "red_pixels": red_pixels,
        "green_pixels": green_pixels,
        "mixed_pixels": mixed_pixels,
        "ratio_red_green": ratio_red_green,
        "mask_clean": mask_clean,
        "overlay": overlay
    }

# -------------------------
# Main Execution
# -------------------------
if uploaded:
    for f in uploaded:
        f.seek(0)
        results.append(analyze_image(
            f, f.name,
            mode, offset, manual_thresh,
            hue_red_low, hue_red_high,
            hue_green_low, hue_green_high,
            otsu_placeholder
        ))

    # Ergebnisse-Tabelle
    df = pd.DataFrame(results)[[
        "image","collagen_pixels","area_percent",
        "mean_intensity","red_pixels","green_pixels","mixed_pixels","ratio_red_green"
    ]]
    st.subheader("📄 Ergebnisse")
    st.dataframe(df)

    # CSV-Download
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 CSV herunterladen", csv,
                       "collagen_results.csv", "text/csv")

    # Debug-Ansicht
    st.subheader("🔍 Debug‑Ansicht pro Bild")
    for r in results:
        st.markdown(f"### {r['image']}")
        st.image(r['mask_clean'], caption="Kollagen-Maske")
        st.image(r['overlay'], caption="Overlay: Rot • Grün • Gelb (Mischfasern)")
else:
    st.info("Bitte Bilder hochladen, um die Analyse zu starten.")
