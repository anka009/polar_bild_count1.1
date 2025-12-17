import streamlit as st
import cv2
import numpy as np
import pandas as pd

# =====================================================
# App Config
# =====================================================
st.set_page_config(
    page_title="PSR Polarized Collagen Quantification",
    layout="wide"
)
st.title("📊 PSR Polarized Collagen Quantifier (Collagen I, III, I+III)")

# =====================================================
# Sidebar Controls
# =====================================================
st.sidebar.header("Threshold-Modus")
mode = st.sidebar.radio("Modus", ["Auto+Offset", "Manuell"])

offset = st.sidebar.slider("Otsu Offset (±)", -40, 40, 0)
manual_thresh = st.sidebar.slider("Manueller Threshold (Value V)", 0, 255, 120)

st.sidebar.header("Sättigungsfilter (gegen False Positives)")
sat_min = st.sidebar.slider("Min. Saturation (S)", 0, 255, 50)

st.sidebar.header("Hue – Kollagen III (Grün)")
green_low = st.sidebar.slider("Grün Hue LOW", 30, 90, 40)
green_high = st.sidebar.slider("Grün Hue HIGH", 60, 120, 90)

uploaded = st.sidebar.file_uploader(
    "PSR-Bilder hochladen (polarisiert)",
    type=["tif", "tiff", "png", "jpg"],
    accept_multiple_files=True
)

# =====================================================
# Analyse-Funktion
# =====================================================
def analyze_image(file):

    # --- Image load
    data = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    h = hsv[:, :, 0]   # 0–179
    s = hsv[:, :, 1]   # 0–255
    v = hsv[:, :, 2]   # 0–255

    # =================================================
    # Kollagen-Gesamtmaske (Value + Saturation)
    # =================================================
    otsu, _ = cv2.threshold(
        v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    thresh = manual_thresh if mode == "Manuell" else np.clip(otsu + offset, 0, 255)

    # ❗ WICHTIG: Maske muss 0/255 sein
    collagen_mask = ((v > thresh) & (s > sat_min)).astype(np.uint8) * 255

    # Morphologische Reinigung (PSR-typisch)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    collagen_mask = cv2.morphologyEx(collagen_mask, cv2.MORPH_OPEN, kernel)
    collagen_mask = cv2.morphologyEx(collagen_mask, cv2.MORPH_CLOSE, kernel)

    collagen_bool = collagen_mask.astype(bool)

    # =================================================
    # Hue-Klassifikation (OpenCV-konform)
    # =================================================
    # Kollagen I – Rot (zyklisch!)
    red_mask = (
        ((h >= 0) & (h <= 10)) |
        ((h >= 170) & (h <= 179))
    ) & collagen_bool

    # Kollagen III – Grün
    green_mask = (
        (h >= green_low) & (h <= green_high)
    ) & collagen_bool

    # Mischfasern = Überlappung
    mix_mask = red_mask & green_mask

    red_only = red_mask & ~green_mask
    green_only = green_mask & ~red_mask

    # =================================================
    # Quantifizierung (Area-based, paper-ready)
    # =================================================
    total_area = np.sum(collagen_bool)
    red_area = np.sum(red_only)
    green_area = np.sum(green_only)
    mix_area = np.sum(mix_mask)

    def pct(x):
        return 100 * x / (total_area + 1e-6)

    # =================================================
    # Overlay für QC
    # =================================================
    overlay = img_rgb.copy()
    overlay[red_only] = [255, 0, 0]       # Collagen I
    overlay[green_only] = [0, 255, 0]     # Collagen III
    overlay[mix_mask] = [255, 165, 0]     # Collagen I+III

    return {
        "Image": file.name,
        "Total Collagen Area (px)": total_area,
        "Collagen I Area (%)": pct(red_area),
        "Collagen III Area (%)": pct(green_area),
        "Collagen I+III Area (%)": pct(mix_area),
        "overlay": overlay,
        "mask": collagen_mask
    }

# =====================================================
# Main
# =====================================================
results = []

if uploaded:
    for f in uploaded:
        f.seek(0)
        results.append(analyze_image(f))

    # -------------------------
    # Results Table
    # -------------------------
    df = pd.DataFrame(results).drop(columns=["overlay", "mask"])
    st.subheader("📄 Quantitative Ergebnisse (Paper-ready)")
    st.dataframe(df)

    st.download_button(
        "📥 CSV herunterladen",
        df.to_csv(index=False).encode("utf-8"),
        "psr_collagen_quantification.csv"
    )

    # -------------------------
    # QC View
    # -------------------------
    st.subheader("🔍 Qualitätskontrolle (pro Bild)")
    for r in results:
        st.markdown(f"### {r['Image']}")
        st.image(r["mask"], caption="Gesamt-Kollagen-Maske")
        st.image(
            r["overlay"],
            caption="Overlay: Rot (Collagen I) · Grün (Collagen III) · Orange (I+III)"
        )
else:
    st.info("Bitte PSR-Bilder hochladen, um die Analyse zu starten.")
