import streamlit as st
import cv2
import numpy as np
import pandas as pd

st.set_page_config(page_title="PSR Polarized Collagen Quantification", layout="wide")
st.title("📊 PSR Polarized Collagen Quantifier (Collagen I, III, I+III)")

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Threshold-Modus")
mode = st.sidebar.radio("Modus", ["Auto+Offset", "Manuell"])

offset = st.sidebar.slider("Otsu Offset (±)", -40, 40, 0)
manual_thresh = st.sidebar.slider("Manueller Threshold (V)", 0, 255, 120)

st.sidebar.header("Sättigungsfilter")
sat_min = st.sidebar.slider("Min. Saturation (S)", 0, 255, 50)

st.sidebar.header("Hue – Kollagen I (Rot)")
red_low1, red_high1 = 0, 10
red_low2, red_high2 = 170, 180

st.sidebar.header("Hue – Kollagen III (Grün)")
green_low = st.sidebar.slider("Grün LOW", 30, 90, 40)
green_high = st.sidebar.slider("Grün HIGH", 60, 120, 90)

uploaded = st.sidebar.file_uploader(
    "PSR-Bilder hochladen",
    type=["tif", "tiff", "png", "jpg"],
    accept_multiple_files=True
)

# -------------------------
# Analyse
# -------------------------
def analyze_image(file):

    data = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

    # --- Kollagen-Maske (V + S)
    otsu, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = manual_thresh if mode == "Manuell" else np.clip(otsu + offset, 0, 255)

    collagen_mask = (v > thresh) & (s > sat_min)

    # Morphologie (PSR-typisch)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    collagen_mask = cv2.morphologyEx(collagen_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    collagen_mask = cv2.morphologyEx(collagen_mask, cv2.MORPH_CLOSE, kernel)

    # --- Hue-Klassifikation
    red_mask = (
        ((h >= red_low1) & (h <= red_high1)) |
        ((h >= red_low2) & (h <= red_high2))
    ) & collagen_mask.astype(bool)

    green_mask = (
        (h >= green_low) & (h <= green_high)
    ) & collagen_mask.astype(bool)

    # Mischfasern = Überlappung
    mix_mask = red_mask & green_mask

    red_only = red_mask & ~green_mask
    green_only = green_mask & ~red_mask

    # --- Pixel & Area
    total_area = np.sum(collagen_mask)
    red_area = np.sum(red_only)
    green_area = np.sum(green_only)
    mix_area = np.sum(mix_mask)

    # --- Prozentuale Flächen
    def pct(x): return 100 * x / (total_area + 1e-6)

    # --- Overlay
    overlay = img_rgb.copy()
    overlay[red_only]   = [255, 0, 0]
    overlay[green_only] = [0, 255, 0]
    overlay[mix_mask]   = [255, 165, 0]

    return {
        "Image": file.name,
        "Total Collagen Area (px)": total_area,
        "Collagen I Area (%)": pct(red_area),
        "Collagen III Area (%)": pct(green_area),
        "Collagen I+III Area (%)": pct(mix_area),
        "overlay": overlay,
        "mask": collagen_mask
    }

# -------------------------
# Run
# -------------------------
results = []

if uploaded:
    for f in uploaded:
        f.seek(0)
        results.append(analyze_image(f))

    df = pd.DataFrame(results).drop(columns=["overlay", "mask"])
    st.subheader("📄 Quantitative Ergebnisse (Paper-ready)")
    st.dataframe(df)

    st.download_button(
        "📥 CSV herunterladen",
        df.to_csv(index=False).encode("utf-8"),
        "psr_collagen_quantification.csv"
    )

    st.subheader("🔍 Qualitätskontrolle")
    for r in results:
        st.markdown(f"### {r['Image']}")
        st.image(r["mask"], caption="Gesamt-Kollagenmaske")
        st.image(r["overlay"], caption="Overlay: Rot (I) · Grün (III) · Orange (I+III)")
else:
    st.info("Bitte PSR-Bilder hochladen.")
