import streamlit as st
import cv2
import numpy as np
import pandas as pd

st.set_page_config(page_title="PSR Collagen Quantification", layout="wide")
st.title("📊 PSR Polarized Collagen Quantifier (I / III / I+III)")

# =====================================================
# Sidebar
# =====================================================
st.sidebar.header("Threshold")
mode = st.sidebar.radio("Modus", ["Auto+Offset", "Manuell"])
offset = st.sidebar.slider("Otsu Offset", -40, 40, -10)
manual_thresh = st.sidebar.slider("Manueller Threshold (V)", 0, 255, 110)

sat_min = st.sidebar.slider("Min. Saturation", 0, 255, 30)

st.sidebar.header("Hue-Bereiche (OpenCV HSV)")
red_max = st.sidebar.slider("Rot max", 5, 15, 10)
orange_low = st.sidebar.slider("Orange low", 8, 20, 12)
orange_high = st.sidebar.slider("Orange high", 20, 40, 30)
green_low = st.sidebar.slider("Grün low", 30, 60, 40)
green_high = st.sidebar.slider("Grün high", 60, 120, 90)

uploaded = st.sidebar.file_uploader(
    "PSR Bilder hochladen",
    type=["tif", "tiff", "png", "jpg"],
    accept_multiple_files=True
)

# =====================================================
# Analyse
# =====================================================
def analyze_image(file):

    data = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

    # -------------------------
    # Kollagen-Gesamtmaske
    # -------------------------
    otsu, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = manual_thresh if mode == "Manuell" else np.clip(otsu + offset, 0, 255)

    collagen_mask = ((v > thresh) & (s > sat_min)).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    collagen_mask = cv2.morphologyEx(collagen_mask, cv2.MORPH_OPEN, kernel)
    collagen_mask = cv2.morphologyEx(collagen_mask, cv2.MORPH_CLOSE, kernel)

    cm = collagen_mask.astype(bool)

    # -------------------------
    # Hue-Klassifikation (REAL PSR)
    # -------------------------
    red_mask = (
        ((h >= 0) & (h <= red_max)) |
        ((h >= 170) & (h <= 179))
    ) & cm

    orange_mask = (
        (h >= orange_low) & (h <= orange_high)
    ) & cm

    green_mask = (
        (h >= green_low) & (h <= green_high)
    ) & cm

    # -------------------------
    # Quantifizierung
    # -------------------------
    total = np.sum(cm)
    red = np.sum(red_mask)
    orange = np.sum(orange_mask)
    green = np.sum(green_mask)

    def pct(x): return 100 * x / (total + 1e-6)

    # -------------------------
    # Overlay
    # -------------------------
    overlay = img_rgb.copy()
    overlay[red_mask] = [255, 0, 0]
    overlay[orange_mask] = [255, 165, 0]
    overlay[green_mask] = [0, 255, 0]

    return {
        "Image": file.name,
        "Total Collagen Area (px)": total,
        "Collagen I (red) %": pct(red),
        "Collagen I+III (orange) %": pct(orange),
        "Collagen III (green) %": pct(green),
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

    df = pd.DataFrame(results).drop(columns=["overlay", "mask"])
    st.subheader("📄 Ergebnisse")
    st.dataframe(df)

    st.download_button(
        "📥 CSV herunterladen",
        df.to_csv(index=False).encode("utf-8"),
        "psr_collagen_results.csv"
    )

    st.subheader("🔍 Qualitätskontrolle")
    for r in results:
        st.markdown(f"### {r['Image']}")
        st.image(r["mask"], caption="Gesamt-Kollagen-Maske")
        st.image(
            r["overlay"],
            caption="Overlay: Rot (I) · Orange (I+III) · Grün (III)"
        )
else:
    st.info("Bitte PSR-Bilder hochladen.")
