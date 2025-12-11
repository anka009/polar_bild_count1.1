import streamlit as st
import cv2
import numpy as np
import pandas as pd
from io import BytesIO
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt

st.set_page_config(page_title="Polarized Collagen Quantification — Debug", layout="wide")
st.title("📊 Polarized Collagen Batch‑Quantifier (Sirius‑Red + Polarized Light)")

# -------------------------
# Sidebar / Controls
# -------------------------
st.sidebar.header("Analyse‑Einstellungen")
hue_thick_low = st.sidebar.slider("Dicke Fasern: Hue LOW", 0, 30, 25)
hue_thick_high = st.sidebar.slider("Dicke Fasern: Hue HIGH", 150, 180, 160)
hue_thin_low = st.sidebar.slider("Dünne Fasern: Hue LOW", 30, 90, 40)
hue_thin_high = st.sidebar.slider("Dünne Fasern: Hue HIGH", 60, 120, 90)

st.sidebar.header("Debug & Output")
show_overlays = st.sidebar.checkbox("Overlay anzeigen (Rot=dick, Grün=dünn)", value=True)
show_hue_hist = st.sidebar.checkbox("Hue‑Histogramm anzeigen", value=True)
show_masks = st.sidebar.checkbox("Masken anzeigen (birefringence / thick / thin)", value=True)
export_overlays = st.sidebar.checkbox("Overlay als PNG zum Download bereitstellen", value=True)
export_masks = st.sidebar.checkbox("Masken als PNG zum Download bereitstellen", value=True)

pixel_size = st.sidebar.number_input("Pixelgröße (µm) — optional, für µm² Berechnung (0 = aus)", min_value=0.0, value=0.0, step=0.01)

uploaded = st.sidebar.file_uploader("Bilder hochladen (TIFF/JPG/PNG)", accept_multiple_files=True)

# -------------------------
# Helper functions
# -------------------------

def _img_to_bytes(img_rgb):
    # img_rgb: uint8 RGB
    ok, buf = cv2.imencode('.png', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    return buf.tobytes()


def _mask_to_bytes(mask):
    # mask: binary or boolean
    if mask.dtype != np.uint8:
        mask_img = (mask > 0).astype(np.uint8) * 255
    else:
        mask_img = mask
    ok, buf = cv2.imencode('.png', mask_img)
    return buf.tobytes()


def analyze_image(file, fname):
    # read
    data = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Unable to read image {fname}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # HSV / brightness
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    brightness = hsv[:, :, 2]

    # 1) Birefringenz mask (Otsu on V)
    _, mask = cv2.threshold(brightness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 2) Hue classification
    h = hsv[:, :, 0]
    thick = ((h < hue_thick_low) | (h > hue_thick_high))
    thin = ((h > hue_thin_low) & (h < hue_thin_high))

    thick_mask = np.logical_and(thick, mask_clean > 0)
    thin_mask = np.logical_and(thin, mask_clean > 0)

    # measurements
    collagen_pixels = int(np.sum(mask_clean > 0))
    total_pixels = int(mask_clean.size)
    area_percent = float(collagen_pixels / total_pixels * 100)

    mean_intensity = float(np.mean(brightness[mask_clean > 0])) if collagen_pixels > 0 else 0.0

    thick_pixels = int(np.sum(thick_mask))
    thin_pixels = int(np.sum(thin_mask))
    ratio = float((thick_pixels + 1e-9) / (thin_pixels + 1e-9))

    # optionally convert to µm²
    if pixel_size > 0:
        collagen_um2 = collagen_pixels * (pixel_size ** 2)
        thick_um2 = thick_pixels * (pixel_size ** 2)
        thin_um2 = thin_pixels * (pixel_size ** 2)
    else:
        collagen_um2 = thick_um2 = thin_um2 = None

    # overlays / debug assets
    debug = {}
    if show_overlays:
        overlay = img_rgb.copy()
        overlay[thick_mask] = [255, 0, 0]
        overlay[thin_mask] = [0, 255, 0]
        debug['overlay_rgb'] = overlay
        if export_overlays:
            debug['overlay_png'] = _img_to_bytes(overlay)

    if show_masks:
        debug['mask_clean'] = (mask_clean > 0).astype(np.uint8) * 255
        debug['thick_mask'] = (thick_mask > 0).astype(np.uint8) * 255
        debug['thin_mask'] = (thin_mask > 0).astype(np.uint8) * 255
        if export_masks:
            debug['mask_clean_png'] = _mask_to_bytes(debug['mask_clean'])
            debug['thick_mask_png'] = _mask_to_bytes(debug['thick_mask'])
            debug['thin_mask_png'] = _mask_to_bytes(debug['thin_mask'])

    if show_hue_hist:
        hist, bins = np.histogram(h.flatten(), bins=180, range=(0, 180))
        debug['hue_hist'] = (hist, bins)

    return {
        'image': fname,
        'collagen_pixels': collagen_pixels,
        'area_percent': area_percent,
        'mean_intensity': mean_intensity,
        'thick_pixels': thick_pixels,
        'thin_pixels': thin_pixels,
        'ratio_thick_thin': ratio,
        'collagen_um2': collagen_um2,
        'thick_um2': thick_um2,
        'thin_um2': thin_um2,
        'debug': debug
    }

# -------------------------
# Main
# -------------------------

if uploaded:
    results = []
    for f in uploaded:
        try:
            res = analyze_image(f, f.name)
            results.append(res)
        except Exception as e:
            st.error(f"Fehler bei {f.name}: {e}")

    if len(results) == 0:
        st.warning("Keine validen Ergebnisse — bitte Bilder prüfen.")
    else:
        df = pd.DataFrame([{k: v for k, v in r.items() if k != 'debug'} for r in results])
        st.subheader("📄 Ergebnisse")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 CSV herunterladen", csv, "collagen_results.csv", "text/csv")

        # Debug panels per image
        for r in results:
            st.markdown("---")
            st.header(f"🔍 Debug — {r['image']}")
            dbg = r.get('debug', {})

            cols = st.columns(3)
            with cols[0]:
                if 'overlay_rgb' in dbg and show_overlays:
                    st.image(dbg['overlay_rgb'], caption='Overlay (Rot=dick, Grün=dünn)')
                    if export_overlays and 'overlay_png' in dbg:
                        st.download_button(f"Download Overlay — {r['image']}", dbg['overlay_png'], f"overlay_{r['image']}.png")
                elif show_overlays:
                    st.info('Overlay nicht verfügbar')

            with cols[1]:
                if show_masks:
                    st.image(dbg['mask_clean'], caption='Kollagenmaske (birefringence)')
                    st.image(dbg['thick_mask'], caption='Dicke Fasern (mask)')
                    st.image(dbg['thin_mask'], caption='Dünne Fasern (mask)')
                    if export_masks and 'mask_clean_png' in dbg:
                        st.download_button(f"Download Masks — {r['image']}", dbg['mask_clean_png'], f"masks_{r['image']}.zip")
                else:
                    st.info('Masken unterdrückt')

            with cols[2]:
                if 'hue_hist' in dbg and show_hue_hist:
                    hist, bins = dbg['hue_hist']
                    fig, ax = plt.subplots()
                    ax.plot(bins[:-1], hist)
                    ax.set_xlabel('Hue (0-180)')
                    ax.set_ylabel('Count')
                    ax.set_title('Hue Histogramm')
                    st.pyplot(fig)
                else:
                    st.info('Hue histogramm unterdrückt')

else:
    st.info("Bitte Bilder hochladen, um die Analyse zu starten.")
