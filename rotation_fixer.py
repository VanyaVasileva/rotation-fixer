"""
Rotation Fixer — Sharp Rotation for Illustration Motifs
--------------------------------------------------------
Problem this tool solves:
In Procreate, motifs often become blurry when rotated (simple rotation
algorithms). This tool uses high-quality LANCZOS/BICUBIC resampling so
every motif stays perfectly sharp when rotated.

Workflow:
1) Upload a sheet with one or more motifs (transparent background,
   motifs not overlapping)
2) The app automatically detects each individual motif
3) Each motif gets its own rotation slider (0-360°)
4) Live preview
5) Download as a sharp, transparent PNG sheet

Run locally (in terminal):
    pip install streamlit pillow numpy scipy
    streamlit run rotation_fixer.py
"""

import streamlit as st
from PIL import Image, ImageCms
import numpy as np
from scipy import ndimage
import io

st.set_page_config(page_title="Rotation Fixer", layout="centered")

# ---------------------------------------------------------
# Scandi look, matching the rest of the project
# ---------------------------------------------------------
st.markdown("""
<style>
.stApp { background-color: #FAFAFA; color: #2D2D2D; }
.stButton>button, .stDownloadButton>button {
    background-color: #A2B5A2; color: white; border: none;
    border-radius: 8px; padding: 0.5em 1.2em;
}
.motif-label { font-size: 13px; color: #6b6b66; margin-bottom: -8px; }
</style>
""", unsafe_allow_html=True)

st.title("🌀 Rotation Fixer")
st.caption("Rotate motifs sharply — no more Procreate blur problem")

st.markdown("""
Upload a sheet with one or more motifs (**transparent background,
motifs not overlapping**). A rotation slider will appear for each
detected motif.
""")

uploaded_file = st.file_uploader("Upload sheet (PNG with transparency)", type=["png"])


def normalize_to_srgb(image: Image.Image) -> Image.Image:
    """Converts images with an embedded color profile (e.g. Procreate's Display P3)
    to standard sRGB, so colors don't shift compared to the original artwork."""
    icc = image.info.get("icc_profile")
    if not icc:
        return image
    try:
        input_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        srgb_profile = ImageCms.createProfile("sRGB")
        mode = "RGBA" if image.mode == "RGBA" else "RGB"
        converted = ImageCms.profileToProfile(image, input_profile, srgb_profile, outputMode=mode)
        return converted
    except Exception:
        # If conversion fails for any reason, fall back to the original image
        return image


def detect_motifs(image: Image.Image, dilation_iterations: int = 10, min_size: int = 300):
    """Finds individual motifs on a sheet via connected-components analysis."""
    arr = np.array(image.convert("RGBA"))
    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3]
    is_white = (rgb[:, :, 0] > 245) & (rgb[:, :, 1] > 245) & (rgb[:, :, 2] > 245)
    content = (alpha > 10) & (~is_white)

    dilated = ndimage.binary_dilation(content, iterations=dilation_iterations)
    labeled, n = ndimage.label(dilated, structure=np.ones((3, 3)))
    sizes = ndimage.sum(content, labeled, range(1, n + 1))
    objs = ndimage.find_objects(labeled)

    crops = []
    for i in range(n):
        if sizes[i] < min_size:
            continue
        sl = objs[i]
        padding = 10
        y0 = max(sl[0].start - padding, 0)
        y1 = min(sl[0].stop + padding, image.height)
        x0 = max(sl[1].start - padding, 0)
        x1 = min(sl[1].stop + padding, image.width)
        crop = image.crop((x0, y0, x1, y1))
        crops.append(crop)
    return crops


def rotate_sharp(motif: Image.Image, angle: float) -> Image.Image:
    """Rotates a motif losslessly using BICUBIC resampling (no blur)."""
    return motif.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)


def crop_to_content_local(img: Image.Image, padding: int = 4) -> Image.Image:
    """Crops away the extra transparent canvas that rotation adds,
    so the preview doesn't look smaller just because the bounding box grew."""
    arr = np.array(img)
    alpha = arr[:, :, 3]
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0:
        return img
    x0, x1 = max(xs.min() - padding, 0), min(xs.max() + padding, img.width)
    y0, y1 = max(ys.min() - padding, 0), min(ys.max() + padding, img.height)
    return img.crop((x0, y0, x1, y1))


if uploaded_file:
    original = Image.open(uploaded_file).convert("RGBA")
    original = normalize_to_srgb(original)
    motifs = detect_motifs(original)

    if not motifs:
        st.warning("No motifs detected. Make sure the background is transparent and there's enough contrast.")
    else:
        st.success(f"{len(motifs)} motif(s) detected.")

        rotated_motifs = []
        cols = st.columns(min(3, len(motifs)))
        for idx, motif in enumerate(motifs):
            col = cols[idx % len(cols)]
            with col:
                st.markdown(f'<p class="motif-label">Motif {idx+1}</p>', unsafe_allow_html=True)
                angle = st.slider(
                    f"Angle motif {idx+1}",
                    min_value=0.0, max_value=360.0, value=0.0, step=0.1,
                    key=f"angle_{idx}",
                    label_visibility="collapsed",
                )
                rotated = rotate_sharp(motif, angle)
                preview = crop_to_content_local(rotated)
                st.image(preview, use_container_width=True)
                rotated_motifs.append(rotated)

        st.divider()
        st.subheader("Result sheet")

        # Place all rotated motifs side by side on a new transparent sheet
        padding = 40
        max_h = max(m.height for m in rotated_motifs)
        total_w = sum(m.width for m in rotated_motifs) + padding * (len(rotated_motifs) + 1)
        result_sheet = Image.new("RGBA", (total_w, max_h + padding * 2), (0, 0, 0, 0))

        x_cursor = padding
        for m in rotated_motifs:
            y = (result_sheet.height - m.height) // 2
            result_sheet.paste(m, (x_cursor, y), m)
            x_cursor += m.width + padding

        st.image(result_sheet, caption="All rotated motifs (transparent)", use_container_width=True)

        buf = io.BytesIO()
        result_sheet.save(buf, format="PNG")
        st.download_button(
            "📥 Download rotated sheet as PNG",
            data=buf.getvalue(),
            file_name="rotated_motifs.png",
            mime="image/png",
        )
else:
    st.info("No file uploaded yet.")
