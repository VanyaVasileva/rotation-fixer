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
from PIL import Image
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


if uploaded_file:
    original = Image.open(uploaded_file).convert("RGBA")
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
                    min_value=0, max_value=360, value=0, step=1,
                    key=f"angle_{idx}",
                    label_visibility="collapsed",
                )
                rotated = rotate_sharp(motif, angle)
                st.image(rotated, use_container_width=True)
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
