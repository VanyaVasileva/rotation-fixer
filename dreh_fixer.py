"""
Dreh-Fixer — Scharfes Drehen von Illustrations-Motiven
--------------------------------------------------------
Problem, das dieses Tool löst:
In Procreate werden Motive beim Drehen oft unscharf/blurry (einfache
Rotations-Algorithmen). Dieses Tool nutzt hochwertiges LANCZOS/BICUBIC-
Resampling, damit jedes Motiv beim Drehen absolut scharf bleibt.

Workflow:
1) Sheet mit mehreren Motiven hochladen (transparenter Hintergrund,
   Motive nicht überlappend)
2) Die App erkennt jedes einzelne Motiv automatisch
3) Für jedes Motiv gibt es einen Dreh-Regler (0-360°)
4) Live-Vorschau
5) Download als scharfes, transparentes PNG-Sheet

Starten (lokal, im Terminal):
    pip install streamlit pillow numpy scipy
    streamlit run dreh_fixer.py
"""

import streamlit as st
from PIL import Image
import numpy as np
from scipy import ndimage
import io

st.set_page_config(page_title="Dreh-Fixer", layout="centered")

# ---------------------------------------------------------
# Scandi-Look, passend zum Rest des Projekts
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

st.title("🌀 Dreh-Fixer")
st.caption("Motive scharf drehen — ohne das Procreate-Unschärfe-Problem")

st.markdown("""
Lade ein Sheet mit einem oder mehreren Motiven hoch (**transparenter Hintergrund,
Motive nicht überlappend**). Für jedes erkannte Motiv erscheint ein Dreh-Regler.
""")

uploaded_file = st.file_uploader("Sheet hochladen (PNG mit Transparenz)", type=["png"])


def detect_motifs(image: Image.Image, dilation_iterations: int = 10, min_size: int = 300):
    """Findet einzelne Motive auf einem Sheet per Connected-Components-Analyse."""
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
    """Dreht ein Motiv verlustarm mit BICUBIC-Resampling (keine Unschärfe)."""
    return motif.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)


if uploaded_file:
    original = Image.open(uploaded_file).convert("RGBA")
    motifs = detect_motifs(original)

    if not motifs:
        st.warning("Keine Motive erkannt. Achte auf transparenten Hintergrund und ausreichenden Kontrast.")
    else:
        st.success(f"{len(motifs)} Motiv(e) erkannt.")

        rotated_motifs = []
        cols = st.columns(min(3, len(motifs)))
        for idx, motif in enumerate(motifs):
            col = cols[idx % len(cols)]
            with col:
                st.markdown(f'<p class="motif-label">Motiv {idx+1}</p>', unsafe_allow_html=True)
                angle = st.slider(
                    f"Winkel Motiv {idx+1}",
                    min_value=0, max_value=360, value=0, step=1,
                    key=f"angle_{idx}",
                    label_visibility="collapsed",
                )
                rotated = rotate_sharp(motif, angle)
                st.image(rotated, use_container_width=True)
                rotated_motifs.append(rotated)

        st.divider()
        st.subheader("Ergebnis-Sheet")

        # Alle gedrehten Motive nebeneinander auf ein neues, transparentes Sheet legen
        padding = 40
        max_h = max(m.height for m in rotated_motifs)
        total_w = sum(m.width for m in rotated_motifs) + padding * (len(rotated_motifs) + 1)
        result_sheet = Image.new("RGBA", (total_w, max_h + padding * 2), (0, 0, 0, 0))

        x_cursor = padding
        for m in rotated_motifs:
            y = (result_sheet.height - m.height) // 2
            result_sheet.paste(m, (x_cursor, y), m)
            x_cursor += m.width + padding

        st.image(result_sheet, caption="Alle gedrehten Motive (transparent)", use_container_width=True)

        buf = io.BytesIO()
        result_sheet.save(buf, format="PNG")
        st.download_button(
            "📥 Gedrehtes Sheet als PNG herunterladen",
            data=buf.getvalue(),
            file_name="gedrehte_motive.png",
            mime="image/png",
        )
else:
    st.info("Noch keine Datei hochgeladen.")
