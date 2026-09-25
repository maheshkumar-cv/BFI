
import inspect
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from skimage import data as skdata
from skimage import img_as_ubyte
from skimage import io as skio
from skimage import transform

import bfi as bp

st.set_page_config(page_title="Bipolar fuzzy set · low-light enhancement", layout="wide")

if not hasattr(bp, "bfi_general") or "history" not in inspect.signature(bp.bfi_general).parameters:
    st.error(
        "The `bipolar_enhance.py` being imported does not have the `bfi_general` "
        "function (positive/negative/post-process chosen independently).\n\n"
        f"Loaded from: `{bp.__file__}`\n\n"
        "Replace it with the latest `bipolar_enhance.py`, then stop Streamlit (Ctrl+C) and run "
        "`streamlit run app_bipolar.py` again so the module is re-imported."
    )
    st.stop()

POSITIVE_OPTIONS = {"yagers": "Yager's", "sugeno": "Sugeno's"}
NEGATIVE_OPTIONS = {"yagers": "Yager's", "sugeno": "Sugeno's"}
POST_PROCESS_OPTIONS = {"none": "None", "clahe": "CLAHE", "he": "Histogram equalization", "bpdfhe": "BPDFHE"}
OBJECTIVES = {"psnr": "PSNR", "corr": "Correlation (r)", "ssim": "SSIM",
              "piqe": "PIQE (no-reference, lower is better)",
              "niqe": "NIQE (no-reference, lower is better)"}


def method_label(positive, negative, post_process):
    label = f"{POSITIVE_OPTIONS[positive]} positive · {NEGATIVE_OPTIONS[negative]} negative"
    if post_process != "none":
        label += f" + {POST_PROCESS_OPTIONS[post_process]}"
    return label


def to_u8(x):
    x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0)
    return np.clip(np.rint(x * 255), 0, 255).astype(np.uint8)


def show(img, caption=None):
    try:
        st.image(img, caption=caption, width="stretch")
    except (TypeError, ValueError):
        st.image(img, caption=caption, use_container_width=True)


DEFAULT_IMAGE = Path(__file__).with_name("sample.png")


def load_image(file_bytes, max_side):
    if file_bytes is None:
        if DEFAULT_IMAGE.exists():
            A = skio.imread(DEFAULT_IMAGE)
            if A.ndim == 2:
                A = np.dstack([A] * 3)
            A = img_as_ubyte(A[..., :3])
        else:
            try:
                A = skdata.astronaut()
                A = to_u8((A / 255.0) ** 2.2)
            except Exception:
                yy, xx = np.mgrid[0:256, 0:256]
                A = np.dstack([xx, yy, (xx + yy) / 2]).astype(float) / 255.0
                A = to_u8(A ** 2.5)
    else:
        A = skio.imread(io.BytesIO(file_bytes))
        if A.ndim == 2:
            A = np.dstack([A] * 3)
        A = img_as_ubyte(A[..., :3])
    h, w = A.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        A = img_as_ubyte(transform.resize(A, (int(h * s), int(w * s)), anti_aliasing=True))
    return A


@st.cache_data(show_spinner=False)
def run_pipeline(A, positive, negative, post_process, gamma, fusion, objective):
    hsi = bp.rgb2hsi(A)
    H, S, I = hsi[..., 0], hsi[..., 1], hsi[..., 2]

    history = []
    out, final_negamma, best_score = bp.bfi_general(
        A, gamma, fusion, objective,
        positive=positive, negative=negative, post_process=post_process,
        history=history,
    )

    return dict(H=H, S=S, I=I, gamma=gamma, negamma=final_negamma, score=best_score,
                history=history, out=out)


def hist_fig(arrs, labels, title):
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.8))
    x = np.linspace(0, 1, 256)
    for a, lab in zip(arrs, labels):
        h, _ = np.histogram(np.clip(a, 0, 1), bins=256, range=(0, 1))
        frac = h / h.sum()
        axes[0].plot(x, frac, label=lab, lw=1.2)
        axes[1].plot(x, np.where(h > 0, frac, np.nan), label=lab, lw=1.2)
    axes[1].set_yscale("log")
    axes[0].set_title(f"{title} – linear", fontsize=10)
    axes[1].set_title(f"{title} – log scale", fontsize=10)
    axes[0].set_ylabel("fraction of pixels")
    axes[1].set_ylabel("fraction of pixels (log)")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.set_xlabel("intensity")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# sidebar
# ----------------------------------------------------------------------------
st.title("Adaptive low-light enhancement using a bipolar fuzzy set")
st.caption("RGB → HSI → positive image (fixed γ) fused with a searched negative image → HSI → RGB")

with st.sidebar:
    st.header("Input")
    up = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
    max_side = st.slider("Max image side (px)", 64, 1024, 384, 32)

    st.header("Method")
    positive = st.selectbox("Positive membership function", list(POSITIVE_OPTIONS),
                             format_func=POSITIVE_OPTIONS.get)
    negative = st.selectbox("Negative membership function", list(NEGATIVE_OPTIONS),
                             format_func=NEGATIVE_OPTIONS.get)
    post_process = st.selectbox("Post-process", list(POST_PROCESS_OPTIONS),
                                 format_func=POST_PROCESS_OPTIONS.get)

    if positive == "sugeno":
        gamma = st.slider("Positive γ (fixed, Sugeno PMI)", 1.0, 50.0, float(bp.DEFAULT_GAMMA_SUGENO), 1.0)
    else:
        gamma = st.slider("Positive γ (fixed, Yager's PMI)", 0.1, 1.5, float(bp.DEFAULT_GAMMA_YAGERS), 0.05)
    fusion = st.radio("Fusion", ["sa", "pca"], horizontal=True,
                       format_func=lambda k: "Simple average" if k == "sa" else "PCA-weighted")
    objective = st.selectbox("Objective (scores each negamma candidate)",
                              list(OBJECTIVES), format_func=OBJECTIVES.get)
    if objective == "niqe":
        st.caption(
            "ℹ️ NIQE needs a pretrained `modelparameters.mat` file next to "
            "`bipolar_enhance.py` — see `bp.NIQE_MODEL_HELP` if it's missing."
        )
    if objective in ("piqe", "niqe"):
        st.caption("PIQE/NIQE score the enhanced image alone (no reference needed) — the search picks the *lowest*-scoring candidate.")

A = load_image(up.getvalue() if up else None, max_side)
if up is None:
    src = "the bundled sample image" if DEFAULT_IMAGE.exists() else "a darkened stock image"
    st.info(f"No file uploaded – showing {src}. Upload your own in the sidebar.")

try:
    with st.spinner("Searching negamma candidates…"):
        R = run_pipeline(A, positive, negative, post_process, gamma, fusion, objective)
except (NotImplementedError, FileNotFoundError, ImportError, ValueError, RuntimeError) as e:
    st.error(str(e))
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Fixed γ (positive)", f"{R['gamma']:.3f}")
c2.metric("Best negamma", f"{R['negamma']:.4f}")
c3.metric(f"Best {OBJECTIVES[objective]}", f"{R['score']:.4f}")
c4.metric("Candidates searched", len(R["history"]))

tab1, tab2, tab3, tab4 = st.tabs([
    "1 · RGB → H, S, I",
    "2 · Negamma search",
    "3 · Enhanced intensity",
    "4 · Final RGB",
])

# ----------------------------------------------------------------------------
# tab 1
# ----------------------------------------------------------------------------
with tab1:
    cols = st.columns(4)
    items = [("Original RGB", A), ("H – hue", to_u8(R["H"])),
             ("S – saturation", to_u8(R["S"])), ("I – intensity", to_u8(R["I"]))]
    for col, (name, img) in zip(cols, items):
        with col:
            show(img, name)
    st.write("")
    stats = pd.DataFrame({
        "channel": ["H", "S", "I"],
        "min": [R[k].min() for k in "HSI"],
        "max": [R[k].max() for k in "HSI"],
        "mean": [R[k].mean() for k in "HSI"],
        "entropy": [bp.entropy(R[k]) for k in "HSI"],
    })
    st.dataframe(stats, hide_index=True)

# ----------------------------------------------------------------------------
# tab 2
# ----------------------------------------------------------------------------
with tab2:
    hist = R["history"]
    st.subheader(f"Score vs. negamma  ({method_label(positive, negative, post_process)}, {fusion} fusion)")
    df = pd.DataFrame(hist)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(df["negamma"], df["score"], "o-")
    best_idx = df["score"].idxmin() if objective in bp.LOWER_IS_BETTER_OBJECTIVES else df["score"].idxmax()
    ax.plot(df["negamma"][best_idx], df["score"][best_idx], "r*", ms=16, label="best")
    ax.set_xlabel("negamma")
    ax.set_ylabel(OBJECTIVES[objective])
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(df.rename(columns={"negamma": "negamma", "score": OBJECTIVES[objective]}),
                 hide_index=True)

# ----------------------------------------------------------------------------
# tab 3
# ----------------------------------------------------------------------------
with tab3:
    out_hsi = bp.rgb2hsi(R["out"])
    enhanced_I = out_hsi[..., 2]
    cols = st.columns(2)
    steps = [("I – original", R["I"]), (f"Enhanced I  (negamma={R['negamma']:.4f})", enhanced_I)]
    for col, (name, arr) in zip(cols, steps):
        with col:
            show(to_u8(arr), f"{name}  (E={bp.entropy(arr):.4f})")
    fig = hist_fig([R["I"], enhanced_I], ["I", "Enhanced I"], "Intensity histograms")
    st.pyplot(fig)
    plt.close(fig)

# ----------------------------------------------------------------------------
# tab 4
# ----------------------------------------------------------------------------
with tab4:
    a, b = st.columns(2)
    with a:
        show(A, "Original")
    with b:
        show(R["out"], f"Enhanced ({method_label(positive, negative, post_process)})")
    buf = io.BytesIO()
    Image.fromarray(R["out"]).save(buf, format="PNG")
    st.download_button("Download enhanced PNG", buf.getvalue(), "enhanced_bipolar.png", "image/png")