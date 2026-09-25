# Adaptive Low-Light Image Enhancement Using Bipolar Fuzzy Set

Python implementation and interactive demo accompanying the article published
in **IEEE Transactions on Fuzzy Systems**.

The method enhances low-light images by:

1. Converting the image from RGB to **HSI** (Hue, Saturation, Intensity) space.
2. Building a **positive** fuzzy image from the intensity channel `I` with a
   fixed parameter γ, using either a Yager-style generator (default γ = 0.4)
   or a Sugeno-style generator (default γ = 25).
3. Searching a range of **negative**-membership parameters (`negamma`) with
   the same or the other generator family, building a candidate negative
   image for each.
4. **Fusing** each positive/negative pair — either a simple average or a
   PCA-weighted combination — and scoring the fusion against the original
   intensity using a reference-based objective (**PSNR**, correlation, or
   **SSIM**). The negamma that scores highest is kept.
5. Optionally applying **CLAHE** or global histogram equalization to the
   final fused intensity.
6. Converting back from HSI to RGB.

Only the intensity channel is modified — hue and saturation are carried
through unchanged.

**Live demo:** https://bfi.streamlit.app

## Article

> **Adaptive Low-Light Image Enhancement Using Bipolar Fuzzy Set**
> Mahesh Kumar, C.V., Nithyanandham, Deva, David Raj, M., Augustin, Felix,
> Ramasamy, Saravanakumar, Saraswathi, D., Zhang, Ye
> *IEEE Transactions on Fuzzy Systems*, Volume 34, Issue 5, pages 1600–1614, 2026
> DOI: [10.1109/TFUZZ.2026.3666002](https://doi.org/10.1109/TFUZZ.2026.3666002)

See [Citation](#citation) below for the full BibTeX entry.

## Repository contents

| File | Description |
|---|---|
| `bfi.py` | Self-contained pipeline: `rgb2hsi`/`hsi2rgb`/`entropy`/`clahe`/`histeq`/`bpdfhe`, the positive/negative generators (Yager- and Sugeno-style), PCA/simple-average fusion, the PSNR/correlation/SSIM objectives, the seven `bfi*` variants, and `bfi_general` (choose positive generator, negative generator, and post-process independently). Also runnable as a script. |
| `appr.py` | Interactive Streamlit demo that visualizes every stage of the pipeline — H, S, I channels, the negamma search curve, the enhanced intensity, and the final image — with independent dropdowns for the positive membership function, negative membership function, and post-process step. |
| `requirements.txt` | Python dependencies. |
| `sample.png` | *(optional)* Bundle your own low-light test image under this name and both scripts will use it as their default input. |

### What's not ported, and why

### About the no-reference objectives (PIQE / NIQE)

- **`piqe`** is fully ported and self-contained (needs `opencv-python` in
  addition to the base requirements) — no external data file required.
- **`niqe`** is ported too, but needs a pretrained natural-scene-statistics
  model file, `modelparameters.mat`, placed next to `bfi.py`.
  

Both are *no-reference* metrics — they score the enhanced image alone, not
against the original — and both are lower-is-better distortion scores, the
opposite sense from PSNR/correlation/SSIM. The demo and the negamma search
handle that automatically when you pick `piqe`/`niqe` as the objective.

## Installation

```bash
git clone https://github.com/maheshkumar-cv/BFI.git
cd BFI
pip install -r requirements.txt
```

## Usage

### Command line

```bash
# Uses sample.png in this folder, the bfi variant, by default
# gamma defaults to 0.4 for Yager-positive variants (bfi, nsugeno, clahe, he)
# and to 25 for Sugeno-positive variants (psugeno, pnsugeno)
python bfi.py

# Or specify everything explicitly
python bfi.py path/to/image.jpg out.png bfi 0.5 sa psnr
python bfi.py path/to/image.jpg out.png psugeno 0.5 pca ssim
```

Prints the fixed γ, the best negamma found, the best objective score, and
saves the enhanced image.

Variants: `bfi`, `psugeno`, `nsugeno`, `pnsugeno`, `clahe`, `he`, `bpdfhe`
Fusion: `sa` (simple average), `pca`
Objective: `psnr`, `corr`, `ssim`

### Interactive demo

```bash
streamlit run app.py
```

Opens a browser tab where you can:
- Upload your own image (or use the bundled `sample.png` / a stock
  fallback image if none is present).
- Choose the variant, fix γ for the positive image, pick the fusion method
  and the scoring objective.
- Inspect the RGB → H, S, I decomposition.
- View the negamma search curve, with the winning candidate marked.
- Compare the original and enhanced intensity channels with linear- and
  log-scale histograms.
- View and download the final enhanced RGB image.

## Using the library in your own code

```python
import bfi as bp
from skimage import io

A = io.imread("your_image.jpg")

enhanced_rgb, best_negamma, best_score = bp.bfi(A, objective="psnr")  # gamma defaults to 0.4

# Or one of the other variants:
enhanced_rgb, best_negamma, best_score = bp.bfi_clahe(A, fusion="pca", objective="ssim")
enhanced_rgb, best_negamma, best_score = bp.bfi_psugeno(A, objective="psnr")  # gamma defaults to 25
```

Available functions: `bfi`, `bfi_psugeno`, `bfi_nsugeno`, `bfi_pnsugeno`,
`bfi_clahe`, `bfi_he`. Each takes `(A, gamma=..., fusion="sa", objective="psnr", history=None)`
and returns `(enhanced_rgb, best_negamma, best_score)`. `gamma` defaults to
`bp.DEFAULT_GAMMA_YAGERS` (0.4) for the Yager-positive variants (`bfi`,
`bfi_nsugeno`, `bfi_clahe`, `bfi_he`) and to `bp.DEFAULT_GAMMA_SUGENO` (25)
for the Sugeno-positive variants (`bfi_psugeno`, `bfi_pnsugeno`).

There's also a standalone entropy-optimal fuzzy-image helper, independent
of the bipolar fusion pipeline:

```python
enhanced, gamma = bp.IFI(A)          # searches gamma in [0.1, 1.0] for max entropy
enhanced = bp.IFI_gamma(A, gamma)    # apply a specific gamma directly
```

## Citation

If you use this code, please cite the article:

```bibtex
@ARTICLE{2026MaheshBipolar,
  author={Mahesh Kumar, C. V. and Nithyanandham, Deva and David Raj, M. and Augustin, Felix and Ramasamy, Saravanakumar and Saraswathi, D. and Zhang, Ye},
  journal={IEEE Transactions on Fuzzy Systems},
  title={Adaptive Low-Light Image Enhancement Using Bipolar Fuzzy Set},
  year={2026},
  volume={34},
  number={5},
  pages={1600-1614},
  doi={10.1109/TFUZZ.2026.3666002}}
```

## Contact

For questions about the code or the article, feel free to email
maheshkumarcv961@gmail.com.
