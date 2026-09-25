import sys
import math
from pathlib import Path

import numpy as np
from skimage import io, exposure, img_as_float
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


try:
    import cv2
except ImportError:
    cv2 = None
try:
    import scipy.io
    import scipy.ndimage
    import scipy.special
    import scipy.linalg
except ImportError:
    scipy = None

EPS = np.finfo(float).eps
DEFAULT_IMAGE = Path(__file__).with_name("sample.png")



def rgb2hsi(rgb):
    rgb = img_as_float(rgb)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
    theta = np.arccos(np.clip(num / (den + EPS), -1.0, 1.0))

    H = theta.copy()
    H[b > g] = 2 * np.pi - H[b > g]
    H = H / (2 * np.pi)

    S = 1 - 3 * np.minimum(np.minimum(r, g), b) / (r + g + b + EPS)
    I = (r + g + b) / 3.0
    return np.dstack([H, S, I])


def hsi2rgb(hsi):
    h, s, i = (hsi[..., k].astype(float) for k in range(3))
    theta = h * 2 * np.pi

    achro = s == 0
    m1 = ~achro & (theta < 2 * np.pi / 3)
    m2 = ~achro & (theta >= 2 * np.pi / 3) & (theta < 4 * np.pi / 3)
    m3 = ~achro & (theta >= 4 * np.pi / 3)

    t = theta.copy()
    t[m2] -= 2 * np.pi / 3
    t[m3] -= 4 * np.pi / 3

    with np.errstate(divide="ignore", invalid="ignore"):
        lo = i * (1 - s)
        f = i * (1 + (s * np.cos(t)) / np.cos(np.pi / 3 - t))
        hi = 3 * i - (f + lo)

    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    r[achro] = g[achro] = b[achro] = i[achro]
    b[m1], r[m1], g[m1] = lo[m1], f[m1], hi[m1]
    r[m2], g[m2], b[m2] = lo[m2], f[m2], hi[m2]
    g[m3], b[m3], r[m3] = lo[m3], f[m3], hi[m3]

    out = np.dstack([r, g, b]) * 255.0
    out = np.nan_to_num(out, nan=0.0)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def entropy(img):
    u = np.rint(np.clip(np.nan_to_num(img, nan=0.0), 0, 1) * 255).astype(np.uint8)
    counts = np.bincount(u.ravel(), minlength=256)
    p = counts[counts > 0] / u.size
    return float(-np.sum(p * np.log2(p)))


def clahe(I, clip_limit=0.01, nbins=256):
    return exposure.equalize_adapthist(np.clip(I, 0, 1), clip_limit=clip_limit, nbins=nbins)


def histeq(I):
    return exposure.equalize_hist(np.clip(I, 0, 1))


def _trimf(x, a, b, c):
    """Triangular membership function, numerically matching
    `skfuzzy.trimf(x, [a, b, c])` — reimplemented here so this module stays
    dependency-free (no scikit-fuzzy required)."""
    x = np.asarray(x, dtype=float)
    y = np.ones_like(x)
    idx = x <= b
    y[idx] = (x[idx] - a) / (b - a)
    idx = x > b
    y[idx] = (c - x[idx]) / (c - b)
    y[x <= a] = 0.0
    y[x >= c] = 0.0
    return y


def _bpdfhe_u8(image_u8):
    """Brightness-preserving dynamic fuzzy histogram equalization (MATLAB's
    `fcnBPDFHE`), ported from a user-supplied Python implementation.

    Builds a fuzzy histogram (triangular membership, as in the original),
    finds its local maxima via zero-crossings of the smoothed first
    derivative with a negative second derivative, uses those maxima to split
    the 0..255 range into sub-histograms, equalizes each sub-histogram, and
    re-scales it to a span of the output range proportional to
    span * log10(sub-histogram mass) — brighter/denser regions get more of
    the output range, dimmer/sparser ones less.

    Operates on (and returns) a single-channel uint8 array. One robustness
    tweak versus the original: sub-histogram mass M is floored at 1 whenever
    it is <= 0 (the original only guarded M < 0), to avoid a log10(0) domain
    error on an empty sub-histogram; everything else follows the original
    line for line.
    """
    hist, _ = np.histogram(image_u8, bins=256, range=(0, 256))

    x_qual = np.arange(0, 11, 1)
    membership = _trimf(x_qual, 0, 5, 10)

    fuzzyhist = np.zeros(hist.size + membership.size - 1)
    for counter in range(membership.size):
        fuzzyhist = fuzzyhist + membership[counter] * np.concatenate((
            np.zeros(counter), hist, np.zeros(membership.size - counter - 1)
        ))
    fuzzyhist = fuzzyhist[math.ceil(membership.size / 2):-math.floor(membership.size / 2) + 1]

    deltahist = np.zeros(256)
    delta2hist = np.zeros(256)
    histmax = np.zeros(256, dtype=int)
    for i in range(255):
        deltahist[i] = (fuzzyhist[i + 1] - fuzzyhist[i - 1]) / 2
        delta2hist[i] = fuzzyhist[i + 1] - 2 * fuzzyhist[i] + fuzzyhist[i - 1]
    for i in range(255):
        if deltahist[i + 1] * deltahist[i - 1] < 0 and delta2hist[i] < 0:
            histmax[i] = 1

    parts = np.where(histmax)[0]
    x = np.concatenate(([0], parts, [255]))
    n = x.shape[0]

    span = np.zeros(n)
    M = np.zeros(n)
    factor = np.zeros(n)
    rang = np.zeros(n)

    Msum = np.cumsum(hist)

    for i in range(n):
        span[i - 1] = x[i] - x[i - 1]
        if span[i - 1] < 0:
            span[i - 1] = 1
        M[i - 1] = Msum[x[i]] - Msum[x[i - 1]]
        if M[i - 1] <= 0:
            M[i - 1] = 1  # original only guarded `< 0`; `<= 0` avoids log10(0)
        factor[i - 1] = span[i - 1] * math.log10(M[i - 1])

    factorsum = factor.sum()
    for i in range(n):
        rang[i - 1] = 255 * factor[i - 1] / factorsum if factorsum else 0.0

    start = np.cumsum(rang)
    start = np.concatenate(([0], start)).astype(int)

    y = np.zeros(hist.shape, dtype=float)
    for i in range(start.shape[0]):
        lo, hi = start[i - 1], start[i]
        y[lo:hi] = start[i - 1] + rang[i - 1] / M[i - 1] * np.cumsum(hist[lo:hi])

    return y[image_u8]


def bpdfhe(I):
    """BPDFHE post-process for a float intensity channel in [0, 1]. Converts
    to uint8 (the ported algorithm indexes its output mapping curve by
    integer pixel value), runs `_bpdfhe_u8`, and rescales the result back to
    [0, 1]."""
    u8 = np.rint(np.clip(np.nan_to_num(I, nan=0.0), 0, 1) * 255).astype(np.uint8)
    return np.clip(_bpdfhe_u8(u8) / 255.0, 0, 1)


# --------------------------------------------------------------------------
# Positive / negative membership generators
# --------------------------------------------------------------------------
def _yagers(I, g):
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return 1 - (1 - I ** g) ** (1 / g)


def _sugeno(I, g):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1 - (1 - I) / (1 + g * I)


_NEGAMMA_YAGERS = np.linspace(1.1, 3.0, 20)
_NEGAMMA_SUGENO = np.round(np.arange(-0.10, -0.95 - 1e-9, -0.05), 2)  # -0.10 .. -0.95, 18 values


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------
def pca_fuse(im1, im2):
    """Weighted fusion using the principal-component weighting from bipolar.pca."""
    X = np.vstack([im1.ravel(), im2.ravel()])
    C = np.cov(X)
    eigvals, eigvecs = np.linalg.eigh(C)  # ascending eigenvalues
    if eigvals[0] >= eigvals[1]:
        w = eigvecs[:, 0]
    else:
        w = eigvecs[:, 1]
    w = w / w.sum()
    return w[0] * im1 + w[1] * im2


def fuse(positive, negative, fusion):
    fusion = fusion.lower()
    if fusion in ("sa", "s a"):
        return (positive + negative) / 2.0
    if fusion == "pca":
        return pca_fuse(positive, negative)
    raise ValueError(f"Unknown fusion method: {fusion!r} (expected 'sa' or 'pca')")


# --------------------------------------------------------------------------
# PIQE — no-reference quality metric, ported from a user-supplied Python
# implementation of MATLAB Image Processing Toolbox's `piqe`. Self-contained
# (only needs numpy + opencv); no trained model file required. Lower score
# means better perceived quality.
# --------------------------------------------------------------------------
def _piqe_calculate_mscn(dis_image):
    dis_image = dis_image.astype(np.float32)
    ux = cv2.GaussianBlur(dis_image, (7, 7), 7 / 6)
    ux_sq = ux * ux
    sigma = np.sqrt(np.abs(cv2.GaussianBlur(dis_image ** 2, (7, 7), 7 / 6) - ux_sq))
    return (dis_image - ux) / (1 + sigma)


def _piqe_segment_edge(block_edge, n_segments, block_size, window_size):
    segments = np.zeros((n_segments, window_size))
    for i in range(n_segments):
        segments[i, :] = block_edge[i:window_size]
        if window_size <= (block_size + 1):
            window_size = window_size + 1
    return segments


def _piqe_notice_dist_criterion(block, n_segments, block_size, window_size,
                                 block_impaired_threshold, n):
    top_edge = block[0, :]
    seg_top_edge = _piqe_segment_edge(top_edge, n_segments, block_size, window_size)

    right_side_edge = np.transpose(block[:, n - 1])
    seg_right_side_edge = _piqe_segment_edge(right_side_edge, n_segments, block_size, window_size)

    down_side_edge = block[n - 1, :]
    seg_down_side_edge = _piqe_segment_edge(down_side_edge, n_segments, block_size, window_size)

    left_side_edge = np.transpose(block[:, 0])
    seg_left_side_edge = _piqe_segment_edge(left_side_edge, n_segments, block_size, window_size)

    seg_top_std = np.std(seg_top_edge, axis=1)
    seg_right_std = np.std(seg_right_side_edge, axis=1)
    seg_down_std = np.std(seg_down_side_edge, axis=1)
    seg_left_std = np.std(seg_left_side_edge, axis=1)

    block_impaired = 0
    for seg_index in range(seg_top_edge.shape[0]):
        if (seg_top_std[seg_index] < block_impaired_threshold or
                seg_right_std[seg_index] < block_impaired_threshold or
                seg_down_std[seg_index] < block_impaired_threshold or
                seg_left_std[seg_index] < block_impaired_threshold):
            block_impaired = 1
            break
    return block_impaired


def _piqe_center_sur_dev(block, block_size):
    center1 = int((block_size + 1) / 2) - 1
    center2 = center1 + 1
    center = np.vstack((block[:, center1], block[:, center2]))
    block = np.delete(block, center1, axis=1)
    block = np.delete(block, center1, axis=1)
    center_std = np.std(center)
    surround_std = np.std(block)
    return center_std / surround_std


def _piqe_noise_criterion(block, block_size, block_var):
    block_sigma = np.sqrt(block_var)
    cen_sur_dev = _piqe_center_sur_dev(block, block_size)
    block_beta = abs(block_sigma - cen_sur_dev) / max(block_sigma, cen_sur_dev)
    return block_sigma, block_beta


def piqe(im):
    """Perception-based Image Quality Evaluator, ported line-for-line from a
    user-supplied Python implementation of MATLAB's `piqe`. `im` is a
    single-channel image (uint8, 0..255 scale — the thresholds below assume
    that scale). Lower score = better perceived quality; no reference image
    needed."""
    if cv2 is None:
        raise ImportError(
            "piqe() needs opencv (cv2). Install it with: pip install opencv-python-headless"
        )
    block_size = 16
    activity_threshold = 0.1
    block_impaired_threshold = 0.1
    window_size = 6
    n_segments = block_size - window_size + 1
    nhsa = 0

    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    rows, columns = im.shape
    rows_pad = block_size - (rows % block_size) if rows % block_size else 0
    cols_pad = block_size - (columns % block_size) if columns % block_size else 0
    im = np.pad(im, ((0, rows_pad), (0, cols_pad)), 'edge')

    imnorm = _piqe_calculate_mscn(im)

    block_scores = []
    for i in np.arange(0, imnorm.shape[0] - 1, block_size):
        for j in np.arange(0, imnorm.shape[1] - 1, block_size):
            wndc = 0
            wnc = 0
            block = imnorm[i:i + block_size, j:j + block_size]
            block_var = np.var(block)

            if block_var > activity_threshold:
                nhsa += 1
                block_impaired = _piqe_notice_dist_criterion(
                    block, n_segments, block_size - 1, window_size,
                    block_impaired_threshold, block_size)
                if block_impaired:
                    wndc = 1
                block_sigma, block_beta = _piqe_noise_criterion(block, block_size - 1, block_var)
                if block_sigma > 2 * block_beta:
                    wnc = 1
                s = wndc * pow(1 - block_var, 2) + wnc * pow(block_var, 2)
                if s > 0:
                    block_scores.append(s)

    block_scores = sorted(block_scores)
    low_sum = sum(block_scores[:int(0.1 * len(block_scores))])
    total = sum(block_scores)
    scores = [(s * 10 * low_sum) / total for s in block_scores] if total > 0 else []
    C = 1
    return ((sum(scores) + C) / (C + nhsa)) * 100


# --------------------------------------------------------------------------
# NIQE — no-reference quality metric, ported from a user-supplied Python
# implementation of MATLAB Image Processing Toolbox's `niqe`. Needs a
# pretrained natural-scene-statistics model (`modelparameters.mat`) that is
# NOT bundled here — see NIQE_MODEL_HELP below. Lower score = better
# perceived quality.
# --------------------------------------------------------------------------
_NIQE_MODEL_PATH = Path(__file__).with_name("modelparameters.mat")
NIQE_MODEL_HELP = (
    f"niqe() needs a pretrained natural-scene-statistics model file at "
    f"{_NIQE_MODEL_PATH} (mean/covariance of NIQE features fit on a pristine "
    "image dataset). It isn't bundled with this module since it's third-party "
    "trained data with its own citation/license terms. Get it from the "
    "official NIQE release (Mittal, Soundararajan & Bovik, LIVE Lab, UT "
    "Austin: http://live.ece.utexas.edu/research/quality/niqe_release.zip) "
    "or one of the widely redistributed 'modelparameters.mat' files bundled "
    "with Python NIQE ports on GitHub, and place it next to bfi.py."
)

_niqe_gamma_range = np.arange(0.2, 10, 0.001)
_niqe_a = None  # lazily built the first time niqe() runs (needs scipy.special)


def _niqe_prec_gammas():
    global _niqe_a
    if _niqe_a is None:
        a = scipy.special.gamma(2.0 / _niqe_gamma_range)
        a *= a
        b = scipy.special.gamma(1.0 / _niqe_gamma_range)
        c = scipy.special.gamma(3.0 / _niqe_gamma_range)
        _niqe_a = a / (b * c)
    return _niqe_a


def _niqe_aggd_features(imdata):
    imdata = imdata.reshape(-1)
    imdata2 = imdata * imdata
    left_data = imdata2[imdata < 0]
    right_data = imdata2[imdata >= 0]
    left_mean_sqrt = np.sqrt(np.average(left_data)) if len(left_data) > 0 else 0
    right_mean_sqrt = np.sqrt(np.average(right_data)) if len(right_data) > 0 else 0

    gamma_hat = left_mean_sqrt / right_mean_sqrt if right_mean_sqrt != 0 else np.inf

    r_hat = ((np.average(np.abs(imdata)) ** 2) / np.average(imdata2)
              if np.mean(imdata2) != 0 else np.inf)
    rhat_norm = r_hat * (((gamma_hat ** 3 + 1) * (gamma_hat + 1)) / (gamma_hat ** 2 + 1) ** 2)

    prec_gammas = _niqe_prec_gammas()
    pos = np.argmin((prec_gammas - rhat_norm) ** 2)
    alpha = _niqe_gamma_range[pos]

    gam1 = scipy.special.gamma(1.0 / alpha)
    gam2 = scipy.special.gamma(2.0 / alpha)
    gam3 = scipy.special.gamma(3.0 / alpha)

    aggdratio = np.sqrt(gam1) / np.sqrt(gam3)
    bl = aggdratio * left_mean_sqrt
    br = aggdratio * right_mean_sqrt
    N = (br - bl) * (gam2 / gam1)
    return alpha, N, bl, br, left_mean_sqrt, right_mean_sqrt


def _niqe_paired_product(new_im):
    shift1 = np.roll(new_im, 1, axis=1)
    shift2 = np.roll(new_im, 1, axis=0)
    shift3 = np.roll(np.roll(new_im, 1, axis=0), 1, axis=1)
    shift4 = np.roll(np.roll(new_im, 1, axis=0), -1, axis=1)
    return shift1 * new_im, shift2 * new_im, shift3 * new_im, shift4 * new_im


def _niqe_gauss_window(lw, sigma):
    sd = np.float32(sigma)
    lw = int(lw)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    total = 1.0
    sd *= sd
    for ii in range(1, lw + 1):
        tmp = np.exp(-0.5 * np.float32(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        total += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= total
    return weights


def _niqe_mscn_transform(image, C=1, avg_window=None, extend_mode='constant'):
    if avg_window is None:
        avg_window = _niqe_gauss_window(3, 7.0 / 6.0)
    h, w = np.shape(image)
    mu_image = np.zeros((h, w), dtype=np.float32)
    var_image = np.zeros((h, w), dtype=np.float32)
    image = np.array(image).astype('float32')
    scipy.ndimage.correlate1d(image, avg_window, 0, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(mu_image, avg_window, 1, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(image ** 2, avg_window, 0, var_image, mode=extend_mode)
    scipy.ndimage.correlate1d(var_image, avg_window, 1, var_image, mode=extend_mode)
    var_image = np.sqrt(np.abs(var_image - mu_image ** 2))
    return (image - mu_image) / (var_image + C), var_image, mu_image


def _niqe_subband_feats(mscncoefs):
    alpha_m, N, bl, br, lsq, rsq = _niqe_aggd_features(mscncoefs.copy())
    pps1, pps2, pps3, pps4 = _niqe_paired_product(mscncoefs)
    alpha1, N1, bl1, br1, lsq1, rsq1 = _niqe_aggd_features(pps1)
    alpha2, N2, bl2, br2, lsq2, rsq2 = _niqe_aggd_features(pps2)
    alpha3, N3, bl3, br3, lsq3, rsq3 = _niqe_aggd_features(pps3)
    alpha4, N4, bl4, br4, lsq4, rsq4 = _niqe_aggd_features(pps4)
    return np.array([alpha_m, (bl + br) / 2.0,
                      alpha1, N1, bl1, br1,
                      alpha2, N2, bl2, br2,
                      alpha3, N3, bl3, bl3,
                      alpha4, N4, bl4, bl4])


def _niqe_extract_on_patches(img, patch_size):
    h, w = img.shape
    patch_size = int(patch_size)
    patches = [img[j:j + patch_size, i:i + patch_size]
               for j in range(0, h - patch_size + 1, patch_size)
               for i in range(0, w - patch_size + 1, patch_size)]
    return np.array([_niqe_subband_feats(p) for p in patches])


def _niqe_patches_test_features(img, patch_size):
    h, w = np.shape(img)
    if h < patch_size or w < patch_size:
        raise ValueError("Input image is too small for niqe()")
    hoffset = h % patch_size
    woffset = w % patch_size
    if hoffset > 0:
        img = img[:-hoffset, :]
    if woffset > 0:
        img = img[:, :-woffset]

    img = img.astype(np.float32)
    img2 = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    mscn1, _, _ = _niqe_mscn_transform(img)
    mscn2, _, _ = _niqe_mscn_transform(img2)

    feats_lvl1 = _niqe_extract_on_patches(mscn1.astype(np.float32), patch_size)
    feats_lvl2 = _niqe_extract_on_patches(mscn2.astype(np.float32), patch_size / 2)
    return np.hstack((feats_lvl1, feats_lvl2))


def niqe(input_img_data):
    """Natural Image Quality Evaluator, ported from a user-supplied Python
    implementation of MATLAB's `niqe`. `input_img_data` is a single-channel
    image (uint8, 0..255 scale). Needs `modelparameters.mat` next to this
    module — see NIQE_MODEL_HELP. Lower score = better perceived quality; no
    reference image needed."""
    if cv2 is None or scipy is None:
        raise ImportError(
            "niqe() needs opencv (cv2) and scipy. Install with: "
            "pip install opencv-python-headless scipy"
        )
    if not _NIQE_MODEL_PATH.exists():
        raise FileNotFoundError(NIQE_MODEL_HELP)

    patch_size = 96
    params = scipy.io.loadmat(str(_NIQE_MODEL_PATH))
    pop_mu = np.ravel(params['mu_prisparam'])
    pop_cov = params['cov_prisparam']

    if input_img_data.ndim == 3:
        input_img_data = cv2.cvtColor(input_img_data, cv2.COLOR_BGR2GRAY)
    m, n = input_img_data.shape
    if m <= (patch_size * 2 + 1) or n <= (patch_size * 2 + 1):
        raise ValueError(
            f"niqe() needs an image larger than {patch_size * 2 + 1}x{patch_size * 2 + 1} "
            f"px with the default patch_size={patch_size}; got {m}x{n}."
        )

    feats = _niqe_patches_test_features(input_img_data, patch_size)
    sample_mu = np.mean(feats, axis=0)
    sample_cov = np.cov(feats.T)

    X = sample_mu - pop_mu
    covmat = (pop_cov + sample_cov) / 2.0
    pinvmat = scipy.linalg.pinv(covmat)
    return float(np.sqrt(np.dot(np.dot(X, pinvmat), X)))


# --------------------------------------------------------------------------
# Objective (reference-based quality) functions
# --------------------------------------------------------------------------
def _corr2(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / denom) if denom > 0 else 0.0


LOWER_IS_BETTER_OBJECTIVES = {"piqe", "niqe"}  # distortion scores: lower = better


def _to_u8_2d(x):
    return np.rint(np.clip(np.nan_to_num(x, nan=0.0), 0, 1) * 255).astype(np.uint8)


def objective_score(enhanced, reference, name):
    name = name.lower()
    e = np.clip(np.nan_to_num(enhanced, nan=0.0), 0, 1)
    if name in ("piqe", "niqe"):
        # No-reference metrics: score the enhanced image only.
        u8 = _to_u8_2d(e)
        return float(piqe(u8)) if name == "piqe" else float(niqe(u8))
    r = np.clip(np.nan_to_num(reference, nan=0.0), 0, 1)
    if name in ("psnr",):
        return float(peak_signal_noise_ratio(r, e, data_range=1.0))
    if name in ("r", "corr"):
        return _corr2(e, r)
    if name in ("ssim",):
        return float(structural_similarity(r, e, data_range=1.0))
    raise ValueError(f"Unknown objective: {name!r}")


# --------------------------------------------------------------------------
# Core search (shared by bfi / bfi_psugeno / bfi_nsugeno / bfi_pnsugeno /
# bfi_clahe / bfi_he)
# --------------------------------------------------------------------------
def _bfi_core(A, gamma, fusion, objective, positive_kind, negative_kind,
              post_process=None, history=None):
    hsi = rgb2hsi(A)
    H, S, I = hsi[..., 0], hsi[..., 1], hsi[..., 2]

    positive = _yagers(I, gamma) if positive_kind == "yagers" else _sugeno(I, gamma)

    if negative_kind == "yagers_range":
        negammas = _NEGAMMA_YAGERS
        neg_func = _yagers
    else:
        negammas = _NEGAMMA_SUGENO
        neg_func = _sugeno

    lower_is_better = objective.lower() in LOWER_IS_BETTER_OBJECTIVES
    # matches MATLAB's `obj = 0; if objvalue > obj` for higher-is-better
    # metrics (psnr/corr/ssim); for the no-reference distortion metrics
    # (piqe/niqe, lower = better) the search instead minimises from +inf.
    best_score = np.inf if lower_is_better else 0.0
    final_negamma = None
    final_enhanced = None

    for negamma in negammas:
        negative = neg_func(I, negamma)
        enhanced = fuse(positive, negative, fusion)
        score = objective_score(enhanced, I, objective)

        if history is not None:
            history.append(dict(negamma=float(negamma), score=float(score)))

        better = (score < best_score) if lower_is_better else (score > best_score)
        if better:
            best_score = score
            final_negamma = float(negamma)
            final_enhanced = enhanced

    if final_enhanced is None:
        raise RuntimeError(
            "No candidate scored better than the initial baseline with this objective — "
            "nothing to select (for higher-is-better objectives this mirrors the MATLAB "
            "code's behaviour, which would leave `finalnegamma` undefined in the same "
            "situation)."
        )

    if post_process is not None:
        final_enhanced = post_process(final_enhanced)

    out = hsi.copy()
    out[..., 2] = np.clip(final_enhanced, 0, 1)
    return hsi2rgb(out), final_negamma, best_score


DEFAULT_GAMMA_YAGERS = 0.4   # default positive-membership parameter for the Yager-style generator
DEFAULT_GAMMA_SUGENO = 25    # default positive-membership parameter for the Sugeno-style generator

POSITIVE_KINDS = ("yagers", "sugeno")
NEGATIVE_KINDS = ("yagers", "sugeno")
POST_PROCESS_KINDS = ("none", "clahe", "he", "bpdfhe")


def _resolve_post_process(name, clip_limit=0.01, nbins=256):
    """Map a post-process name to a callable (or None), independent of which
    positive/negative generators were used to build the fused image."""
    name = (name or "none").lower()
    if name == "none":
        return None
    if name == "clahe":
        return lambda x: clahe(x, clip_limit, nbins)
    if name == "he":
        return histeq
    if name == "bpdfhe":
        return bpdfhe
    raise ValueError(f"Unknown post_process: {name!r} (expected one of {POST_PROCESS_KINDS})")


def bfi_general(A, gamma, fusion="sa", objective="psnr", positive="yagers",
                 negative="yagers", post_process="none", history=None,
                 clip_limit=0.01, nbins=256):
    """Same search/fuse/score pipeline as bfi/bfi_psugeno/..., but with the
    positive generator, negative generator, and post-process step chosen
    independently instead of being bundled into a fixed named variant.

    positive/negative: 'yagers' or 'sugeno'
    post_process:       'none', 'clahe', 'he', or 'bpdfhe'
    """
    positive = positive.lower()
    if positive not in POSITIVE_KINDS:
        raise ValueError(f"Unknown positive: {positive!r} (expected one of {POSITIVE_KINDS})")

    negative = negative.lower()
    if negative not in NEGATIVE_KINDS:
        raise ValueError(f"Unknown negative: {negative!r} (expected one of {NEGATIVE_KINDS})")
    negative_kind = "yagers_range" if negative == "yagers" else "sugeno_range"

    pp_fn = _resolve_post_process(post_process, clip_limit, nbins)

    return _bfi_core(A, gamma, fusion, objective, positive, negative_kind,
                      post_process=pp_fn, history=history)


def bfi(A, gamma=DEFAULT_GAMMA_YAGERS, fusion="sa", objective="psnr", history=None):
    return _bfi_core(A, gamma, fusion, objective, "yagers", "yagers_range", history=history)


def bfi_psugeno(A, gamma=DEFAULT_GAMMA_SUGENO, fusion="sa", objective="psnr", history=None):
    return _bfi_core(A, gamma, fusion, objective, "sugeno", "yagers_range", history=history)


def bfi_nsugeno(A, gamma=DEFAULT_GAMMA_YAGERS, fusion="sa", objective="psnr", history=None):
    return _bfi_core(A, gamma, fusion, objective, "yagers", "sugeno_range", history=history)


def bfi_pnsugeno(A, gamma=DEFAULT_GAMMA_SUGENO, fusion="sa", objective="psnr", history=None):
    return _bfi_core(A, gamma, fusion, objective, "sugeno", "sugeno_range", history=history)


def bfi_clahe(A, gamma=DEFAULT_GAMMA_YAGERS, fusion="sa", objective="psnr", history=None,
              clip_limit=0.01, nbins=256):
    return _bfi_core(A, gamma, fusion, objective, "yagers", "yagers_range",
                      post_process=lambda x: clahe(x, clip_limit, nbins), history=history)


def bfi_he(A, gamma=DEFAULT_GAMMA_YAGERS, fusion="sa", objective="psnr", history=None):
    return _bfi_core(A, gamma, fusion, objective, "yagers", "yagers_range",
                      post_process=histeq, history=history)


def bfi_bpdfhe(A, gamma=DEFAULT_GAMMA_YAGERS, fusion="sa", objective="psnr", history=None):
    return _bfi_core(A, gamma, fusion, objective, "yagers", "yagers_range",
                      post_process=bpdfhe, history=history)


# --------------------------------------------------------------------------
# Standalone Yager-style entropy-optimal fuzzy image (bipolar.IFI / IFI_gamma)
# --------------------------------------------------------------------------
def IFI(A):
    """Entropy-optimal Yager-style fuzzy image, gamma searched over 0.1..1.0."""
    M = img_as_float(A)
    gammas = np.linspace(0.1, 1.0, 10)
    best_gamma, best_E, best_img = None, -np.inf, None
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for g in gammas:
            img = _yagers(M, g)
            E = entropy(img)
            if E > best_E:
                best_E, best_gamma, best_img = E, float(g), img
    return best_img, best_gamma


def IFI_gamma(A, gamma):
    M = img_as_float(A)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return _yagers(M, gamma)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
_VARIANTS = {
    "bfi": bfi,
    "psugeno": bfi_psugeno,
    "nsugeno": bfi_nsugeno,
    "pnsugeno": bfi_pnsugeno,
    "clahe": bfi_clahe,
    "he": bfi_he,
    "bpdfhe": bfi_bpdfhe,
}
_SUGENO_POSITIVE_VARIANTS = {"psugeno", "pnsugeno"}  # these use the Sugeno positive generator

if __name__ == "__main__":
    in_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).with_name("output_bipolar.png")
    variant = sys.argv[3] if len(sys.argv) > 3 else "clahe"
    default_gamma = DEFAULT_GAMMA_SUGENO if variant in _SUGENO_POSITIVE_VARIANTS else DEFAULT_GAMMA_YAGERS
    gamma = float(sys.argv[4]) if len(sys.argv) > 4 else default_gamma
    fusion = sys.argv[5] if len(sys.argv) > 5 else "sa"
    objective = sys.argv[6] if len(sys.argv) > 6 else "corr"

    if not in_path.exists():
        sys.exit(
            f"Image not found: {in_path}\n"
            f"Usage: python {Path(__file__).name} [input] [output] [variant] [gamma] [fusion] [objective]\n"
            f"Variants: {', '.join(_VARIANTS)}   Fusion: sa, pca   Objective: psnr, corr, ssim"
        )
    if variant not in _VARIANTS:
        sys.exit(f"Unknown variant {variant!r}. Choose from: {', '.join(_VARIANTS)}")

    A = io.imread(in_path)
    if A.ndim == 3 and A.shape[2] == 4:
        A = A[..., :3]

    output, final_negamma, best_score = _VARIANTS[variant](A, gamma, fusion, objective)

    print(f"input        = {in_path}")
    print(f"variant      = bfi_{variant}" if variant != "bfi" else "variant      = bfi")
    print(f"gamma        = {gamma}")
    print(f"fusion       = {fusion}")
    print(f"objective    = {objective}")
    print(f"final negamma = {final_negamma:.4f}")
    print(f"best score    = {best_score:.6f}")
    io.imsave(out_path, output)
    print(f"saved         = {out_path}")