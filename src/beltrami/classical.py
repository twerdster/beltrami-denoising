"""Finite-difference Beltrami image flows for grayscale and RGB images.

Image values retain their original scale (usually 0..255); do not normalize.
``ws`` is the radius, so the patch contains (2*ws+1)**2 pixels. Derivatives
replicate the edge value; patch sums use zero padding. There is no clipping,
normalization, determinant regularization, or rescaled time in patch flows.

Source families:
  efficient: Beltrami/Beltrami/Code, double precision, determinant**(-0.5).
  pbframework: PBFramework/PB2D/Method_PatchBeltrami, single precision,
               1/sqrt(determinant).

The integral backend preserves the source's accumulation axis order and the
left-associative ``beta + A + D - B - C`` expression. Single-precision summed
area tables can suffer cancellation; choose precision='double' explicitly
for an intentional numerical variant. No stabilization is applied silently.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.signal import convolve2d


def dminus(a: np.ndarray, axis: int) -> np.ndarray:
    """P - P([1 1:end-1],:) (or the analogous column derivative)."""
    out = np.zeros_like(a)
    dst = [slice(None)] * a.ndim
    src = dst.copy()
    dst[axis] = slice(1, None)
    src[axis] = slice(None, -1)
    out[tuple(dst)] = a[tuple(dst)] - a[tuple(src)]
    return out


def dplus(a: np.ndarray, axis: int) -> np.ndarray:
    """P([2:end end],:) - P (or the analogous column derivative)."""
    out = np.zeros_like(a)
    dst = [slice(None)] * a.ndim
    src = dst.copy()
    dst[axis] = slice(None, -1)
    src[axis] = slice(1, None)
    out[tuple(dst)] = a[tuple(src)] - a[tuple(dst)]
    return out


def _channels_sum(a: np.ndarray) -> np.ndarray:
    # Explicit source order avoids a platform-dependent reduction grouping.
    if a.ndim == 2:
        return a
    return (a[:, :, 0] + a[:, :, 1]) + a[:, :, 2]


def _integral_corners(a: np.ndarray, ws: int):
    # MATLAB: cumsum(cumsum(padarray(padarray(I,[ws ws]),[1 1],'pre')),2)
    padded = np.pad(a, ((ws + 1, ws), (ws + 1, ws)), mode="constant")
    sat = np.cumsum(np.cumsum(padded, axis=0, dtype=a.dtype), axis=1, dtype=a.dtype)
    height, width = a.shape
    k = ws * 2 + 1
    return (sat[:height, :width], sat[k:k + height, k:k + width],
            sat[:height, k:k + width], sat[k:k + height, :width])


def box_sum(a: np.ndarray, ws: int, backend: str = "conv2", beta=0) -> np.ndarray:
    """Unnormalized zero-padded box sum, with optional source-order beta."""
    if backend == "integral":
        aa, dd, bb, cc = _integral_corners(a, ws)
        # Do not move beta to the end: that changes single-precision rounding.
        if beta != 0:
            return beta + aa + dd - bb - cc
        return aa + dd - bb - cc
    k = 2 * ws + 1
    if backend == "conv2":
        out = convolve2d(a, np.ones((k, k), dtype=a.dtype), mode="same",
                         boundary="fill", fillvalue=0)
    elif backend == "separable":
        out = convolve2d(a, np.ones((k, 1), dtype=a.dtype), mode="same",
                         boundary="fill", fillvalue=0)
        out = convolve2d(out, np.ones((1, k), dtype=a.dtype), mode="same",
                         boundary="fill", fillvalue=0)
    else:
        raise ValueError("backend must be 'conv2', 'integral', or 'separable'")
    return beta + out if beta != 0 else out


def _advance(image, ix, iy, gm05, g11, g12, g22, dt, *, original=False):
    if image.ndim == 3:
        gm05, g11, g12, g22 = (a[:, :, None] for a in (gm05, g11, g12, g22))
    fx = gm05 * (g22 * ix - g12 * iy)
    fy = gm05 * (-g12 * ix + g11 * iy)
    div = dplus(fx, 0) + dplus(fy, 1)
    if original:
        # Original local belt() stores gm05.*div before multiplying by dt.
        return image + dt * (gm05 * div)
    return image + (dt * gm05) * div


def _validate(image, beta, alpha, iterations, ws):
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
        raise ValueError("image must have shape (rows,cols) or (rows,cols,3)")
    if not image.size or not np.all(np.isfinite(image)):
        raise ValueError("image must be nonempty and finite")
    if not np.isfinite(beta) or beta <= 0 or not np.isfinite(alpha) or alpha < 0:
        raise ValueError("beta must be positive and alpha nonnegative, both finite")
    if int(iterations) != iterations or iterations < 0 or int(ws) != ws or ws < 0:
        raise ValueError("iterations and ws must be nonnegative integers")


def patch_beltrami(
    image, beta: float, alpha: float = 0.21, iterations: int = 50, ws: int = 3,
    *, backend: str = "conv2", variant: str = "pbframework",
    precision: str | None = None, callback: Callable | None = None,
) -> np.ndarray:
    """Run the patch Beltrami explicit finite-difference flow.

    ``callback(iteration, image, diagnostics)`` is optional and receives the
    image after each completed iteration. Invalid/nonfinite determinants
    raise FloatingPointError, preserving evidence instead of regularizing.
    The efficient grayscale 'integral' backend is the intended integral
    formulation: the archived beltBWIntIm.m actually duplicates conv2.
    """
    if variant not in ("efficient", "pbframework"):
        raise ValueError("variant must be 'efficient' or 'pbframework'")
    precision = precision or ("double" if variant == "efficient" else "single")
    if precision not in ("single", "double"):
        raise ValueError("precision must be 'single' or 'double'")
    dtype = np.float32 if precision == "single" else np.float64
    image = np.array(image, dtype=dtype, copy=True)
    _validate(image, beta, alpha, iterations, ws)
    iterations, ws = int(iterations), int(ws)
    beta, alpha = dtype(beta), dtype(alpha)
    dt = alpha * beta
    for iteration in range(1, int(iterations) + 1):
        ix, iy = dminus(image, 0), dminus(image, 1)
        g11 = box_sum(_channels_sum(ix * ix), ws, backend, beta)
        g12 = box_sum(_channels_sum(ix * iy), ws, backend)
        g22 = box_sum(_channels_sum(iy * iy), ws, backend, beta)
        det = g11 * g22 - g12 * g12
        if not np.all(np.isfinite(det)) or np.any(det <= 0):
            raise FloatingPointError(
                f"invalid metric determinant at iteration {iteration}: "
                f"minimum={np.nanmin(det)}, precision={precision}, backend={backend}")
        if variant == "efficient":
            gm05 = np.power(det, dtype(-0.5))
        else:
            gm05 = dtype(1) / np.sqrt(det)
        image = _advance(image, ix, iy, gm05, g11, g12, g22, dt)
        if not np.all(np.isfinite(image)):
            raise FloatingPointError(f"nonfinite image at iteration {iteration}")
        if callback:
            callback(iteration, image, {"min_determinant": float(np.min(det)),
                                       "max_determinant": float(np.max(det))})
    return image


def beltrami_original(image, beta, alpha=0.21, iterations=50,
                      *, variant="pbframework"):
    """Port Method_Beltrami/{beltramiBW,beltramiRGB}.m or original log RGB.

    alpha is intentionally ignored: both source versions fix dt=0.21*beta.
    PBFramework BW runs single precision and clips the final result; RGB
    rounds input to single, computes in double, rounds output to single,
    then clips. variant='log_rgb_2011' uses log2(A+1), double precision,
    per-iteration clipping to [0,8], and final 2**I-1.
    """
    a = np.asarray(image)
    _validate(a, beta, alpha, iterations, 0)
    if variant == "log_rgb_2011":
        if a.ndim != 3 or np.any(a <= -1):
            raise ValueError("log_rgb_2011 requires RGB image values > -1")
        a = np.log2(a.astype(np.float64) + 1)
    elif variant == "pbframework":
        a = a.astype(np.float32)
        if a.ndim == 3:
            a = a.astype(np.float64)
    else:
        raise ValueError("variant must be 'pbframework' or 'log_rgb_2011'")
    # For BW MATLAB casts mixed single/double arithmetic to single.
    scalar = a.dtype.type
    dt = scalar(0.21 * float(beta))
    beta = scalar(beta)
    for iteration in range(1, int(iterations) + 1):
        ix, iy = dminus(a, 0), dminus(a, 1)
        if a.ndim == 3:
            # Separate-channel MATLAB code adds beta before channels.
            g11 = ((beta + ix[:, :, 0] ** 2) + ix[:, :, 1] ** 2) + ix[:, :, 2] ** 2
            g22 = ((beta + iy[:, :, 0] ** 2) + iy[:, :, 1] ** 2) + iy[:, :, 2] ** 2
        else:
            g11 = beta + ix * ix
            g22 = beta + iy * iy
        g12 = _channels_sum(ix * iy)
        det = g11 * g22 - g12 * g12
        if not np.all(np.isfinite(det)) or np.any(det <= 0):
            raise FloatingPointError(f"invalid determinant at iteration {iteration}")
        gm05 = (scalar(1) / np.sqrt(det) if a.ndim == 2
                else np.power(det, scalar(-0.5)))
        a = _advance(a, ix, iy, gm05, g11, g12, g22, dt, original=True)
        if variant == "log_rgb_2011":
            a = np.clip(a, 0, 8)
    if variant == "log_rgb_2011":
        return np.power(2.0, a) - 1
    return np.clip(a.astype(np.float32), 0, 255)


def psnr(reference, estimate, peak=255.0):
    """Original scripts' unrounded, unclipped, all-channel PSNR formula."""
    ref, est = np.asarray(reference, np.float64), np.asarray(estimate, np.float64)
    if ref.shape != est.shape:
        raise ValueError("PSNR images have different shapes")
    mse = np.mean((ref - est) ** 2)
    return float(10 * np.log10(peak ** 2 / mse)) if mse else float("inf")
