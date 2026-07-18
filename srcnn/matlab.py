"""MATLAB-compatible color conversion and bicubic resizing utilities.

The original SRCNN MATLAB pipeline converts uint8 RGB images with
``rgb2ycbcr`` and then applies ``im2double``.  The functions below reproduce
that limited-range YCbCr convention consistently for Y, Cb, and Cr.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from PIL import Image


def _rgb_unit_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    else:
        rgb = np.asarray(image)
        if rgb.dtype == np.uint8:
            rgb = rgb.astype(np.float64) / 255.0
        else:
            rgb = rgb.astype(np.float64)
            if rgb.max(initial=0.0) > 1.0:
                rgb /= 255.0
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"Expected HxWx3 RGB image, got shape {rgb.shape}")
    return np.clip(rgb[..., :3], 0.0, 1.0)


def rgb_to_ycbcr_matlab(image: Image.Image | np.ndarray) -> np.ndarray:
    """Convert RGB to MATLAB-style limited-range YCbCr in [0, 1].

    The conversion follows ``rgb2ycbcr(uint8_rgb)`` followed by ``im2double``.
    Values are rounded to uint8-equivalent digital levels before division by
    255, matching the original SRCNN MATLAB data preparation.
    """
    rgb = _rgb_unit_array(image)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    y = 16.0 + 65.481 * r + 128.553 * g + 24.966 * b
    cb = 128.0 - 37.797 * r - 74.203 * g + 112.000 * b
    cr = 128.0 + 112.000 * r - 93.786 * g - 18.214 * b

    ycbcr_8bit = np.stack([y, cb, cr], axis=-1)
    ycbcr_8bit = np.round(np.clip(ycbcr_8bit, 0.0, 255.0))
    return (ycbcr_8bit / 255.0).astype(np.float32)


def rgb_to_y_matlab(image: Image.Image | np.ndarray) -> np.ndarray:
    """Return MATLAB-style limited-range luminance Y in [0, 1]."""
    if isinstance(image, np.ndarray) and image.ndim == 2:
        gray = image.astype(np.float32)
        if gray.max(initial=0.0) > 1.0:
            gray /= 255.0
        return np.clip(gray, 0.0, 1.0)
    return rgb_to_ycbcr_matlab(image)[..., 0]


def ycbcr_to_rgb_matlab(ycbcr: np.ndarray) -> np.ndarray:
    """Convert MATLAB-style limited-range YCbCr in [0, 1] to uint8 RGB."""
    array = np.asarray(ycbcr, dtype=np.float64)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 YCbCr image, got shape {array.shape}")

    values = np.clip(array, 0.0, 1.0) * 255.0
    y = values[..., 0] - 16.0
    cb = values[..., 1] - 128.0
    cr = values[..., 2] - 128.0

    r = 1.164383 * y + 1.596027 * cr
    g = 1.164383 * y - 0.391762 * cb - 0.812968 * cr
    b = 1.164383 * y + 2.017232 * cb

    rgb = np.stack([r, g, b], axis=-1)
    return np.round(np.clip(rgb, 0.0, 255.0)).astype(np.uint8)


def ycbcr_to_pil_rgb(ycbcr: np.ndarray) -> Image.Image:
    """Convert a MATLAB-style YCbCr array to a PIL RGB image."""
    return Image.fromarray(ycbcr_to_rgb_matlab(ycbcr), mode="RGB")


def modcrop_np(image: np.ndarray, scale: int) -> np.ndarray:
    """Crop bottom/right edges so H and W are divisible by ``scale``."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    h, w = image.shape[:2]
    cropped_h = h - (h % scale)
    cropped_w = w - (w % scale)
    if cropped_h <= 0 or cropped_w <= 0:
        raise ValueError(f"Image shape {image.shape} is too small for scale={scale}")
    return image[:cropped_h, :cropped_w, ...]


def _cubic(x: np.ndarray) -> np.ndarray:
    absx = np.abs(x)
    absx2 = absx * absx
    absx3 = absx2 * absx
    return (
        (1.5 * absx3 - 2.5 * absx2 + 1.0) * (absx <= 1.0)
        + (-0.5 * absx3 + 2.5 * absx2 - 4.0 * absx + 2.0)
        * ((absx > 1.0) & (absx <= 2.0))
    )


def _weights_indices(
    in_length: int,
    out_length: int,
    scale: float,
    kernel_width: float = 4.0,
    antialiasing: bool = True,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    if scale < 1.0 and antialiasing:
        kernel_width = kernel_width / scale

    x = np.arange(1, out_length + 1, dtype=np.float64)
    u = x / scale + 0.5 * (1.0 - 1.0 / scale)
    left = np.floor(u - kernel_width / 2.0)
    p = int(math.ceil(kernel_width)) + 2
    indices = left[:, None] + np.arange(p, dtype=np.float64)[None, :]
    distance = u[:, None] - indices

    if scale < 1.0 and antialiasing:
        weights = scale * _cubic(distance * scale)
    else:
        weights = _cubic(distance)

    weights_sum = np.sum(weights, axis=1, keepdims=True)
    weights = weights / weights_sum

    if np.all(np.abs(weights[:, 0]) < 1e-12):
        weights = weights[:, 1:]
        indices = indices[:, 1:]
    if np.all(np.abs(weights[:, -1]) < 1e-12):
        weights = weights[:, :-1]
        indices = indices[:, :-1]

    sym_len_start = int(max(0, -indices.min() + 1))
    sym_len_end = int(max(0, indices.max() - in_length))
    indices = indices + sym_len_start - 1
    return weights.astype(np.float64), indices.astype(np.int64), sym_len_start, sym_len_end


def imresize_matlab(
    image: np.ndarray,
    output_shape: tuple[int, int],
    antialiasing: bool = True,
) -> np.ndarray:
    """Resize a 2-D float image with MATLAB-like bicubic interpolation."""
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D image, got shape {image.shape}")

    in_h, in_w = image.shape
    out_h, out_w = int(output_shape[0]), int(output_shape[1])
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"Invalid output shape: {output_shape}")
    if (in_h, in_w) == (out_h, out_w):
        return image.astype(np.float32, copy=True)

    scale_h = out_h / in_h
    scale_w = out_w / in_w

    weights_h, indices_h, sym_hs, sym_he = _weights_indices(
        in_h, out_h, scale_h, antialiasing=antialiasing
    )
    weights_w, indices_w, sym_ws, sym_we = _weights_indices(
        in_w, out_w, scale_w, antialiasing=antialiasing
    )

    image_aug = np.pad(image, ((sym_hs, sym_he), (0, 0)), mode="symmetric")
    tmp = np.empty((out_h, in_w), dtype=np.float64)
    for i in range(out_h):
        tmp[i, :] = np.sum(
            image_aug[indices_h[i], :] * weights_h[i, :, None], axis=0
        )

    tmp_aug = np.pad(tmp, ((0, 0), (sym_ws, sym_we)), mode="symmetric")
    out = np.empty((out_h, out_w), dtype=np.float64)
    for j in range(out_w):
        out[:, j] = np.sum(
            tmp_aug[:, indices_w[j]] * weights_w[j, :][None, :], axis=1
        )

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def imresize_ycbcr_matlab(
    ycbcr: np.ndarray,
    output_shape: tuple[int, int],
    antialiasing: bool = True,
) -> np.ndarray:
    """Resize all three MATLAB-style YCbCr channels consistently."""
    array = np.asarray(ycbcr, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 YCbCr image, got shape {array.shape}")
    channels = [
        imresize_matlab(array[..., channel], output_shape, antialiasing)
        for channel in range(3)
    ]
    return np.stack(channels, axis=-1).astype(np.float32)


def bicubic_degrade_matlab(hr_y: np.ndarray, scale: int) -> np.ndarray:
    """Downsample Y by ``scale`` and bicubic-upsample to its original size."""
    hr_y = modcrop_np(np.asarray(hr_y, dtype=np.float32), scale)
    h, w = hr_y.shape
    lr = imresize_matlab(hr_y, (h // scale, w // scale), antialiasing=True)
    return imresize_matlab(lr, (h, w), antialiasing=False)


def bicubic_degrade_ycbcr_matlab(
    hr_ycbcr: np.ndarray,
    scale: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return true-LR and bicubic-upsampled MATLAB-style YCbCr arrays."""
    hr_ycbcr = modcrop_np(np.asarray(hr_ycbcr, dtype=np.float32), scale)
    h, w = hr_ycbcr.shape[:2]
    lr = imresize_ycbcr_matlab(
        hr_ycbcr,
        (h // scale, w // scale),
        antialiasing=True,
    )
    bicubic = imresize_ycbcr_matlab(lr, (h, w), antialiasing=False)
    return lr, bicubic
