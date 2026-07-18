from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from PIL import Image

from .matlab import ycbcr_to_pil_rgb

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def list_image_files(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"No image files found under: {root}")
    return files


def y_numpy_to_tensor(y: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(y.astype(np.float32))).unsqueeze(0)


def tensor_to_y_numpy(tensor: torch.Tensor, quantize: bool = False) -> np.ndarray:
    tensor = tensor.detach().clamp(0.0, 1.0).cpu()
    while tensor.dim() > 2:
        tensor = tensor.squeeze(0)
    arr = tensor.numpy().astype(np.float64)
    if quantize:
        arr = np.round(arr * 255.0) / 255.0
    return arr


def shave_np(image: np.ndarray, border: int) -> np.ndarray:
    if border <= 0:
        return image
    if image.shape[0] <= 2 * border or image.shape[1] <= 2 * border:
        raise ValueError(f"Image {image.shape} is too small for border={border}")
    return image[border:-border, border:-border]


def calculate_psnr_y(
    sr_y: np.ndarray,
    hr_y: np.ndarray,
    crop_border: int = 0,
    quantize_to_uint8: bool = True,
) -> float:
    sr = np.asarray(sr_y, dtype=np.float64)
    hr = np.asarray(hr_y, dtype=np.float64)
    if quantize_to_uint8:
        sr = np.round(np.clip(sr, 0.0, 1.0) * 255.0)
        hr = np.round(np.clip(hr, 0.0, 1.0) * 255.0)
        peak = 255.0
    else:
        peak = 1.0
    sr = shave_np(sr, crop_border)
    hr = shave_np(hr, crop_border)
    mse = np.mean((sr - hr) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(peak / np.sqrt(mse))


def calculate_ssim_y(
    sr_y: np.ndarray,
    hr_y: np.ndarray,
    crop_border: int = 0,
    quantize_to_uint8: bool = True,
) -> Optional[float]:
    try:
        from skimage.metrics import structural_similarity
    except Exception:
        return None

    sr = np.asarray(sr_y, dtype=np.float64)
    hr = np.asarray(hr_y, dtype=np.float64)
    if quantize_to_uint8:
        sr = np.round(np.clip(sr, 0.0, 1.0) * 255.0)
        hr = np.round(np.clip(hr, 0.0, 1.0) * 255.0)
        data_range = 255.0
    else:
        data_range = 1.0
    sr = shave_np(sr, crop_border)
    hr = shave_np(hr, crop_border)
    return float(structural_similarity(hr, sr, data_range=data_range))


def merge_y_with_bicubic_chroma(
    sr_y: np.ndarray,
    bicubic_ycbcr: np.ndarray,
) -> Image.Image:
    """Combine SRCNN Y with bicubic Cb/Cr."""
    ycbcr = np.asarray(bicubic_ycbcr, dtype=np.float32).copy()
    if ycbcr.ndim != 3 or ycbcr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 bicubic YCbCr, got {ycbcr.shape}")
    y = np.asarray(sr_y, dtype=np.float32)
    if y.shape != ycbcr.shape[:2]:
        raise ValueError(
            f"Y/chroma shape mismatch: Y={y.shape}, YCbCr={ycbcr.shape}"
        )
    ycbcr[..., 0] = np.clip(y, 0.0, 1.0)
    return ycbcr_to_pil_rgb(ycbcr)


def append_csv_row(path: str | Path, fieldnames: Iterable[str], row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
