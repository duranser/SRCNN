from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .matlab import bicubic_degrade_matlab, modcrop_np, rgb_to_y_matlab
from .utils import list_image_files


class SRCNNRandomPatchDataset(Dataset):
    """Draw random aligned SRCNN training patches from complete image pairs.

    ``repeat`` has the same practical meaning as in the companion FSRCNN
    project: the effective number of samples per epoch is
    ``number_of_images * repeat``.  Each access chooses a fresh random crop.

    The complete HR image is converted to MATLAB-style Y and degraded before
    patch extraction, keeping training degradation consistent with evaluation.
    """

    def __init__(
        self,
        image_dir: str | Path,
        scale: int = 2,
        input_size: int = 33,
        repeat: int = 100,
        f1: int = 9,
        f2: int = 1,
        f3: int = 5,
        augment: bool = False,
    ) -> None:
        self.paths = list_image_files(image_dir)
        self.scale = int(scale)
        self.input_size = int(input_size)
        self.repeat = max(1, int(repeat))
        self.augment = bool(augment)

        self.output_size = self.input_size - (f1 - 1) - (f2 - 1) - (f3 - 1)
        if self.output_size <= 0:
            raise ValueError(
                f"input_size={input_size} is too small for filters {f1}-{f2}-{f3}"
            )
        difference = self.input_size - self.output_size
        if difference % 2 != 0:
            raise ValueError("Input/label size difference must be even")
        self.label_border = difference // 2

        self.hr_images: list[np.ndarray] = []
        self.input_images: list[np.ndarray] = []
        self.valid_paths: list[Path] = []

        for path in self.paths:
            with Image.open(path) as image:
                hr_y = rgb_to_y_matlab(image)
            hr_y = modcrop_np(hr_y, self.scale)
            if min(hr_y.shape) < self.input_size:
                continue
            input_y = bicubic_degrade_matlab(hr_y, self.scale)
            self.hr_images.append(hr_y)
            self.input_images.append(input_y)
            self.valid_paths.append(path)

        if not self.hr_images:
            raise RuntimeError(
                f"No training image in {image_dir} is large enough for "
                f"input_size={self.input_size}."
            )

    def __len__(self) -> int:
        return len(self.hr_images) * self.repeat

    @staticmethod
    def _augment_pair(
        input_y: np.ndarray,
        hr_y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        rotation = random.randrange(4)
        if rotation:
            input_y = np.rot90(input_y, rotation)
            hr_y = np.rot90(hr_y, rotation)
        if random.random() < 0.5:
            input_y = np.fliplr(input_y)
            hr_y = np.fliplr(hr_y)
        if random.random() < 0.5:
            input_y = np.flipud(input_y)
            hr_y = np.flipud(hr_y)
        return input_y, hr_y

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_index = index % len(self.hr_images)
        input_y = self.input_images[image_index]
        hr_y = self.hr_images[image_index]

        if self.augment:
            input_y, hr_y = self._augment_pair(input_y, hr_y)

        height, width = input_y.shape
        top = random.randint(0, height - self.input_size)
        left = random.randint(0, width - self.input_size)

        x = input_y[top : top + self.input_size, left : left + self.input_size]
        label_top = top + self.label_border
        label_left = left + self.label_border
        y = hr_y[
            label_top : label_top + self.output_size,
            label_left : label_left + self.output_size,
        ]

        return (
            torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).float(),
            torch.from_numpy(np.ascontiguousarray(y)).unsqueeze(0).float(),
        )
