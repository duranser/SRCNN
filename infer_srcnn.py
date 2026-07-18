from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from evaluate_srcnn import load_checkpoint
from srcnn.matlab import (
    bicubic_degrade_ycbcr_matlab,
    imresize_ycbcr_matlab,
    rgb_to_ycbcr_matlab,
    ycbcr_to_pil_rgb,
)
from srcnn.utils import merge_y_with_bicubic_chroma, tensor_to_y_numpy, y_numpy_to_tensor


def _save_image(image: Image.Image, output_path: str | Path, label: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    print(f"Saved {label}: {path} ({image.width}x{image.height})")


def _modcrop_rgb(image: Image.Image, scale: int) -> Image.Image:
    """Crop only the bottom/right edges so width and height divide by scale."""
    rgb = image.convert("RGB")
    width, height = rgb.size
    cropped_width = width - (width % scale)
    cropped_height = height - (height % scale)
    if cropped_width <= 0 or cropped_height <= 0:
        raise ValueError(
            f"Input image size {width}x{height} is too small for scale={scale}."
        )
    return rgb.crop((0, 0, cropped_width, cropped_height))


def _prepare_from_ground_truth(
    ground_truth_rgb: Image.Image,
    scale: int,
) -> tuple[Image.Image, Image.Image, Image.Image, object, object]:
    """Generate thesis outputs from one HR ground-truth image.

    Returns the scale-compatible ground truth, true LR image, bicubic RGB image,
    bicubic Y channel, and bicubic YCbCr image. The bicubic image has
    the same dimensions as the returned ground truth.
    """
    gt_rgb = _modcrop_rgb(ground_truth_rgb, scale)
    gt_ycbcr = rgb_to_ycbcr_matlab(gt_rgb)
    lr_ycbcr, bicubic_ycbcr = bicubic_degrade_ycbcr_matlab(gt_ycbcr, scale)

    lr_rgb = ycbcr_to_pil_rgb(lr_ycbcr)
    bicubic_rgb = ycbcr_to_pil_rgb(bicubic_ycbcr)

    expected_lr_size = (gt_rgb.width // scale, gt_rgb.height // scale)
    if lr_rgb.size != expected_lr_size:
        raise RuntimeError(
            f"Unexpected LR size {lr_rgb.size}; expected {expected_lr_size}."
        )
    if bicubic_rgb.size != gt_rgb.size:
        raise RuntimeError(
            f"Unexpected bicubic size {bicubic_rgb.size}; expected {gt_rgb.size}."
        )

    return gt_rgb, lr_rgb, bicubic_rgb, bicubic_ycbcr[..., 0], bicubic_ycbcr


def _prepare_from_existing_lr(
    lr_rgb: Image.Image,
    scale: int,
    ground_truth_rgb: Image.Image | None,
) -> tuple[Image.Image | None, Image.Image, Image.Image, object, object]:
    """Prepare inference from a supplied native-resolution LR image.

    When a ground-truth image is supplied, it must be exactly scale times the
    LR dimensions. This guarantees that nearest, bicubic, and SRCNN outputs
    have the same size as the ground truth.
    """
    lr_rgb = lr_rgb.convert("RGB")
    expected_hr_size = (lr_rgb.width * scale, lr_rgb.height * scale)

    gt_rgb: Image.Image | None = None
    if ground_truth_rgb is not None:
        gt_rgb = ground_truth_rgb.convert("RGB")
        if gt_rgb.size != expected_hr_size:
            raise ValueError(
                "Ground-truth size must equal LR size multiplied by scale: "
                f"LR={lr_rgb.size}, scale={scale}, expected GT={expected_hr_size}, "
                f"received GT={gt_rgb.size}."
            )

    lr_ycbcr = rgb_to_ycbcr_matlab(lr_rgb)
    bicubic_ycbcr = imresize_ycbcr_matlab(
        lr_ycbcr,
        (expected_hr_size[1], expected_hr_size[0]),
        antialiasing=False,
    )
    bicubic_rgb = ycbcr_to_pil_rgb(bicubic_ycbcr)
    return gt_rgb, lr_rgb, bicubic_rgb, bicubic_ycbcr[..., 0], bicubic_ycbcr


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run luminance-channel SRCNN and save a native LR image, its "
            "nearest-neighbor and bicubic HR-size visualizations, the ground "
            "truth, and an SRCNN output with the same size as the ground truth."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "HR ground-truth image by default. With --already-lr, this is the "
            "native-resolution LR image."
        ),
    )
    parser.add_argument("--output", required=True, help="Path for the SRCNN RGB result.")
    parser.add_argument("--save-lr", default=None, help="Save the true reduced-size LR image.")
    parser.add_argument(
        "--save-nearest",
        default=None,
        help="Save LR enlarged to ground-truth size using nearest-neighbor interpolation.",
    )
    parser.add_argument(
        "--save-bicubic",
        default=None,
        help="Save LR enlarged to ground-truth size using bicubic interpolation.",
    )
    parser.add_argument(
        "--save-ground-truth",
        "--save-gt",
        dest="save_ground_truth",
        default=None,
        help="Save the ground-truth RGB image used as the HR reference.",
    )
    parser.add_argument("--scale", type=int, default=None)
    parser.add_argument(
        "--already-lr",
        action="store_true",
        help="Treat --input as an existing LR image instead of an HR ground truth.",
    )
    parser.add_argument(
        "--ground-truth",
        default=None,
        help=(
            "Optional HR ground truth for --already-lr mode. Its dimensions must "
            "equal the LR dimensions multiplied by scale."
        ),
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, saved_args = load_checkpoint(args.checkpoint, device)
    model.eval()

    scale = int(args.scale if args.scale is not None else saved_args.get("scale", 2))
    if scale <= 1:
        raise ValueError(f"scale must be greater than 1, got {scale}")

    with Image.open(args.input) as image:
        input_rgb = image.convert("RGB")

        if args.already_lr:
            gt_source = None
            if args.ground_truth:
                with Image.open(args.ground_truth) as gt_image:
                    gt_source = gt_image.convert("RGB")
            gt_rgb, lr_rgb, bicubic_rgb, bicubic_y, bicubic_ycbcr = (
                _prepare_from_existing_lr(input_rgb, scale, gt_source)
            )
        else:
            if args.ground_truth:
                raise ValueError(
                    "--ground-truth is only used with --already-lr. Without "
                    "--already-lr, --input itself is the ground truth."
                )
            gt_rgb, lr_rgb, bicubic_rgb, bicubic_y, bicubic_ycbcr = (
                _prepare_from_ground_truth(input_rgb, scale)
            )

    target_size = gt_rgb.size if gt_rgb is not None else bicubic_rgb.size
    nearest_rgb = lr_rgb.resize(target_size, Image.Resampling.NEAREST)

    input_tensor = y_numpy_to_tensor(bicubic_y).unsqueeze(0).to(device)
    sr_y = tensor_to_y_numpy(model.forward_same(input_tensor), quantize=False)
    srcnn_rgb = merge_y_with_bicubic_chroma(sr_y, bicubic_ycbcr)

    for label, image in {
        "nearest": nearest_rgb,
        "bicubic": bicubic_rgb,
        "SRCNN": srcnn_rgb,
    }.items():
        if image.size != target_size:
            raise RuntimeError(
                f"{label} output size {image.size} does not match target size {target_size}."
            )

    _save_image(srcnn_rgb, args.output, "SRCNN")
    if args.save_lr:
        _save_image(lr_rgb, args.save_lr, "native LR")
    if args.save_nearest:
        _save_image(nearest_rgb, args.save_nearest, "nearest-neighbor visualization")
    if args.save_bicubic:
        _save_image(bicubic_rgb, args.save_bicubic, "bicubic visualization")
    if args.save_ground_truth:
        if gt_rgb is None:
            raise ValueError(
                "A ground-truth image is required to use --save-ground-truth in "
                "--already-lr mode. Pass it with --ground-truth."
            )
        _save_image(gt_rgb, args.save_ground_truth, "ground truth")

    print("\nOutput dimensions")
    print(f"  Ground truth / target: {target_size[0]}x{target_size[1]}")
    print(f"  Native LR:             {lr_rgb.width}x{lr_rgb.height}")
    print(f"  Nearest:               {nearest_rgb.width}x{nearest_rgb.height}")
    print(f"  Bicubic:               {bicubic_rgb.width}x{bicubic_rgb.height}")
    print(f"  SRCNN:                 {srcnn_rgb.width}x{srcnn_rgb.height}")

    if gt_rgb is not None and gt_rgb.size != input_rgb.size and not args.already_lr:
        print(
            "Note: the input ground truth was mod-cropped on the bottom/right "
            "so both dimensions are divisible by the scale factor."
        )


if __name__ == "__main__":
    main()
