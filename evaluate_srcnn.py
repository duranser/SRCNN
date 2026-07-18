from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from srcnn.matlab import (
    bicubic_degrade_ycbcr_matlab,
    modcrop_np,
    rgb_to_ycbcr_matlab,
    ycbcr_to_pil_rgb,
)
from srcnn.model import SRCNN, count_parameters
from srcnn.utils import (
    calculate_psnr_y,
    calculate_ssim_y,
    list_image_files,
    merge_y_with_bicubic_chroma,
    tensor_to_y_numpy,
    y_numpy_to_tensor,
)


@torch.no_grad()
def evaluate_folder(
    model: SRCNN,
    image_dir: str | Path,
    scale: int,
    device: torch.device,
    save_dir: str | Path | None = None,
    crop_border: int | None = None,
) -> dict:
    """Evaluate SRCNN on MATLAB-style Y and bicubic Cb/Cr."""
    model.eval()
    paths = list_image_files(image_dir)
    border = int(scale if crop_border is None else crop_border)

    out_dir = Path(save_dir) if save_dir else None
    if out_dir:
        (out_dir / "srcnn").mkdir(parents=True, exist_ok=True)
        (out_dir / "bicubic").mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for path in paths:
        with Image.open(path) as image:
            hr_ycbcr = modcrop_np(rgb_to_ycbcr_matlab(image), scale)

        _, bicubic_ycbcr = bicubic_degrade_ycbcr_matlab(hr_ycbcr, scale)
        hr_y = hr_ycbcr[..., 0]
        bicubic_y = bicubic_ycbcr[..., 0]

        inputs = y_numpy_to_tensor(bicubic_y).unsqueeze(0).to(device)
        sr_y = tensor_to_y_numpy(model.forward_same(inputs), quantize=False)

        row = {
            "image": path.name,
            "srcnn_y_psnr": calculate_psnr_y(sr_y, hr_y, border, True),
            "bicubic_y_psnr": calculate_psnr_y(bicubic_y, hr_y, border, True),
            "srcnn_y_ssim": calculate_ssim_y(sr_y, hr_y, border, True),
            "bicubic_y_ssim": calculate_ssim_y(bicubic_y, hr_y, border, True),
        }
        rows.append(row)

        if out_dir:
            merge_y_with_bicubic_chroma(sr_y, bicubic_ycbcr).save(
                out_dir / "srcnn" / path.name
            )
            ycbcr_to_pil_rgb(bicubic_ycbcr).save(out_dir / "bicubic" / path.name)

    def average(key: str):
        values = [row[key] for row in rows if row[key] is not None]
        return sum(values) / len(values) if values else None

    return {
        "dataset": str(image_dir),
        "scale": int(scale),
        "crop_border": border,
        "num_images": len(rows),
        "mean_srcnn_y_psnr": average("srcnn_y_psnr"),
        "mean_bicubic_y_psnr": average("bicubic_y_psnr"),
        "mean_srcnn_y_ssim": average("srcnn_y_ssim"),
        "mean_bicubic_y_ssim": average("bicubic_y_ssim"),
        "images": rows,
    }


def load_checkpoint(checkpoint_path: str | Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})
    model = SRCNN(
        n1=int(saved_args.get("n1", 64)),
        n2=int(saved_args.get("n2", 32)),
        f1=int(saved_args.get("f1", 9)),
        f2=int(saved_args.get("f2", 1)),
        f3=int(saved_args.get("f3", 5)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    return model, saved_args


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Y-channel SRCNN.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--scale", type=int, default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--crop-border", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, saved_args = load_checkpoint(args.checkpoint, device)
    scale = int(args.scale if args.scale is not None else saved_args.get("scale", 2))
    result = evaluate_folder(
        model,
        args.test_dir,
        scale,
        device,
        args.save_dir,
        args.crop_border,
    )
    result["checkpoint"] = args.checkpoint
    result["parameters"] = count_parameters(model)
    print(json.dumps(result, indent=2))

    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
