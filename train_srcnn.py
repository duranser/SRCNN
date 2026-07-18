from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evaluate_srcnn import evaluate_folder
from srcnn.dataset import SRCNNRandomPatchDataset
from srcnn.model import SRCNN, count_parameters
from srcnn.utils import append_csv_row, seed_worker, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Y-channel SRCNN with an epoch-based Adam/MSE protocol."
    )
    parser.add_argument("--train-dir", default="data/91-image")
    parser.add_argument("--val-dirs", nargs="*", default=["data/Set5", "data/Set14"])
    parser.add_argument("--scale", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--f1", type=int, default=9)
    parser.add_argument("--f2", type=int, default=1)
    parser.add_argument("--f3", type=int, default=5)
    parser.add_argument("--n1", type=int, default=64)
    parser.add_argument("--n2", type=int, default=32)
    parser.add_argument("--input-size", type=int, default=33)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--crop-border", type=int, default=None)
    parser.add_argument("--save-dir", default="runs/srcnn_adam_x2_9_1_5")
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def save_checkpoint(
    path: Path,
    model: SRCNN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    extra: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "extra": extra,
        },
        path,
    )


def mean_metric(metrics_by_dataset: dict[str, dict], key: str) -> float | None:
    values = [m[key] for m in metrics_by_dataset.values() if m.get(key) is not None]
    return sum(values) / len(values) if values else None


def format_metrics(metrics_by_dataset: dict[str, dict]) -> str:
    parts = []
    for name, metrics in metrics_by_dataset.items():
        parts.append(
            f"{name}: PSNR={metrics['mean_srcnn_y_psnr']:.4f} dB, "
            f"SSIM={metrics['mean_srcnn_y_ssim']:.6f}"
        )
    return " | ".join(parts)


def make_log_fields(val_dirs: list[str]) -> list[str]:
    fields = [
        "epoch",
        "train_mse",
        "learning_rate",
        "elapsed_sec",
        "mean_srcnn_y_psnr",
        "mean_bicubic_y_psnr",
        "mean_srcnn_y_ssim",
        "mean_bicubic_y_ssim",
        "best_psnr",
        "best_ssim",
    ]
    for val_dir in val_dirs:
        name = Path(val_dir).name
        fields.extend(
            [
                f"{name}_srcnn_y_psnr",
                f"{name}_bicubic_y_psnr",
                f"{name}_srcnn_y_ssim",
                f"{name}_bicubic_y_ssim",
            ]
        )
    return fields


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.repeat <= 0 or args.batch_size <= 0:
        raise ValueError("epochs, repeat, and batch-size must all be positive")

    set_seed(args.seed)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    dataset = SRCNNRandomPatchDataset(
        image_dir=args.train_dir,
        scale=args.scale,
        input_size=args.input_size,
        repeat=args.repeat,
        f1=args.f1,
        f2=args.f2,
        f3=args.f3,
        augment=args.augment,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=args.num_workers > 0,
    )

    model = SRCNN(args.n1, args.n2, args.f1, args.f2, args.f3).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start_epoch = 1
    best_psnr = float("-inf")
    best_ssim = float("-inf")
    last_metrics: dict[str, dict] = {}

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        if checkpoint.get("optimizer") is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        extra = checkpoint.get("extra", {})
        best_psnr = float(extra.get("best_psnr", best_psnr))
        best_ssim = float(extra.get("best_ssim", best_ssim))
        last_metrics = extra.get("metrics", {})

    save_dir = Path(args.save_dir)
    epochs_dir = save_dir / "epochs"
    epochs_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "metrics.csv"
    log_fields = make_log_fields(args.val_dirs)

    print(f"Device: {device}")
    print(
        f"Model: SRCNN {args.f1}-{args.f2}-{args.f3}, "
        f"n1={args.n1}, n2={args.n2}, scale=x{args.scale}"
    )
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Training images: {len(dataset.hr_images)}")
    print(f"Repeat: {args.repeat}; samples/epoch: {len(dataset):,}")
    print(f"Batches/epoch: {len(loader):,}; batch size: {args.batch_size}")
    print(f"Input/label size: {dataset.input_size}/{dataset.output_size}")
    print(f"Optimizer: Adam; learning rate: {args.lr:g}")
    print("Loss: mean squared error (PyTorch MSE)")
    print(f"Epochs: {args.epochs}; starting epoch: {start_epoch}")

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=args.amp and device.type == "cuda",
    )
    current_epoch = start_epoch
    completed_epoch = start_epoch - 1

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            current_epoch = epoch
            model.train()
            epoch_start = time.time()
            loss_sum = 0.0

            for inputs, targets in loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=args.amp and device.type == "cuda",
                ):
                    predictions = model(inputs)
                    loss = F.mse_loss(predictions, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_sum += loss.item()


            completed_epoch = epoch
            avg_loss = loss_sum / max(len(loader), 1)
            elapsed = time.time() - epoch_start
            metrics_by_dataset: dict[str, dict] = {}

            if args.val_dirs and args.eval_every > 0 and epoch % args.eval_every == 0:
                for val_dir in args.val_dirs:
                    result = evaluate_folder(
                        model=model,
                        image_dir=val_dir,
                        scale=args.scale,
                        device=device,
                        save_dir=None,
                        crop_border=args.crop_border,
                    )
                    metrics_by_dataset[Path(val_dir).name] = result
                last_metrics = metrics_by_dataset

            mean_psnr = mean_metric(metrics_by_dataset, "mean_srcnn_y_psnr")
            mean_bicubic_psnr = mean_metric(metrics_by_dataset, "mean_bicubic_y_psnr")
            mean_ssim = mean_metric(metrics_by_dataset, "mean_srcnn_y_ssim")
            mean_bicubic_ssim = mean_metric(metrics_by_dataset, "mean_bicubic_y_ssim")

            if metrics_by_dataset:
                print(
                    f"Epoch {epoch:04d} complete | train_mse={avg_loss:.8f} | "
                    f"{format_metrics(metrics_by_dataset)} | {elapsed:.1f}s"
                )
            else:
                print(
                    f"Epoch {epoch:04d} complete | train_mse={avg_loss:.8f} | "
                    f"{elapsed:.1f}s"
                )

            improved_psnr = mean_psnr is not None and mean_psnr > best_psnr
            improved_ssim = mean_ssim is not None and mean_ssim > best_ssim
            if improved_psnr:
                best_psnr = float(mean_psnr)
            if improved_ssim:
                best_ssim = float(mean_ssim)

            extra = {
                "best_psnr": best_psnr,
                "best_ssim": best_ssim,
                "metrics": last_metrics,
                "train_loss": avg_loss,
                "training_protocol": "Adam + mean MSE + epoch/repeat",
            }
            save_checkpoint(
                epochs_dir / f"epoch_{epoch:04d}.pth",
                model,
                optimizer,
                epoch,
                args,
                extra,
            )
            save_checkpoint(save_dir / "latest.pth", model, optimizer, epoch, args, extra)
            if improved_psnr:
                save_checkpoint(
                    save_dir / "best_psnr.pth", model, optimizer, epoch, args, extra
                )
                save_checkpoint(save_dir / "best.pth", model, optimizer, epoch, args, extra)
            if improved_ssim:
                save_checkpoint(
                    save_dir / "best_ssim.pth", model, optimizer, epoch, args, extra
                )

            row = {
                "epoch": epoch,
                "train_mse": avg_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_sec": elapsed,
                "mean_srcnn_y_psnr": "" if mean_psnr is None else mean_psnr,
                "mean_bicubic_y_psnr": ""
                if mean_bicubic_psnr is None
                else mean_bicubic_psnr,
                "mean_srcnn_y_ssim": "" if mean_ssim is None else mean_ssim,
                "mean_bicubic_y_ssim": ""
                if mean_bicubic_ssim is None
                else mean_bicubic_ssim,
                "best_psnr": "" if best_psnr == float("-inf") else best_psnr,
                "best_ssim": "" if best_ssim == float("-inf") else best_ssim,
            }
            for name, metrics in metrics_by_dataset.items():
                row[f"{name}_srcnn_y_psnr"] = metrics["mean_srcnn_y_psnr"]
                row[f"{name}_bicubic_y_psnr"] = metrics["mean_bicubic_y_psnr"]
                row[f"{name}_srcnn_y_ssim"] = metrics["mean_srcnn_y_ssim"]
                row[f"{name}_bicubic_y_ssim"] = metrics["mean_bicubic_y_ssim"]
            append_csv_row(log_path, log_fields, row)

    except KeyboardInterrupt:
        extra = {
            "best_psnr": best_psnr,
            "best_ssim": best_ssim,
            "metrics": last_metrics,
            "interrupted_during_epoch": current_epoch,
            "training_protocol": "Adam + mean MSE + epoch/repeat",
        }
        save_checkpoint(
            save_dir / "interrupted.pth",
            model,
            optimizer,
            completed_epoch,
            args,
            extra,
        )
        print(
            f"\nTraining interrupted. Saved: {save_dir / 'interrupted.pth'} "
            f"(last completed epoch: {completed_epoch})"
        )


if __name__ == "__main__":
    main()
