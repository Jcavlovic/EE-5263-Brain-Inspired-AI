"""Rate-based neural network for gender classification on LFW.

For each parameter listed in the project spec we run a one-at-a-time sweep
while holding the others at their default, and plot the resulting learning
curves and final test accuracies with matplotlib.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset import DatasetConfig, LFWGenderDataset, balance_and_split
from labels import build_labeled_samples, summarize
from model import ModelConfig, RateBasedNN
from train import RunResult, TrainConfig, train_model


# ---- Parameter grids from claude.txt ---------------------------------------
HIDDEN_LAYERS = [1, 2, 3, 4]
NEURONS_PER_LAYER = [64, 128, 256, 512]
SPARSITY = [0.10, 0.20, 0.50, 0.80]
LEARNING_RATES = [0.1, 0.001, 0.0001]
EPOCH_CHOICES = [25, 100, 200]
BATCH_SIZES = [32, 128]


@dataclass
class Defaults:
    hidden_layers: int = 2
    neurons_per_layer: int = 256
    sparsity: float = 0.20
    learning_rate: float = 0.001
    epochs: int = 25
    batch_size: int = 128
    image_size: int = 64


@dataclass
class SweepPoint:
    label: str
    value: object
    result: RunResult


def make_model(d: Defaults, input_dim: int) -> RateBasedNN:
    return RateBasedNN(ModelConfig(
        input_dim=input_dim,
        hidden_layers=d.hidden_layers,
        neurons_per_layer=d.neurons_per_layer,
        sparsity=d.sparsity,
    ))


def make_loaders(train_ds: LFWGenderDataset, test_ds: LFWGenderDataset,
                 batch_size: int, workers: int) -> tuple[DataLoader, DataLoader]:
    common = dict(num_workers=workers, pin_memory=torch.cuda.is_available())
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common),
    )


def run_one(
    defaults: Defaults,
    train_ds: LFWGenderDataset,
    test_ds: LFWGenderDataset,
    device: torch.device,
    workers: int,
    override: dict,
) -> RunResult:
    d = replace(defaults, **override)
    model = make_model(d, input_dim=d.image_size * d.image_size).to(device)
    train_loader, test_loader = make_loaders(train_ds, test_ds, d.batch_size, workers)
    return train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        cfg=TrainConfig(
            learning_rate=d.learning_rate,
            epochs=d.epochs,
            batch_size=d.batch_size,
        ),
        device=device,
    )


def plot_sweep(points: list[SweepPoint], param_name: str, xlabel: str,
               out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for pt in points:
        epochs = [s.epoch for s in pt.result.stats]
        accs = [s.test_acc for s in pt.result.stats]
        ax1.plot(epochs, accs, marker="o", markersize=3, label=pt.label)
    ax1.set_title(f"Test accuracy per epoch — varying {param_name}")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("test accuracy")
    ax1.set_ylim(0.4, 1.0)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    x_labels = [str(pt.value) for pt in points]
    final_acc = [pt.result.final_test_acc for pt in points]
    best_acc = [pt.result.best_test_acc for pt in points]
    xs = range(len(points))
    ax2.bar([i - 0.2 for i in xs], final_acc, width=0.4, label="final")
    ax2.bar([i + 0.2 for i in xs], best_acc, width=0.4, label="best")
    ax2.set_xticks(list(xs))
    ax2.set_xticklabels(x_labels)
    ax2.set_title(f"Final/best test accuracy vs {param_name}")
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("test accuracy")
    ax2.set_ylim(0.4, 1.0)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    out_path = out_dir / f"sweep_{param_name}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_path}")


def sweep(
    defaults: Defaults,
    param_name: str,
    xlabel: str,
    override_key: str,
    values: list,
    label_fmt,
    train_ds: LFWGenderDataset,
    test_ds: LFWGenderDataset,
    device: torch.device,
    workers: int,
    out_dir: Path,
    summary: list,
) -> None:
    print(f"\n=== Sweeping {param_name}: {values} ===")
    points: list[SweepPoint] = []
    for v in values:
        label = label_fmt(v)
        print(f"-- {param_name} = {label}")
        result = run_one(defaults, train_ds, test_ds, device, workers,
                         override={override_key: v})
        print(
            f"  done in {result.wall_time_s:.1f}s  "
            f"final_test_acc={result.final_test_acc:.3f}  "
            f"best_test_acc={result.best_test_acc:.3f}"
        )
        points.append(SweepPoint(label=label, value=v, result=result))
        summary.append({
            "parameter": param_name,
            "value": v,
            "final_test_acc": result.final_test_acc,
            "best_test_acc": result.best_test_acc,
            "wall_time_s": result.wall_time_s,
            "epochs": [asdict(s) for s in result.stats],
        })
    plot_sweep(points, param_name, xlabel, out_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--lfw-root",
        default="data/LFW/lfw-deepfunneled/lfw-deepfunneled",
        help="Directory of per-identity LFW folders.",
    )
    p.add_argument("--out-dir", default="results")
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument(
        "--quick",
        action="store_true",
        help="Skip the expensive 100/200-epoch sweep points for a fast demo run.",
    )
    p.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Run only the named sweeps (e.g. --only hidden_layers sparsity).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    lfw_root = (repo_root / args.lfw_root).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"LFW root:     {lfw_root}")
    print(f"Output dir:   {out_dir}")

    print("Building labeled samples from identity names...")
    samples = build_labeled_samples(lfw_root)
    stats = summarize(samples)
    print(f"  labeled: {stats}")

    ds_cfg = DatasetConfig(image_size=args.image_size)
    train_samples, test_samples = balance_and_split(samples, ds_cfg)
    print(
        f"Train: {len(train_samples)} images "
        f"(M={sum(1 for s in train_samples if s.label == 0)}, "
        f"F={sum(1 for s in train_samples if s.label == 1)})"
    )
    print(
        f"Test:  {len(test_samples)} images "
        f"(M={sum(1 for s in test_samples if s.label == 0)}, "
        f"F={sum(1 for s in test_samples if s.label == 1)})"
    )

    train_ds = LFWGenderDataset(train_samples, ds_cfg)
    test_ds = LFWGenderDataset(test_samples, ds_cfg)

    defaults = Defaults(image_size=args.image_size)
    summary: list[dict] = []

    epoch_values = [e for e in EPOCH_CHOICES if not args.quick or e <= 25]

    all_sweeps = {
        "hidden_layers": dict(
            xlabel="# hidden layers",
            override_key="hidden_layers",
            values=HIDDEN_LAYERS,
            label_fmt=lambda v: f"{v} layer(s)",
        ),
        "neurons_per_layer": dict(
            xlabel="neurons per hidden layer",
            override_key="neurons_per_layer",
            values=NEURONS_PER_LAYER,
            label_fmt=lambda v: f"{v} units",
        ),
        "sparsity": dict(
            xlabel="sparsity (fraction of weights zeroed)",
            override_key="sparsity",
            values=SPARSITY,
            label_fmt=lambda v: f"{int(v * 100)}% sparse",
        ),
        "learning_rate": dict(
            xlabel="learning rate",
            override_key="learning_rate",
            values=LEARNING_RATES,
            label_fmt=lambda v: f"lr={v}",
        ),
        "epochs": dict(
            xlabel="epochs",
            override_key="epochs",
            values=epoch_values,
            label_fmt=lambda v: f"{v} epochs",
        ),
        "batch_size": dict(
            xlabel="batch size",
            override_key="batch_size",
            values=BATCH_SIZES,
            label_fmt=lambda v: f"batch={v}",
        ),
    }

    selected = args.only or list(all_sweeps.keys())
    for name in selected:
        if name not in all_sweeps:
            raise SystemExit(f"Unknown sweep: {name}")
        spec = all_sweeps[name]
        sweep(
            defaults=defaults,
            param_name=name,
            xlabel=spec["xlabel"],
            override_key=spec["override_key"],
            values=spec["values"],
            label_fmt=spec["label_fmt"],
            train_ds=train_ds,
            test_ds=test_ds,
            device=device,
            workers=args.workers,
            out_dir=out_dir,
            summary=summary,
        )

    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(
            {"defaults": asdict(defaults), "results": summary},
            f,
            indent=2,
            default=str,
        )
    print(f"\nWrote summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
