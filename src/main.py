"""Rate-based neural network for gender classification on LFW.

For each parameter listed in the project spec we run a one-at-a-time sweep
while holding the others at their default, and plot the resulting learning
curves and final test accuracies with matplotlib.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

# Ensure the flat src/ modules (`dataset`, `labels`, `model`, `train`) remain
# importable when Python's spawn start method re-imports this script inside
# DataLoader worker processes (this is the only method available on Windows).
_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import matplotlib

matplotlib.use("Agg")  # headless-safe on Windows, Linux, macOS

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import (
    DatasetConfig,
    LFWGenderDataset,
    PreloadedLFWGenderDataset,
    balance_and_split,
)
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


def make_loaders(train_ds, test_ds, batch_size: int,
                 workers: int) -> tuple[DataLoader, DataLoader]:
    # If the data is preloaded (often onto GPU), worker processes can't share
    # those tensors and pin_memory is a no-op — force the simple path.
    if isinstance(train_ds, PreloadedLFWGenderDataset):
        common = dict(num_workers=0, pin_memory=False)
    else:
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


# ---- Cross-product runner --------------------------------------------------

CROSS_AXES = (
    "hidden_layers", "neurons_per_layer", "sparsity",
    "learning_rate", "epochs", "batch_size",
)


def _config_key(cfg: dict) -> str:
    """Stable identifier for one cross-product config (used for resume)."""
    return "|".join(f"{k}={cfg[k]}" for k in CROSS_AXES)


def _load_done_keys(jsonl_path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if not jsonl_path.exists():
        return done
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done[_config_key(rec["config"])] = rec
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def run_cross_product(
    defaults: Defaults,
    train_ds,
    test_ds,
    device: torch.device,
    workers: int,
    out_dir: Path,
) -> list[dict]:
    """Full cross-product: 4 × 4 × 4 × 3 × 3 × 2 = 1,152 runs.

    Results stream to `cross_results.jsonl` after each run so the sweep can be
    resumed after interruption without losing work.
    """
    jsonl_path = out_dir / "cross_results.jsonl"
    already_done = _load_done_keys(jsonl_path)
    if already_done:
        print(f"Resuming: {len(already_done)} configs already complete in "
              f"{jsonl_path.name}")

    combos = list(itertools.product(
        HIDDEN_LAYERS, NEURONS_PER_LAYER, SPARSITY,
        LEARNING_RATES, EPOCH_CHOICES, BATCH_SIZES,
    ))
    print(f"Cross-product: {len(combos)} total runs "
          f"({len(combos) - len(already_done)} remaining)")

    records: list[dict] = list(already_done.values())
    bar = tqdm(combos, desc="cross-product", unit="run", dynamic_ncols=True)

    with jsonl_path.open("a") as jf:
        for layers, neurons, sparsity, lr, epochs, batch in bar:
            cfg = dict(
                hidden_layers=layers,
                neurons_per_layer=neurons,
                sparsity=sparsity,
                learning_rate=lr,
                epochs=epochs,
                batch_size=batch,
            )
            key = _config_key(cfg)
            if key in already_done:
                continue

            bar.set_postfix(
                L=layers, N=neurons, S=f"{int(sparsity * 100)}",
                lr=lr, E=epochs, B=batch,
            )
            result = run_one(defaults, train_ds, test_ds, device, workers,
                             override=cfg)
            rec = {
                "config": cfg,
                "final_test_acc": result.final_test_acc,
                "best_test_acc": result.best_test_acc,
                "final_train_acc": result.stats[-1].train_acc if result.stats else float("nan"),
                "wall_time_s": result.wall_time_s,
            }
            jf.write(json.dumps(rec, default=str) + "\n")
            jf.flush()
            records.append(rec)
    bar.close()
    return records


def _pivot(records: list[dict], row_axis: str, col_axis: str,
           row_vals: list, col_vals: list, metric: str,
           filter_: dict | None = None) -> np.ndarray:
    """Average `metric` over records that match `filter_`, producing an
    (|row_vals|, |col_vals|) grid. Missing cells come back as NaN.
    """
    grid = np.full((len(row_vals), len(col_vals)), np.nan, dtype=np.float64)
    counts = np.zeros_like(grid)

    for rec in records:
        cfg = rec["config"]
        if filter_ and any(cfg[k] != v for k, v in filter_.items()):
            continue
        try:
            i = row_vals.index(cfg[row_axis])
            j = col_vals.index(cfg[col_axis])
        except ValueError:
            continue
        val = rec[metric]
        if np.isnan(grid[i, j]):
            grid[i, j] = val
            counts[i, j] = 1
        else:
            grid[i, j] = (grid[i, j] * counts[i, j] + val) / (counts[i, j] + 1)
            counts[i, j] += 1
    return grid


def plot_cross_marginals(records: list[dict], out_dir: Path) -> None:
    """One bar chart per axis: mean best-test-acc over all other dims."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    axes = axes.flatten()

    axis_values = {
        "hidden_layers": HIDDEN_LAYERS,
        "neurons_per_layer": NEURONS_PER_LAYER,
        "sparsity": SPARSITY,
        "learning_rate": LEARNING_RATES,
        "epochs": EPOCH_CHOICES,
        "batch_size": BATCH_SIZES,
    }

    for ax, (axis, values) in zip(axes, axis_values.items()):
        means = []
        for v in values:
            vals = [r["best_test_acc"] for r in records if r["config"][axis] == v]
            means.append(float(np.mean(vals)) if vals else float("nan"))
        ax.bar(range(len(values)), means, color="#3b7ddd")
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([str(v) for v in values])
        ax.set_title(f"Mean best test acc vs {axis}")
        ax.set_ylabel("mean best test acc")
        ax.set_ylim(0.4, 1.0)
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / "cross_marginals.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_cross_heatmaps(records: list[dict], out_dir: Path) -> None:
    """For each (learning_rate, epochs, batch_size) triple, show a
    (hidden_layers × neurons_per_layer) heatmap of mean best test acc
    averaged across the 4 sparsity values.
    """
    n_rows = len(LEARNING_RATES)
    n_cols = len(EPOCH_CHOICES) * len(BATCH_SIZES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.5 * n_rows),
                             squeeze=False)

    # Shared color scale across all heatmaps for comparability.
    all_grids = []
    for lr in LEARNING_RATES:
        for ep in EPOCH_CHOICES:
            for bs in BATCH_SIZES:
                grid = _pivot(records, "hidden_layers", "neurons_per_layer",
                              HIDDEN_LAYERS, NEURONS_PER_LAYER,
                              metric="best_test_acc",
                              filter_={"learning_rate": lr, "epochs": ep, "batch_size": bs})
                all_grids.append(grid)
    finite = [g[np.isfinite(g)] for g in all_grids]
    flat = np.concatenate(finite) if any(len(a) for a in finite) else np.array([0.5, 1.0])
    vmin = float(np.min(flat)) if flat.size else 0.5
    vmax = float(np.max(flat)) if flat.size else 1.0

    col_idx = 0
    for j_ep, ep in enumerate(EPOCH_CHOICES):
        for j_bs, bs in enumerate(BATCH_SIZES):
            for i_lr, lr in enumerate(LEARNING_RATES):
                grid = _pivot(records, "hidden_layers", "neurons_per_layer",
                              HIDDEN_LAYERS, NEURONS_PER_LAYER,
                              metric="best_test_acc",
                              filter_={"learning_rate": lr, "epochs": ep, "batch_size": bs})
                ax = axes[i_lr][col_idx]
                im = ax.imshow(grid, vmin=vmin, vmax=vmax, cmap="viridis",
                               aspect="auto")
                ax.set_xticks(range(len(NEURONS_PER_LAYER)))
                ax.set_xticklabels(NEURONS_PER_LAYER, fontsize=7)
                ax.set_yticks(range(len(HIDDEN_LAYERS)))
                ax.set_yticklabels(HIDDEN_LAYERS, fontsize=7)
                if i_lr == 0:
                    ax.set_title(f"epochs={ep}, batch={bs}", fontsize=8)
                if col_idx == 0:
                    ax.set_ylabel(f"lr={lr}\nlayers", fontsize=8)
                else:
                    ax.set_ylabel("layers", fontsize=7)
                ax.set_xlabel("neurons", fontsize=7)
                for ii in range(grid.shape[0]):
                    for jj in range(grid.shape[1]):
                        v = grid[ii, jj]
                        if np.isfinite(v):
                            ax.text(jj, ii, f"{v:.2f}", ha="center", va="center",
                                    color="white" if v < (vmin + vmax) / 2 else "black",
                                    fontsize=7)
            col_idx += 1

    fig.suptitle("Mean best test accuracy — averaged over sparsity", y=1.02)
    fig.tight_layout()
    out_path = out_dir / "cross_heatmaps.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def write_cross_top_configs(records: list[dict], out_dir: Path, n: int = 20) -> None:
    top = sorted(records, key=lambda r: r["best_test_acc"], reverse=True)[:n]
    out_path = out_dir / "cross_top_configs.txt"
    lines = [f"Top {n} configs by best test accuracy\n" + "=" * 44 + ""]
    for r in top:
        cfg = r["config"]
        lines.append(
            f"best_test_acc={r['best_test_acc']:.4f}  "
            f"final_test_acc={r['final_test_acc']:.4f}  "
            f"wall={r['wall_time_s']:.1f}s  "
            f"| layers={cfg['hidden_layers']} neurons={cfg['neurons_per_layer']} "
            f"sparsity={cfg['sparsity']} lr={cfg['learning_rate']} "
            f"epochs={cfg['epochs']} batch={cfg['batch_size']}"
        )
    out_path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out_path}")


# ---- CLI -------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--lfw-root",
        default="data/LFW/lfw-deepfunneled/lfw-deepfunneled",
        help="Directory of per-identity LFW folders.",
    )
    p.add_argument("--out-dir", default="results")
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "DataLoader worker processes. Default 0 (no multiprocessing) is "
            "the safest choice on Windows, which uses spawn. On Linux/macOS "
            "values like 2-4 can be faster."
        ),
    )
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
    p.add_argument(
        "--cross-product",
        action="store_true",
        help=(
            "Run the full 1,152-config cross-product sweep "
            "(hidden_layers × neurons × sparsity × lr × epochs × batch). "
            "Implies --preload. Results stream to cross_results.jsonl so the "
            "run is resumable."
        ),
    )
    p.add_argument(
        "--preload",
        action="store_true",
        help=(
            "Decode and resize every LFW image once into a cached tensor in RAM. "
            "Makes epochs ~10x faster on GPU by removing the DataLoader "
            "preprocessing bottleneck. Auto-enabled with --cross-product."
        ),
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

    preload = args.preload or args.cross_product
    if preload:
        train_ds = PreloadedLFWGenderDataset(train_samples, ds_cfg,
                                             desc="preload train")
        test_ds = PreloadedLFWGenderDataset(test_samples, ds_cfg,
                                            desc="preload test ")
        # Move cached tensors to GPU up front — turns __getitem__ into a
        # device-local tensor view, so DataLoader copies nothing per batch.
        if device.type == "cuda":
            train_ds.to(device)
            test_ds.to(device)
            print("Preloaded datasets moved to GPU.")
    else:
        train_ds = LFWGenderDataset(train_samples, ds_cfg)
        test_ds = LFWGenderDataset(test_samples, ds_cfg)

    defaults = Defaults(image_size=args.image_size)
    summary: list[dict] = []

    if args.cross_product:
        records = run_cross_product(
            defaults=defaults,
            train_ds=train_ds,
            test_ds=test_ds,
            device=device,
            workers=args.workers,
            out_dir=out_dir,
        )
        print(f"\nCross-product complete: {len(records)} runs total")
        plot_cross_marginals(records, out_dir)
        plot_cross_heatmaps(records, out_dir)
        write_cross_top_configs(records, out_dir)
        return

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
