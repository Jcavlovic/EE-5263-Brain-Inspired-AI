"""Training / evaluation loops for the rate-based gender classifier."""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from model import RateBasedNN


@dataclass
class TrainConfig:
    learning_rate: float = 0.001
    epochs: int = 25
    batch_size: int = 128


@dataclass
class EpochStats:
    epoch: int
    train_loss: float
    train_acc: float
    test_loss: float
    test_acc: float


@dataclass
class RunResult:
    stats: list[EpochStats] = field(default_factory=list)
    wall_time_s: float = 0.0

    @property
    def final_test_acc(self) -> float:
        return self.stats[-1].test_acc if self.stats else float("nan")

    @property
    def best_test_acc(self) -> float:
        return max((s.test_acc for s in self.stats), default=float("nan"))


def _run_epoch(
    model: RateBasedNN,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    progress_desc: str | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_n = 0

    bar = tqdm(
        loader,
        desc=progress_desc or ("train" if training else "eval"),
        leave=False,
        unit="batch",
        dynamic_ncols=True,
    )
    for x, y in bar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            pred = model(x)
            loss = loss_fn(pred, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                model.reapply_masks()

        total_loss += loss.item() * x.size(0)
        total_correct += int(((pred > 0.5).float() == y).sum().item())
        total_n += x.size(0)
        bar.set_postfix(
            loss=f"{total_loss / total_n:.4f}",
            acc=f"{total_correct / total_n:.3f}",
        )

    bar.close()
    return total_loss / max(total_n, 1), total_correct / max(total_n, 1)


def train_model(
    model: RateBasedNN,
    train_loader: DataLoader,
    test_loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
    log_fn=print,
) -> RunResult:
    loss_fn = nn.BCELoss()
    # Adam gives stacked-sigmoid networks a fighting chance against vanishing
    # gradients without changing the learning-rate grid specified in claude.txt.
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    result = RunResult()

    start = perf_counter()
    epoch_bar = tqdm(
        range(1, cfg.epochs + 1),
        desc="epochs",
        unit="epoch",
        dynamic_ncols=True,
    )
    for epoch in epoch_bar:
        tr_loss, tr_acc = _run_epoch(
            model, train_loader, loss_fn, device, optimizer,
            progress_desc=f"  train e{epoch:>3}/{cfg.epochs}",
        )
        te_loss, te_acc = _run_epoch(
            model, test_loader, loss_fn, device, None,
            progress_desc=f"  eval  e{epoch:>3}/{cfg.epochs}",
        )
        result.stats.append(EpochStats(epoch, tr_loss, tr_acc, te_loss, te_acc))
        epoch_bar.set_postfix(
            train_acc=f"{tr_acc:.3f}",
            test_acc=f"{te_acc:.3f}",
        )
        if log_fn and (epoch == 1 or epoch == cfg.epochs or epoch % 10 == 0):
            # tqdm.write keeps the log line above the live progress bar.
            tqdm.write(
                f"  epoch {epoch:>3}/{cfg.epochs}  "
                f"train_loss={tr_loss:.4f} train_acc={tr_acc:.3f}  "
                f"test_loss={te_loss:.4f} test_acc={te_acc:.3f}"
            )
    epoch_bar.close()
    result.wall_time_s = perf_counter() - start
    return result
