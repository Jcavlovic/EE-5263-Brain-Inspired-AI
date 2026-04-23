"""Training / evaluation loops for the rate-based gender classifier."""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader

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
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_n = 0

    for x, y in loader:
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
    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, loss_fn, device, optimizer)
        te_loss, te_acc = _run_epoch(model, test_loader, loss_fn, device, None)
        result.stats.append(EpochStats(epoch, tr_loss, tr_acc, te_loss, te_acc))
        if log_fn and (epoch == 1 or epoch == cfg.epochs or epoch % 10 == 0):
            log_fn(
                f"  epoch {epoch:>3}/{cfg.epochs}  "
                f"train_loss={tr_loss:.4f} train_acc={tr_acc:.3f}  "
                f"test_loss={te_loss:.4f} test_acc={te_acc:.3f}"
            )
    result.wall_time_s = perf_counter() - start
    return result
