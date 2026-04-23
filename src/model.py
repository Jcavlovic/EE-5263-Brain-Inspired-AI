"""Rate-based feed-forward network.

Each unit's output is a continuous sigmoid "firing rate" in [0, 1], so every
layer (hidden and output) uses a sigmoid nonlinearity — this mirrors the
classical rate-coded neuron model. Sparsity is applied as a fixed binary mask
on each weight matrix, zeroing out a given fraction of synapses at init and
keeping them zero during training.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class ModelConfig:
    input_dim: int = 64 * 64
    hidden_layers: int = 2
    neurons_per_layer: int = 256
    sparsity: float = 0.2        # fraction of weights forced to zero
    seed: int = 42


class _SparseLinear(nn.Module):
    """Linear layer with a fixed sparsity mask.

    Uses a Normal weight init with std = 1 / sqrt(fan_in_kept) so that
    pre-activations stay in sigmoid's responsive range (otherwise stacked
    sigmoids saturate and gradients vanish). `fan_in_kept` accounts for the
    sparsity mask: with X% of synapses zeroed, only (1-X) of them carry signal.
    """

    def __init__(self, in_features: int, out_features: int, sparsity: float,
                 generator: torch.Generator):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        keep = max(1.0 - sparsity, 1e-6)
        # Scale std by effective fan-in after masking.
        std = 1.0 / (keep * in_features) ** 0.5

        with torch.no_grad():
            self.weight.normal_(mean=0.0, std=std, generator=generator)
            if sparsity > 0.0:
                mask = (torch.rand(self.weight.shape, generator=generator) < keep).float()
                self.weight.mul_(mask)
            else:
                mask = torch.ones_like(self.weight)

        # Keep the mask out of trainable params; register as a buffer so it
        # moves with .to(device) and is re-applied after every optimiser step.
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight * self.mask, self.bias)

    @torch.no_grad()
    def reapply_mask(self) -> None:
        self.weight.mul_(self.mask)


class RateBasedNN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        gen = torch.Generator().manual_seed(cfg.seed)

        dims = [cfg.input_dim] + [cfg.neurons_per_layer] * cfg.hidden_layers + [1]
        self.layers = nn.ModuleList([
            _SparseLinear(dims[i], dims[i + 1], cfg.sparsity, gen)
            for i in range(len(dims) - 1)
        ])
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sigmoid on every layer (rate coding): hidden layers output rates in
        # [0, 1], and the output is a rate interpreted as P(class = female).
        for layer in self.layers:
            x = self.activation(layer(x))
        return x.squeeze(-1)

    def reapply_masks(self) -> None:
        for layer in self.layers:
            layer.reapply_mask()

    def active_fraction(self) -> float:
        """Fraction of synapses that are still non-zero (sanity check)."""
        total = 0
        active = 0
        for layer in self.layers:
            total += layer.mask.numel()
            active += int(layer.mask.sum().item())
        return active / max(total, 1)
