# EVOLVE-BLOCK-START
"""Baseline fused bias + SiLU implementation."""

import torch


def fused_silu(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(x + bias)


# EVOLVE-BLOCK-END

