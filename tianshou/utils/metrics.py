"""RL training metrics."""

from typing import Union

import torch


def explained_variance(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """1 - Var(y - y_pred) / Var(y); SB3 / Baselines convention."""
    y_true = y_true.detach().flatten()
    y_pred = y_pred.detach().flatten()
    var_y = y_true.var()
    if var_y.item() < eps:
        return float("nan")
    var_residual = (y_true - y_pred).var()
    return (1.0 - var_residual / (var_y + eps)).item()
