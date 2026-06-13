"""Training profile hooks + per-run JSON summary for focused experiments."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

from torch.utils.tensorboard import SummaryWriter


def make_epoch_profile_hooks(
    writer: SummaryWriter,
    step_per_epoch: int,
) -> tuple[Callable[[int, int], None], Callable[[int, Optional[int]], None]]:
    """Log per-epoch wall time and env-step throughput after each test phase."""
    epoch_t0 = [time.perf_counter()]

    def train_fn(epoch: int, env_step: int) -> None:
        epoch_t0[0] = time.perf_counter()

    def test_fn(epoch: int, env_step: Optional[int]) -> None:
        if env_step is None:
            return
        dt = max(time.perf_counter() - epoch_t0[0], 1e-9)
        writer.add_scalar("train/epoch_wall_s", dt, env_step)
        writer.add_scalar("train/steps_per_sec", step_per_epoch / dt, env_step)
        writer.flush()

    return train_fn, test_fn


def chain_test_fn(
    *fns: Optional[Callable[[int, Optional[int]], None]],
) -> Optional[Callable[[int, Optional[int]], None]]:
    active = [f for f in fns if f is not None]
    if not active:
        return None

    def test_fn(epoch: int, env_step: Optional[int]) -> None:
        for fn in active:
            fn(epoch, env_step)

    return test_fn


def write_run_summary(
    log_path: str,
    meta: dict[str, Any],
    trainer_result: dict[str, Any],
    final_eval: dict[str, Any],
    wall_s: float,
) -> Path:
    """Write machine-readable summary next to tensorboard run."""
    out = Path(log_path) / "run_summary.json"
    payload = {
        **meta,
        "wall_s": wall_s,
        "trainer": {
            k: (float(v) if isinstance(v, (int, float)) else v)
            for k, v in trainer_result.items()
            if k in ("best_reward", "best_epoch", "train_step", "train_episode", "train_time")
        },
        "final_eval": final_eval,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return out
