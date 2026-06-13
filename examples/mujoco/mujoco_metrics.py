"""Wang-style evaluation metrics for MuJoCo training scripts."""
from __future__ import annotations

from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Batch


def _apply_obs_norm(obs: np.ndarray, obs_rms: Any) -> np.ndarray:
    if obs_rms is None:
        return obs
    return obs_rms.norm(obs)


def compute_eta_s0(
    policy: Any,
    task: str,
    obs_rms: Any,
    gamma: float,
    *,
    n_episodes: int = 10,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[float, float]:
    """η(s₀) = Σ γᵗ rₜ − V(s₀) with deterministic policy (Wang Eq. 10)."""
    policy.eval()
    prev_det = getattr(policy, "_deterministic_eval", False)
    policy._deterministic_eval = True
    env = gym.make(task)
    etas: list[float] = []
    try:
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + ep)
            obs = _apply_obs_norm(np.asarray(obs, dtype=np.float32), obs_rms)
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                v0 = policy.critic(obs_t).flatten().item()
            discounted_return = 0.0
            t = 0
            terminated = truncated = False
            while not (terminated or truncated):
                batch = Batch(obs=obs_t, info={})
                act = policy(batch).act
                if isinstance(act, torch.Tensor):
                    act_np = act.detach().cpu().numpy().reshape(-1)
                else:
                    act_np = np.asarray(act).reshape(-1)
                obs, reward, terminated, truncated, _ = env.step(act_np)
                obs = _apply_obs_norm(np.asarray(obs, dtype=np.float32), obs_rms)
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                discounted_return += (gamma ** t) * float(reward)
                t += 1
            etas.append(discounted_return - v0)
    finally:
        policy._deterministic_eval = prev_det
        env.close()
    arr = np.asarray(etas, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0)


def make_eta_test_fn(
    policy: Any,
    task: str,
    train_envs: Any,
    gamma: float,
    writer: SummaryWriter,
    *,
    n_episodes: int = 10,
    seed: int = 0,
    device: str = "cpu",
) -> Callable[[int, Optional[int]], None]:
    """Tensorboard hook: log test/eta_s0 each test phase."""

    def test_fn(epoch: int, env_step: Optional[int]) -> None:
        if env_step is None:
            return
        mean_eta, std_eta = compute_eta_s0(
            policy,
            task,
            train_envs.get_obs_rms(),
            gamma,
            n_episodes=n_episodes,
            seed=seed,
            device=device,
        )
        writer.add_scalar("test/eta_s0", mean_eta, env_step)
        writer.add_scalar("test/eta_s0_std", std_eta, env_step)
        writer.flush()

    return test_fn
