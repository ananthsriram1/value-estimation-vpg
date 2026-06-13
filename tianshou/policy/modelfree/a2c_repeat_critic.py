import math
import time
from typing import Any, Dict, List, Optional, Type

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tianshou.data import Batch, ReplayBuffer, to_torch_as
from tianshou.policy import PGPolicy
from tianshou.utils.metrics import explained_variance
from tianshou.utils.net.common import ActorCritic


class A2CRepeatPolicy(PGPolicy):
    """Implementation of Synchronous Advantage Actor-Critic. arXiv:1602.01783.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.nn.Module critic: the critic network. (s -> V(s))
    :param torch.optim.Optimizer optim: the optimizer for actor and critic network.
    :param dist_fn: distribution class for computing the action.
    :type dist_fn: Type[torch.distributions.Distribution]
    :param float discount_factor: in [0, 1]. Default to 0.99.
    :param float vf_coef: weight for value loss. Default to 0.5.
    :param float ent_coef: weight for entropy loss. Default to 0.01.
    :param float max_grad_norm: clipping gradients in back propagation. Default to
        None.
    :param float gae_lambda: in [0, 1], param for Generalized Advantage Estimation.
        Default to 0.95.
    :param bool reward_normalization: normalize estimated values to have std close to
        1. Default to False.
    :param int max_batchsize: the maximum size of the batch when computing GAE,
        depends on the size of available memory and the memory cost of the
        model; should be as large as possible within the memory constraint.
        Default to 256.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action), "tanh" (for applying tanh
        squashing) for now, or empty string for no bounding. Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).
    :param bool deterministic_eval: whether to use deterministic action instead of
        stochastic action sampled by the policy. Default to False.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        optim: torch.optim.Optimizer,
        dist_fn: Type[torch.distributions.Distribution],
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: Optional[float] = None,
        gae_lambda: float = 0.95,
        max_batchsize: int = 256,
        repeat_critic: int = 1,
        ev_gate: bool = False,
        ev_tau: float = 0.0,
        ev_delta: float = 0.0,
        ev_holdout_frac: float = 0.2,
        ev_gate_mode: str = "threshold",
        **kwargs: Any
    ) -> None:
        super().__init__(actor, optim, dist_fn, **kwargs)
        self.critic = critic
        assert 0.0 <= gae_lambda <= 1.0, "GAE lambda should be in [0, 1]."
        self._lambda = gae_lambda
        self._weight_vf = vf_coef
        self._weight_ent = ent_coef
        self._grad_norm = max_grad_norm
        self._batch = max_batchsize
        self._actor_critic = ActorCritic(self.actor, self.critic)
        self._repeat_critic = repeat_critic
        self._ev_gate = ev_gate
        self._ev_tau = ev_tau
        self._ev_delta = ev_delta
        self._ev_holdout_frac = ev_holdout_frac
        self._ev_gate_mode = ev_gate_mode
        self._cumulative_inner_steps = 0.0
        self._cumulative_inner_saved = 0.0

    @staticmethod
    def _holdout_mask(n: int, frac: float, device: torch.device) -> torch.Tensor:
        perm = torch.randperm(n, device=device)
        n_hold = max(1, int(n * frac))
        mask = torch.zeros(n, dtype=torch.bool, device=device)
        mask[perm[:n_hold]] = True
        return mask

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        batch = self._compute_returns(batch, buffer, indices)
        batch.act = to_torch_as(batch.act, batch.v_s)
        return batch

    def _compute_returns(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        v_s, v_s_ = [], []
        with torch.no_grad():
            for minibatch in batch.split(self._batch, shuffle=False, merge_last=True):
                v_s.append(self.critic(minibatch.obs))
                v_s_.append(self.critic(minibatch.obs_next))
        batch.v_s = torch.cat(v_s, dim=0).flatten()  # old value
        v_s = batch.v_s.cpu().numpy()
        v_s_ = torch.cat(v_s_, dim=0).flatten().cpu().numpy()
        # when normalizing values, we do not minus self.ret_rms.mean to be numerically
        # consistent with OPENAI baselines' value normalization pipeline. Emperical
        # study also shows that "minus mean" will harm performances a tiny little bit
        # due to unknown reasons (on Mujoco envs, not confident, though).
        if self._rew_norm:  # unnormalize v_s & v_s_
            v_s = v_s * np.sqrt(self.ret_rms.var + self._eps)
            v_s_ = v_s_ * np.sqrt(self.ret_rms.var + self._eps)
        unnormalized_returns, advantages = self.compute_episodic_return(
            batch,
            buffer,
            indices,
            v_s_,
            v_s,
            gamma=self._gamma,
            gae_lambda=self._lambda
        )
        if self._rew_norm:
            batch.returns = unnormalized_returns / \
                np.sqrt(self.ret_rms.var + self._eps)
            self.ret_rms.update(unnormalized_returns)
        else:
            batch.returns = unnormalized_returns
        batch.returns = to_torch_as(batch.returns, batch.v_s)
        batch.adv = to_torch_as(advantages, batch.v_s)
        return batch

    def learn(  # type: ignore
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        losses, actor_losses, vf_losses, ent_losses = [], [], [], []
        kv_effective, ev_last, ev_best_log, update_wall_s = [], [], [], []
        cum_kv, cum_saved = [], []
        k_max = float(self._repeat_critic)
        for _ in range(repeat):
            for minibatch in batch.split(batch_size, merge_last=True):
                t_update0 = time.perf_counter()
                # calculate loss for actor
                dist = self(minibatch).dist
                log_prob = dist.log_prob(minibatch.act)
                log_prob = log_prob.reshape(len(minibatch.adv), -1).transpose(0, 1)
                actor_loss = -(log_prob * minibatch.adv).mean()
                # calculate loss for critic
                vf_losses_repeat = []
                kv_used = 0
                ev_best = -float("inf")
                ev = float("nan")
                holdout = None
                if self._ev_gate:
                    holdout = self._holdout_mask(
                        len(minibatch.returns),
                        self._ev_holdout_frac,
                        minibatch.returns.device,
                    )
                for j in range(self._repeat_critic):
                    value = self.critic(minibatch.obs).flatten()
                    vf_loss = F.mse_loss(minibatch.returns, value)
                    self.optim.zero_grad()
                    vf_loss.backward()
                    if self._grad_norm:
                        nn.utils.clip_grad_norm_(
                            self.critic.parameters(), max_norm=self._grad_norm
                        )
                    self.optim.step()
                    vf_losses_repeat.append(vf_loss.item())
                    kv_used = j + 1

                    if self._ev_gate:
                        with torch.no_grad():
                            v_h = self.critic(minibatch.obs[holdout]).flatten()
                            y_h = minibatch.returns[holdout].flatten()
                            ev = explained_variance(y_h, v_h)
                        prev_best = ev_best
                        ev_best = max(ev_best, ev)
                        stop = False
                        if not math.isnan(ev):
                            if self._ev_gate_mode in ("threshold", "combo") and ev < self._ev_tau:
                                stop = True
                            if (
                                self._ev_delta > 0
                                and self._ev_gate_mode in ("delta", "combo")
                                and prev_best > -float("inf")
                                and ev < prev_best - self._ev_delta
                            ):
                                stop = True
                        if stop:
                            break
                # calculate regularization and overall loss
                ent_loss = dist.entropy().mean()
                loss = actor_loss - self._weight_ent * ent_loss
                self.optim.zero_grad()
                loss.backward()
                if self._grad_norm:  # clip large gradient
                    nn.utils.clip_grad_norm_(
                        self._actor_critic.parameters(), max_norm=self._grad_norm
                    )
                self.optim.step()
                actor_losses.append(actor_loss.item())
                vf_losses.append(sum(vf_losses_repeat) / len(vf_losses_repeat))
                ent_losses.append(ent_loss.item())
                losses.append(loss.item())
                kv_effective.append(float(kv_used))
                ev_last.append(ev if not math.isnan(ev) else 0.0)
                ev_best_log.append(ev_best if ev_best > -float("inf") else 0.0)
                update_wall_s.append(time.perf_counter() - t_update0)
                self._cumulative_inner_steps += float(kv_used)
                saved = k_max - float(kv_used)
                self._cumulative_inner_saved += saved
                cum_kv.append(self._cumulative_inner_steps)
                cum_saved.append(self._cumulative_inner_saved)

        result = {
            "loss": losses,
            "loss/actor": actor_losses,
            "loss/vf": vf_losses,
            "loss/ent": ent_losses,
            "train/kv_effective": kv_effective,
            "train/kv_frac": [k / k_max for k in kv_effective],
            "train/kv_saved": [k_max - k for k in kv_effective],
            "train/update_wall_s": update_wall_s,
            "train/cumulative_inner_steps": cum_kv,
            "train/cumulative_inner_saved": cum_saved,
        }
        if self._ev_gate:
            result["train/ev"] = ev_last
            result["train/ev_best"] = ev_best_log
        return result
