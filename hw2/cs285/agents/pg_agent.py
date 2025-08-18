from typing import Optional, Sequence
import numpy as np
import torch

from cs285.networks.policies import MLPPolicyPG
from cs285.networks.critics import ValueCritic
from cs285.infrastructure import pytorch_util as ptu
from torch import nn


class PGAgent(nn.Module):
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        gamma: float,
        learning_rate: float,
        use_baseline: bool,
        use_reward_to_go: bool,
        baseline_learning_rate: Optional[float],
        baseline_gradient_steps: Optional[int],     # 通常，actor (策略) 的更新只需要一步，因为策略梯度是 on-policy 的，用过一次的数据就要丢掉。但是，用于训练 critic 的数据 (s_t, R_t) 可以在当前批次中被重复使用来更好地拟合值函数，而不会引入偏差。self.baseline_gradient_steps（比如设为5）就是一个超参数，它允许我们用同一批数据对 critic 网络进行多次梯度下降更新。这有助于 critic 更快、更稳定地收敛到目标值，从而为 actor 提供更准确的基线，最终帮助 actor 学得更好。
        gae_lambda: Optional[float],
        normalize_advantages: bool,
    ):
        super().__init__()

        # create the actor (policy) network
        self.actor = MLPPolicyPG(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate
        )

        # create the critic (baseline) network, if needed
        if use_baseline:
            self.critic = ValueCritic(
                ob_dim, n_layers, layer_size, baseline_learning_rate
            )
            self.baseline_gradient_steps = baseline_gradient_steps
        else:
            self.critic = None

        # other agent parameters
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

    def update(
        self,
        obs: Sequence[np.ndarray],
        actions: Sequence[np.ndarray],
        rewards: Sequence[np.ndarray],
        terminals: Sequence[np.ndarray],
    ) -> dict:
        """The train step for PG involves updating its actor using the given observations/actions and the calculated
        qvals/advantages that come from the seen rewards.

        Each input is a list of NumPy arrays, where each array corresponds to a single trajectory. The batch size is the
        total number of samples across all trajectories (i.e. the sum of the lengths of all the arrays).
        """

        # step 1: calculate Q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards)

        # TODO: flatten the lists of arrays into single arrays, so that the rest of the code can be written in a vectorized
        # way. obs, actions, rewards, terminals, and q_values should all be arrays with a leading dimension of `batch_size`
        # beyond this point.
        obs = np.concatenate(obs)               # (batch_size, dim) <- (len1 + len2 + ..., ob_dim) <- (len1, ob_dim) + (len2, ob_dim) + ...
        actions = np.concatenate(actions)       # (batch_size, dim)
        rewards = np.concatenate(rewards)       # (batch_size,)     <- (len1 + len2 + ...) <- (len1,) + (len2,) + ...
        terminals = np.concatenate(terminals)   # (batch_size,)
        q_values = np.concatenate(q_values)     # (batch_size,)

        # step 2: calculate advantages from Q values
        advantages: np.ndarray = self._estimate_advantage(
            obs, rewards, q_values, terminals
        )

        # step 3: use all datapoints (s_t, a_t, adv_t) to update the PG actor/policy
        # TODO: update the PG actor/policy network once using the advantages 
        info: dict = self.actor.update(obs, actions, advantages)    # 更新 actor，只更新一次，因为策略梯度是 on-policy 算法，理论上用过一次的数据就应该丢弃，以避免偏差。

        # step 4: if needed, use all datapoints (s_t, a_t, q_t) to update the PG critic/baseline
        if self.critic is not None:
            # TODO: perform `self.baseline_gradient_steps` updates to the critic/baseline network
            for _ in range(self.baseline_gradient_steps):           # 更新 critic，用一个 for 循环更新 critic 多次，critic 的学习是一个标准的监督学习回归任务：输入是状态 s，目标是拟合 q_values (也就是 Reward-To-Go)。对于一个固定的数据集 (s, q_values)，我们可以反复训练模型来更好地拟合它，这和我们用同一批图片训练一个图像分类器很多个 epoch 是一个道理。
                critic_info: dict = self.critic.update(obs, q_values)

            info.update(critic_info)

        return info

    def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """Monte Carlo estimation of the Q function."""

        if not self.use_reward_to_go:
            # Case 1: in trajectory-based PG, we ignore the timestep and instead use the discounted return for the entire
            # trajectory at each point.
            # In other words: Q(s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}
            # TODO: use the helper function self._discounted_return to calculate the Q-values
            q_values = [self._discounted_return(reward) for reward in rewards]
        else:
            # Case 2: in reward-to-go PG, we only use the rewards after timestep t to estimate the Q-value for (s_t, a_t).
            # In other words: Q(s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
            # TODO: use the helper function self._discounted_reward_to_go to calculate the Q-values
            q_values = [self._discounted_reward_to_go(reward) for reward in rewards]
        return q_values

    def _estimate_advantage(
        self,
        obs: np.ndarray,
        rewards: np.ndarray,
        q_values: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        """Computes advantages by (possibly) subtracting a value baseline from the estimated Q-values.

        Operates on flat 1D NumPy arrays.
        """
        if self.critic is None: 
            # TODO: if no baseline, then what are the advantages?
            advantages = q_values
        else:    # 使用Critic网络估计V，为了降低方差
            # TODO: run the critic and use it as a baseline
            values = np.squeeze(ptu.to_numpy(self.critic(ptu.from_numpy(obs))))
            assert values.shape == q_values.shape

            if self.gae_lambda is None:     # 不用GAE，低偏差，高方差
                # TODO: if using a baseline, but not GAE, what are the advantages?
                advantages = q_values - values
            else:   # 用GAE，低方差，高偏差
                # TODO: implement GAE
                batch_size = obs.shape[0]

                # HINT: append a dummy T+1 value for simpler recursive calculation
                values = np.append(values, [0])
                advantages = np.zeros(batch_size + 1)

                for i in reversed(range(batch_size)):
                    # TODO: recursively compute advantage estimates starting from timestep T.
                    # HINT: use terminals to handle edge cases. terminals[i] is 1 if the state is the last in its
                    # trajectory, and 0 otherwise.
                    # A_GAE = delta_t + gamma*lambda*A_GAE(s_t+1,a_t+1)
                    # delta_t = r_t + gamma*V(s_t+1) − V(s_t) 
                    if terminals[i] != 1:
                        advantages[i] = (rewards[i] + self.gamma * values[i+1] - values[i]) + self.gamma * self.gae_lambda * advantages[i+1]
                    else:
                        advantages[i] = rewards[i] - values[i]

                # remove dummy advantage
                advantages = advantages[:-1]

        # TODO: normalize the advantages to have a mean of zero and a standard deviation of one within the batch
        if self.normalize_advantages:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        return advantages

    def _discounted_return(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns
        a list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}

        Note that all entries of the output list should be the exact same because each sum is from 0 to T (and doesn't
        involve t)!
        """
        total_return = 0
        for i in range(len(rewards)):
            total_return += self.gamma ** i * rewards[i]
        return [total_return] * len(rewards)


    def _discounted_reward_to_go(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns a list where the entry
        in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}.
        """
        len_rewards = len(rewards)
        discounted_reward_to_go = [0] * len_rewards
        discounted_reward_to_go[len_rewards - 1] = rewards[len_rewards - 1]
        for i in range(len_rewards-2, -1, -1):
            discounted_reward_to_go[i] = rewards[i] + self.gamma * discounted_reward_to_go[i+1]
        return discounted_reward_to_go
