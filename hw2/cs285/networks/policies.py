import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        # TODO: implement get_action
        if obs.ndim == 1:
            obs = obs[None, :]
        obs_tensor = ptu.from_numpy(obs)
        actions_tensor = self.forward(obs_tensor).sample()
        actions_numpy = ptu.to_numpy(actions_tensor)
        return actions_numpy

    def forward(self, obs: torch.FloatTensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            # TODO: define the forward pass for a policy with a discrete action space.
            logits = self.logits_net(obs)
            policy_distribution = distributions.Categorical(logits=logits)
        else:
            # TODO: define the forward pass for a policy with a continuous action space.
            mean = self.mean_net(obs)
            std = torch.exp(self.logstd)
            # policy_distribution = distributions.normal.Normal(mean, std)  错误的，
            # 在 HalfCheetah-v4 环境中，动作空间维度是6。所以 mean_net 输出的 mean 张量形状是 (5000, 6)。
            # 在实验二Cheetah环境中我们想要的不是 5000 * 6 个独立的分布，而是 5000 个6维的、各向同性 (diagonal covariance) 的高斯分布。我们希望 .log_prob() 计算的是一个6维动作向量的总对数概率（即6个维度对数概率的和），最终返回一个形状为 (5000,) 的张量。
            # 当我们用一个形状为 (5000, 6) 的 mean 创建一个 distributions.Normal 对象时，PyTorch 会把它理解为一批 (a batch of) 5000 * 6 个完全独立的、一维的正态分布。
            # 因此，当调用 .log_prob(actions) 时（其中 actions 的形状是 (5000, 6)），它会为这 5000 * 6 个点中的每一个都独立计算对数概率，最终返回一个形状为 (5000, 6) 的张量。
            base_distribution = distributions.normal.Normal(mean, std)
            policy_distribution = distributions.Independent(base_distribution, 1)
            # 把 base_distribution 的最后一个维度（由 1 指定）看作是一个多维事件 (event) 的各个部分，而不是独立的批次数据。这样修改后，当再次调用 .log_prob(actions) 时，它就会正确地将6个动作维度的对数概率加起来，为每个样本返回一个标量，最终得到的 log_probs 张量形状就是 (5000,)。这样，它就可以和 advantages 张量正常相乘.
        return policy_distribution

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # TODO: implement the policy gradient actor update.
        policy_distribution = self.forward(obs)
        log_probs = policy_distribution.log_prob(actions)
        loss = - (log_probs * advantages).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "Actor Loss": ptu.to_numpy(loss),
        }
