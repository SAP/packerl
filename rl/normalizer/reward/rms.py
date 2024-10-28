import torch

from rl.normalizer.running_mean_std import TorchRunningMeanStd
from rl.normalizer.reward._reward_normalizer import RewardNormalizer
from utils.types import Tensor


class RMSRewardNormalizer(RewardNormalizer):
    """
    Reward normalizer that normalizes the rewards using the RunningMeanStd approach.
    """
    def __init__(self, discount_factor: float, reward_clip: float = 5, epsilon: float = 1.0e-6):
        super().__init__(discount_factor, reward_clip, epsilon)
        self.reward_normalizer = TorchRunningMeanStd(epsilon=epsilon, shape=(1,))
        self.returns = torch.tensor([0])

    def reset(self):
        self.returns = torch.tensor([0])

    def update_and_normalize(self, reward: Tensor):
        self._update_reward(reward)
        reward = self._normalize_reward(reward)
        return reward

    def _update_reward(self, reward: Tensor):
        # normalize reward according to PPO update rule. Taken from stable_baselines3
        self.returns = self.returns * self.discount_factor + torch.mean(reward)
        # we just take the mean over rewards for now, as there may be a different number of rewards each step
        self.reward_normalizer.update(self.returns)

    def _normalize_reward(self, reward: Tensor) -> Tensor:
        scaled_reward = reward / torch.sqrt(self.reward_normalizer.var + self.epsilon)
        scaled_reward = torch.clip(scaled_reward, -self.reward_clip, self.reward_clip)[0]
        return scaled_reward
