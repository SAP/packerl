from rl.normalizer._normalizer import AbstractNormalizer
from utils.types import Tensor


class RewardNormalizer(AbstractNormalizer):
    """
    Abstract class for reward normalizers.
    """
    def __init__(self, discount_factor: float, reward_clip: float = 5, epsilon: float = 1.0e-6):
        super().__init__(epsilon)
        self.discount_factor = discount_factor
        self.reward_clip = reward_clip

    def reset(self):
        """
        Reset the normalizer parameters.
        """
        raise NotImplementedError

    def update_and_normalize(self, reward: float):
        """
        Update the normalizer parameters with the given reward and return the normalized reward.
        """
        raise NotImplementedError

    def _update(self, reward):
        """
        Update the normalizer parameters with the given reward.
        """
        raise NotImplementedError

    def _normalize(self, reward: float) -> Tensor:
        """
        Normalize the given reward.
        """
        raise NotImplementedError
