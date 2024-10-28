from rl.normalizer.reward._reward_normalizer import RewardNormalizer


class DummyRewardNormalizer(RewardNormalizer):
    """
    Dummy reward normalizer that does nothing.
    """
    def __init__(self, discount_factor: float, reward_clip: float = 5, epsilon: float = 1.0e-6):
        super().__init__(discount_factor, reward_clip, epsilon)

    def reset(self):
        pass

    def update_and_normalize(self, reward: float = None):
        return reward

    def _update(self, reward):
        pass

    def _normalize(self, reward: float):
        return reward
