from rl.normalizer.reward._reward_normalizer import RewardNormalizer
from rl.normalizer.reward.dummy import DummyRewardNormalizer
from rl.normalizer.reward.rms import RMSRewardNormalizer


def get_reward_normalizer(config) -> RewardNormalizer:
    """
    Get the reward normalizer according to the configuration (config.reward_normalization_mode).
    """
    mode = config.reward_normalization_mode
    if mode == "rms":
        return RMSRewardNormalizer(discount_factor=config.discount_factor)
    elif mode == "none":
        return DummyRewardNormalizer(discount_factor=config.discount_factor)
    else:
        raise ValueError(f"get_reward_normalizer(): unknown reward_normalization mode: {mode}")