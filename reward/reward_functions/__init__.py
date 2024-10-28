from .global_rewards import (dummy_reward,
                             max_LU,
                             drop_ratio,
                             avg_delay,
                             max_delay,
                             weighted_delay,
                             avg_packet_jitter,
                             avg_routing_loops_per_node,
                             received_MB,
                             retransmitted_MB)
from .local_rewards import routing_loops_per_node


global_reward_functions = [
    dummy_reward,
    max_LU,
    drop_ratio,
    avg_delay,
    max_delay,
    weighted_delay,
    avg_packet_jitter,
    avg_routing_loops_per_node,
    received_MB,
    retransmitted_MB,
]

local_reward_functions = [
    routing_loops_per_node,
]


def get_reward_preset(reward_preset) -> dict:
    """
    The reward preset determines which reward functions are used. Each of its letters corresponds to a reward function.
    These reward functions come with their own weights, which are used to calculate the final reward.
    """
    reward_preset_dict = dict()
    if "l" in reward_preset:
        reward_preset_dict |= REWARD_PRESET_LU
    if "r" in reward_preset:
        reward_preset_dict |= REWARD_PRESET_RECEIVED
    if "d" in reward_preset:
        reward_preset_dict |= REWARD_PRESET_DROP_RATIO
    if "w" in reward_preset:
        reward_preset_dict |= REWARD_PRESET_WEIGHTED_DELAY
    if "a" in reward_preset:
        reward_preset_dict |= REWARD_PRESET_AVG_DELAY
    if "j" in reward_preset:
        reward_preset_dict |= REWARD_PRESET_JITTER
    if "t" in reward_preset:
        reward_preset_dict |= REWARD_PRESET_RETRANSMITTED

    return reward_preset_dict or REWARD_PRESET_DUMMY  # if empty, return dummy


REWARD_PRESET_DUMMY = {
    dummy_reward: -1.0,
}
REWARD_PRESET_RECEIVED = {
    received_MB: 2,
}
REWARD_PRESET_DROP_RATIO = {
    drop_ratio: -0.5,
}
REWARD_PRESET_AVG_DELAY = {
    avg_delay: -10,
}
REWARD_PRESET_WEIGHTED_DELAY = {
    weighted_delay: -10,
}
REWARD_PRESET_JITTER = {
    avg_packet_jitter: -500,
}
REWARD_PRESET_RETRANSMITTED = {
    retransmitted_MB: -1,
}
REWARD_PRESET_LU = {
    max_LU: -0.1,
}
