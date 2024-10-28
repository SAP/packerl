import torch

from reward.reward_functions import global_reward_functions, local_reward_functions, get_reward_preset


class RewardModule:
    """
    This module encapsulates the reward computation from given network states and actions.
    """
    def __init__(self, reward_preset_str, global_reward_ratio):

        assert 0.0 <= global_reward_ratio <= 1.0
        self.global_reward_ratio = global_reward_ratio

        # get reward items
        reward_preset = get_reward_preset(reward_preset_str)
        self._global_r_items = [(r_func, coeff) for r_func, coeff in reward_preset.items()
                                if r_func in global_reward_functions]
        self._local_r_items = [(r_func, coeff) for r_func, coeff in reward_preset.items()
                               if r_func in local_reward_functions]

    def collect_reward(self, reward_input: dict) -> dict:
        """
        Calculate the reward for the agents based on the current network state and actions.
        Depending on the reward preset, different reward functions are used.
        Depending on the global_reward_ratio, the global and local rewards are weighted differently.
        """
        raw_reward = {}

        global_reward = torch.tensor(0, dtype=torch.float)  # dtype=torch.float32
        for r_func, r_coeff in self._global_r_items:
            gr = r_func(**reward_input)
            raw_reward[r_func.__name__] = gr
            global_reward += r_coeff * gr

        local_reward = torch.zeros((reward_input["actions"].num_nodes,), dtype=torch.float)
        for r_func, r_coeff in self._local_r_items:
            lr = r_func(**reward_input)
            raw_reward[r_func.__name__] = lr
            local_reward += r_coeff * lr

        mixed_reward = (self.global_reward_ratio * global_reward * torch.ones_like(local_reward)
                        + (1 - self.global_reward_ratio) * local_reward)

        return {
            "global": global_reward,
            "local": local_reward,
            "mixed": mixed_reward,
            "all": raw_reward
        }
