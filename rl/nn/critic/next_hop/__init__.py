from ._next_hop_critic import NextHopCritic
from .mpn import MPNNextHopCritic


def build_next_hop_critic(nn_config, feature_counts, critic_mode, value_scope, device) -> NextHopCritic:
    """
    Returns a next-hop critic architecture based on the given critic_mode
    """
    if critic_mode == "mpn":
        return MPNNextHopCritic(nn_config, feature_counts, value_scope, device)
    else:
        raise ValueError(f"build_node_centric_critic(): unknown critic_mode: {critic_mode}")
