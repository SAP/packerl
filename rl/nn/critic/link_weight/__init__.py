from ._link_weight_critic import LinkWeightCritic
from .mpn import MPNLinkWeightCritic
from .magnneto_like import MagnnetoLikeCritic


def build_link_weight_critic(nn_config, feature_counts, critic_mode, value_scope, device) -> LinkWeightCritic:
    """
    Returns a link-weight critic architecture depending on the given critic_mode.
    """

    # Swap node and edge features, since LinkWeightActorCritic modules operate on the line graph.
    # The regular input graph is converted into a line graph in the LinkWeightActorCritic module.
    feature_counts = {
        "node": feature_counts["edge"],
        "edge": feature_counts["node"],
        "global": feature_counts["global"]
    }

    if critic_mode == "mpn":
        return MPNLinkWeightCritic(nn_config, feature_counts, value_scope, device)
    elif critic_mode == "magnneto_like":
        if feature_counts['node'] != MagnnetoLikeCritic.NUM_FEATURES:
            raise ValueError(f"node feature count must be "
                             f"{MagnnetoLikeCritic.NUM_FEATURES} for critic_mode 'magnneto_like' "
                             f"(is {feature_counts['node']})")
        return MagnnetoLikeCritic(device)
    else:
        raise ValueError(f"build_link_weight_critic(): unknown critic_mode: {critic_mode}")
