from ._link_weight_actor import LinkWeightActor
from .magnneto_slim import MagnnetoSlimActor
from .magnneto_like import MagnnetoLikeActor


def build_link_weight_actor(nn_config, feature_counts, actor_mode, device) -> LinkWeightActor:
    """
    Returns a link-weight actor model instantiation depending on the given actor_mode.
    """

    # IMPORTANT: swaps node and edge features, since LinkWeightActor modules operate on the line graph.
    # The regular input graph is converted into a line graph in the LinkWeightActor module.
    feature_counts = {
        "node": feature_counts["edge"],
        "edge": feature_counts["node"],
        "global": feature_counts["global"]
    }

    if actor_mode == "magnneto_slim":
        return MagnnetoSlimActor(nn_config, feature_counts, device)
    elif actor_mode == "magnneto_like":
        if feature_counts['node'] != MagnnetoLikeActor.NUM_FEATURES:
            raise ValueError(f"node feature count must be "
                             f"{MagnnetoLikeActor.NUM_FEATURES} for actor_mode 'magnneto_like' "
                             f"(is {feature_counts['node']})")
        return MagnnetoLikeActor(device)
    else:
        raise ValueError(f"build_edge_centric_actor(): unknown actor_mode: {actor_mode}")
