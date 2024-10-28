from ._next_hop import NextHopActor
from .fieldlines import FieldLinesActor


def build_next_hop_actor(nn_config, feature_counts, actor_mode, device) -> NextHopActor:
    """
    Returns a next-hop actor model instantiation depending on the given actor_mode.
    """
    if actor_mode == "fieldlines_no_spdist":
        return FieldLinesActor(nn_config, feature_counts, device, use_spdist=False)
    elif actor_mode == "fieldlines":
        return FieldLinesActor(nn_config, feature_counts, device, use_spdist=True)
    else:
        raise ValueError(f"build_next_hop_actor(): unknown actor_mode: {actor_mode}")
