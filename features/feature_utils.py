import numpy as np

from features.feature_presets.edge_presets import get_edge_features
from features.feature_presets.node_presets import get_node_features
from features.feature_presets.global_presets import get_global_features
from features.features import ALL_GLOBAL_FEATURES, ALL_EDGE_FEATURES, ALL_NODE_FEATURES


"""
These are the metrics that are logged separately to evaluate the performance of the network.
"""
highlight_metrics = ["sentMB", "receivedMB", "droppedMB", "retransmittedMB",
                     "avgPacketDelay_ms", "maxPacketDelay_ms", "avgPacketJitter_ms",
                     "maxLU", "dropRatio", "oscillation_ratio_global", "next_hop_spread_global",
                     "maxSentMBPerStep", "maxReceivedMBPerStep"]


def is_performance_feature(name, loc) -> bool:
    """
    Returns whether the given feature is a performance feature.
    """
    if loc == "global":
        if name in ALL_GLOBAL_FEATURES:
            return ALL_GLOBAL_FEATURES[name][1]
    elif loc == "node":
        if name in ALL_NODE_FEATURES:
            return ALL_NODE_FEATURES[name][1]
    elif loc == "edge":
        if name in ALL_EDGE_FEATURES:
            return ALL_EDGE_FEATURES[name][1]
    else:
        raise ValueError(f"invalid loc: {loc}")
    return False  # fall-through for when the feature is not found (e.g. features not applicable to the policy)


def get_acceptable_features(config):
    """
    Returns those features for the given config that shall be used by the policy.
    """

    node_features = get_node_features(config.node_features)
    edge_features = get_edge_features(config.edge_features)
    global_features = get_global_features(config.global_features)

    if not config.use_flow_control:
        node_features = [f for f in node_features if 'queueDisc' not in f]
        edge_features = [f for f in edge_features if 'queueDisc' not in f]
        global_features = [f for f in global_features if 'queueDisc' not in f]

    if config.link_weights_as_input:
        edge_features.append("linkWeight")

    return {
        "node": node_features,
        "edge": edge_features,
        "global": global_features
    }


metric_aggregators = {
    "min": np.min,
    "avg": np.average,
    "max": np.max,
    "sum": np.sum,
}


def get_default_aggregator(metric_name: str):
    if "min" in metric_name:
        return metric_aggregators["min"]
    elif "avg" in metric_name:
        return metric_aggregators["avg"]
    elif "max" in metric_name:
        return metric_aggregators["max"]
    else:
        return metric_aggregators["sum"]


def aggregate_metrics(metric_values: list, metric_name: str = None, aggregator_str: str = None):
    if aggregator_str is None:
        if metric_name is None:
            raise ValueError("invoked aggregate_metrics without metric_name or aggregator_str")
        aggregator = get_default_aggregator(metric_name)
    elif aggregator_str in metric_aggregators.keys():
        aggregator = metric_aggregators[aggregator_str]
    else:
        raise ValueError(f"invalid aggregator_str: {aggregator_str}")
    return aggregator(metric_values)
