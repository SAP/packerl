def get_edge_features(feature_mode):
    """
    Returns a list of edge features for the given feature_mode.
    """
    if feature_mode == "none":
        return EDGE_FEATURES_NONE
    elif feature_mode == "LU":
        return EDGE_FEATURES_LU
    elif feature_mode == "load":
        return EDGE_FEATURES_LOAD
    elif feature_mode == "config":
        return EDGE_FEATURES_CONFIG
    elif feature_mode == "srd":
        return EDGE_FEATURES_SRD
    elif feature_mode == "load_config":
        return EDGE_FEATURES_LOAD_CONFIG
    elif feature_mode == "load_srd":
        return EDGE_FEATURES_LOAD_SRD
    elif feature_mode == "config_srd":
        return EDGE_FEATURES_CONFIG_SRD
    elif feature_mode == "load_config_srd":
        return EDGE_FEATURES_LOAD_CONFIG_SRD
    else:
        raise ValueError("Unknown edge feature mode: {}".format(feature_mode))


EDGE_FEATURES_NONE = []

EDGE_FEATURES_LU = [ "LU" ]

EDGE_FEATURES_LOAD = [
    "LU", "txQueueLastLoad", "txQueueMaxLoad", "queueDiscLastLoad", "queueDiscMaxLoad"
]

EDGE_FEATURES_CONFIG = [
    "txQueueCapacity", "queueDiscCapacity", "channelDelay", "channelDataRate"
]

EDGE_FEATURES_SRD = [
    "sentBytes", "receivedBytes", "droppedBytes"
]

EDGE_FEATURES_LOAD_CONFIG = EDGE_FEATURES_LOAD + EDGE_FEATURES_CONFIG

EDGE_FEATURES_LOAD_SRD = EDGE_FEATURES_LOAD + EDGE_FEATURES_SRD

EDGE_FEATURES_CONFIG_SRD = EDGE_FEATURES_CONFIG + EDGE_FEATURES_SRD

EDGE_FEATURES_LOAD_CONFIG_SRD = EDGE_FEATURES_LOAD + EDGE_FEATURES_CONFIG + EDGE_FEATURES_SRD
