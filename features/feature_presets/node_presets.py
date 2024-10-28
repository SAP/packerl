def get_node_features(feature_mode):
    """
    Returns a list of node features for the given feature_mode.
    """
    if feature_mode == "none":
        return NODE_FEATURES_NONE
    elif feature_mode == "srr":
        return NODE_FEATURES_SRR
    elif feature_mode == "retransmitted":
        return NODE_FEATURES_RETRANSMITTED
    else:
        raise ValueError("Unknown node feature mode: {}".format(feature_mode))


NODE_FEATURES_NONE = []

NODE_FEATURES_SRR = [
    "receivedBytes", "sentBytes", "retransmittedBytes"
]

NODE_FEATURES_RETRANSMITTED = [
    "retransmittedBytes"
]
