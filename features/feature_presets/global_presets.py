def get_global_features(feature_mode):
    """
    Returns a list of global features for the given feature_mode.
    """
    if feature_mode == "none":
        return GLOBAL_FEATURES_NONE
    elif feature_mode == "load":
        return GLOBAL_FEATURES_LOAD
    elif feature_mode == "dj":
        return GLOBAL_FEATURES_DJ
    elif feature_mode == "srdr":
        return GLOBAL_FEATURES_SRDR
    elif feature_mode == "load_dj":
        return GLOBAL_FEATURES_LOAD_DJ
    elif feature_mode == "load_srdr":
        return GLOBAL_FEATURES_LOAD_SRDR
    elif feature_mode == "dj_srdr":
        return GLOBAL_FEATURES_DJ_SRDR
    elif feature_mode == "load_dj_srdr":
        return GLOBAL_FEATURES_LOAD_DJ_SRDR
    else:
        raise ValueError("Unknown global feature mode: {}".format(feature_mode))


GLOBAL_FEATURES_NONE = []

GLOBAL_FEATURES_DJ = [
    "avgPacketDelay",
    "maxPacketDelay",
    "avgPacketJitter",
]

GLOBAL_FEATURES_SRDR = [
    "sentBytes",
    "receivedBytes",
    "droppedBytes",
    "retransmittedBytes",
]

GLOBAL_FEATURES_LOAD = [
    "maxLU",
    "avgTDU",
]

GLOBAL_FEATURES_LOAD_DJ = GLOBAL_FEATURES_DJ + GLOBAL_FEATURES_LOAD

GLOBAL_FEATURES_LOAD_SRDR = GLOBAL_FEATURES_LOAD + GLOBAL_FEATURES_SRDR

GLOBAL_FEATURES_DJ_SRDR = GLOBAL_FEATURES_DJ + GLOBAL_FEATURES_SRDR

GLOBAL_FEATURES_LOAD_DJ_SRDR = GLOBAL_FEATURES_DJ + GLOBAL_FEATURES_LOAD + GLOBAL_FEATURES_SRDR
