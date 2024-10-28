from rl.normalizer.observation._obs_normalizer import ObservationNormalizer
from rl.normalizer.observation.dummy import DummyObservationNormalizer
from rl.normalizer.observation.rms import RMSObservationNormalizer
from rl.normalizer.observation.naive import NaiveObservationNormalizer


def get_obs_normalizer(config, acceptable_features, feature_counts) -> ObservationNormalizer:
    """
    Get the observation normalizer according to the configuration (config.obs_normalization_mode).
    """
    mode = config.obs_normalization_mode
    if mode == "rms":
        return RMSObservationNormalizer(num_node_features=feature_counts["node"],
                                        num_edge_features=feature_counts["edge"],
                                        num_global_features=feature_counts["global"])
    elif mode == "naive":
        return NaiveObservationNormalizer(acceptable_features)
    elif mode == "none":
        return DummyObservationNormalizer()
    else:
        raise ValueError(f"get_obs_normalizer(): unknown obs_normalization mode: {mode}")