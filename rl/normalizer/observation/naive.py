import torch

from features.features import ALL_NODE_FEATURES, ALL_EDGE_FEATURES, ALL_GLOBAL_FEATURES
from rl.normalizer.observation._obs_normalizer import ObservationNormalizer
from utils.types import Data, Tensor


class NaiveObservationNormalizer(ObservationNormalizer):
    """
    Naive observation normalizer that normalizes large config features (e.g. datarate, delay) to [0, 1] range by
    dividing by the maximum observed value.
    """
    def __init__(self, acceptable_features):
        super().__init__()
        self._node_feat = acceptable_features["node"]
        self._edge_feat = acceptable_features["edge"]
        self._global_feat = acceptable_features["global"]

    def _normalize_large_config_feat(self, feat_values: Tensor, feat_type: str):
        """
        Normalizes large config features (e.g. datarate, delay) to [0, 1] range by dividing by the maximum value
        """
        if feat_type == "node":
            feat_info = [ALL_NODE_FEATURES[feat_name] for feat_name in self._node_feat]
        elif feat_type == "edge":
            feat_info = [ALL_EDGE_FEATURES[feat_name] for feat_name in self._edge_feat]
        else:  # "global"
            feat_info = [ALL_GLOBAL_FEATURES[feat_name] for feat_name in self._global_feat]
        is_large_feat = torch.tensor([data_range == "large" for (data_range, _) in feat_info], dtype=torch.bool)

        max_values = feat_values.max(dim=0).values
        min_values = feat_values.min(dim=0).values
        diff = max_values - min_values
        subtracted_values = feat_values - min_values
        rescaled_values = torch.where(condition=diff != 0,
                                      input=2 * (subtracted_values / diff) - 1,
                                      other=torch.zeros_like(subtracted_values)
                                      )
        normalized_values = torch.where(condition=is_large_feat,
                                        input=rescaled_values,
                                        other=feat_values
                                        )
        return normalized_values

    def reset(self, obs: Data) -> Data:
        return self.normalize(obs)

    def update_and_normalize(self, obs) -> Data:
        return self.normalize(obs)

    def _update(self, obs: Data):
        pass

    def normalize(self, obs: Data) -> Data:
        """
        Normalize the given observation separately for node, edge, and global features.
        """
        if len(obs.x) > 0:
            new_obs = self._normalize_large_config_feat(obs.x, "node")
            obs.__setattr__("x", new_obs)
        if len(obs.edge_attr) > 0:
            new_edge_attr = self._normalize_large_config_feat(obs.edge_attr, "edge")
            obs.__setattr__("edge_attr", new_edge_attr)
        if len(obs.u) > 0:
            new_u = self._normalize_large_config_feat(obs.u, "global")
            obs.__setattr__("u", new_u)
        return obs
