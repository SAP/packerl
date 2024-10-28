import torch

from rl.normalizer.running_mean_std import TorchRunningMeanStd
from rl.normalizer.observation._obs_normalizer import ObservationNormalizer
from utils.types import Data, Tensor


class RMSObservationNormalizer(ObservationNormalizer):
    """
    Observation normalizer that normalizes the observations of a graph environment using the RunningMeanStd approach.
    """
    def __init__(self, num_node_features: int, num_edge_features: int, num_global_features: int,
                 observation_clip: float = 10, epsilon: float = 1.0e-6):
        """
        Normalizes the observations of a graph environment
        Args:
            graph_environment: the graph environment to normalize the observations of
            normalize_nodes: whether to normalize the node features
            normalize_edges: whether to normalize the edge features
            normalize_globals: whether to normalize the global features
            observation_clip: the maximum absolute value of the normalized observations
            epsilon: a small value to add to the variance to avoid division by zero

        """
        super().__init__(obs_clip=observation_clip, epsilon=epsilon)
        self.node_normalizers = TorchRunningMeanStd(epsilon=epsilon, shape=(num_node_features,))
        self.edge_normalizers = TorchRunningMeanStd(epsilon=epsilon, shape=(num_edge_features,))
        self.global_normalizer = TorchRunningMeanStd(epsilon=epsilon, shape=(num_global_features,))


    def reset(self, observations: Data) -> Data:
        """
        To be called after the reset() of the environment is called. Used to update the normalizer statistics
        with the initial observations of the environment, and potentially reset parts of the normalizer that
        depend on the environment episode
        Args:
            observations: the initial observations of the environment

        Returns: the normalized observations

        """
        self._update(obs=observations)
        return self.normalize(obs=observations)

    def update_and_normalize(self, obs) -> Data:
        """
        Update the normalizer statistics with the given observations and return the normalized observations
        Args:
            obs: the observation to update the normalizer with
            **kwargs: additional arguments

        Returns: the normalized observations. Also returns all additional arguments that were passed in **kwargs
        """
        self._update(obs=obs)
        obs = self.normalize(obs=obs)
        return obs

    def _update(self, obs: Data):
        """
        Update the normalizer statistics with the given observations.
        """
        self.node_normalizers.update(obs.x)
        self.edge_normalizers.update(obs.edge_attr)
        self.global_normalizer.update(obs.u)

    def normalize(self, obs: Data) -> Data:
        """
        Normalize observations using this instance's current statistics.
        Calling this method does not update statistics. It can thus be called for training as well as evaluation.
        """
        obs.__setattr__("x", self._normalize(obs=obs.x, normalizer=self.node_normalizers))
        obs.__setattr__("edge_attr", self._normalize(obs=obs.edge_attr, normalizer=self.edge_normalizers))
        obs.__setattr__("u", self._normalize(obs=obs.u, normalizer=self.global_normalizer))
        return obs

    def _normalize(self, obs: Tensor, normalizer: TorchRunningMeanStd) -> Tensor:
        """
        Helper to normalize a given observation.
        * param observation:
        * param normalizer: associated statistics
        * return: normalized observation
        """
        scaled_observation = (obs - normalizer.mean) / torch.sqrt(normalizer.var + self.epsilon)
        scaled_observation = torch.clip(scaled_observation, -self.obs_clip, self.obs_clip)
        return scaled_observation.float()
