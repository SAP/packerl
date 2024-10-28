from torch_geometric.data import Data

from rl.normalizer._normalizer import AbstractNormalizer


class ObservationNormalizer(AbstractNormalizer):
    """
    Abstract class for observation normalizers.
    """
    def __init__(self, obs_clip: float = 10, epsilon: float = 1.0e-6):
        super().__init__(epsilon)
        self.obs_clip = obs_clip

    def reset(self, obs: Data) -> Data:
        """
        Reset the normalizer parameters with the given observation and return the normalized observation.
        """
        raise NotImplementedError

    def update_and_normalize(self, obs) -> Data:
        """
        Update the normalizer parameters with the given observation and return the normalized observation.
        """
        raise NotImplementedError

    def _update(self, obs: Data):
        """
        Update the normalizer parameters with the given observation.
        """
        raise NotImplementedError

    def normalize(self, obs: Data) -> Data:
        """
        Normalize the given observation.
        """
        raise NotImplementedError
