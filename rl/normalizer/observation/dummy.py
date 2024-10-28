from torch_geometric.data import Data

from rl.normalizer.observation._obs_normalizer import ObservationNormalizer


class DummyObservationNormalizer(ObservationNormalizer):
    """
    Dummy observation normalizer that does nothing.
    """
    def __init__(self):
        super().__init__()

    def reset(self, obs: Data) -> Data:
        return obs

    def update_and_normalize(self, obs) -> Data:
        return obs

    def _update(self, obs: Data):
        pass

    def normalize(self, obs: Data) -> Data:
        return obs
