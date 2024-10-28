import pickle as pkl
import pathlib

class AbstractNormalizer:
    """
    Abstract class for normalizers.
    """
    def __init__(self, epsilon: float = 1.0e-6):
        self.epsilon = epsilon


    def save(self, destination_path: pathlib.Path) -> None:
        """
        Saves the normalizer to a checkpoint file.

        Args:
            destination_path: the path to checkpoint to
        Returns:

        """
        with destination_path.open("wb") as f:
            pkl.dump(self, f)

    @staticmethod
    def load(checkpoint_path: pathlib.Path) -> "AbstractNormalizer":
        """
        Loads existing normalizers from a checkpoint.
        Args:
            checkpoint_path: The checkpoints directory of a previous experiment.

        Returns: A new normalizer object with the loaded normalizer parameters.

        """
        with checkpoint_path.open("rb") as f:  # load the file, create a new normalizer object and return it
            return pkl.load(f)
