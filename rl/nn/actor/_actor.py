from torch import nn

from utils.types import Batch, Tensor


class Actor(nn.Module):
    """
    Abstract class for an actor.
    """
    def __init__(self, device: str):
        super().__init__()
        self.device = device

    def forward(self, input: Batch) -> Tensor:
        raise NotImplementedError
