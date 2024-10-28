from torch import nn

from utils.types import Batch, Tensor


class Critic(nn.Module):
    """
    Abstract critic class
    """
    def __init__(self, device: str):
        super().__init__()
        self.device = device

    def forward(self, input: Batch) -> Tensor:
        raise NotImplementedError
