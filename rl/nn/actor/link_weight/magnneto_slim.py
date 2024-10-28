from torch import nn as nn, Tensor
from torch_geometric.data import Batch

from rl.nn.hmpn import get_hmpn
from rl.nn.actor.link_weight._link_weight_actor import LinkWeightActor


class MagnnetoSlimActor(LinkWeightActor):
    """
    Link-weight actor network architecture that leverages the MPN model for link weight prediction.
    """
    def __init__(self, nn_config: dict, feature_counts: dict, device: str):
        super().__init__(device)
        self._embedding = get_hmpn(in_node_features=feature_counts["node"],
                                   in_edge_features=feature_counts["edge"],
                                   in_global_features=feature_counts["global"],
                                   latent_dimension=nn_config["latent_dimension"],
                                   base_config=nn_config["base"],
                                   unpack_output=False,
                                   device=self.device
                                   )
        self._readout = nn.Linear(nn_config["latent_dimension"], 1, device=device)

    def forward(self, input: Batch) -> Tensor:
        hidden = self._embedding(input)
        res = self._readout(hidden.x).squeeze(dim=-1)
        return res
