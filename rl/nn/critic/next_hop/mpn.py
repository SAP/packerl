from torch import nn
from torch_scatter import scatter_mean

from rl.nn.hmpn import get_hmpn
from rl.nn.hmpn.common.latent_mlp import LatentMLP
from rl.nn.critic.next_hop._next_hop_critic import NextHopCritic
from utils.types import Batch, Tensor


class MPNNextHopCritic(NextHopCritic):
    """
    Critic network architecture for next-hop prediction using the MPN design
    """
    def __init__(self,
                 nn_config: dict,
                 feature_counts: dict,
                 value_scope: str,
                 device: str):
        super(MPNNextHopCritic, self).__init__(device)
        self._embedding = get_hmpn(in_node_features=feature_counts["node"],
                                   in_edge_features=feature_counts["edge"],
                                   in_global_features=feature_counts["global"],
                                   latent_dimension=nn_config["latent_dimension"],
                                   base_config=nn_config["base"],
                                   unpack_output=False,
                                   device=self.device)
        self._edge_mlp = LatentMLP(in_features=nn_config["latent_dimension"],
                                   latent_dimension=nn_config["latent_dimension"],
                                   config=nn_config["base"]["stack"]["mlp"]
                                   ).to(self.device)
        self._edge_value = nn.Linear(in_features=nn_config["latent_dimension"], out_features=1,
                                     device=self.device)

        if value_scope == "graph":
            self._get_value = lambda edge_values, edge_batch: scatter_mean(edge_values, edge_batch, dim=0)
        else:
            raise NotImplementedError(f"invalid value scope: {value_scope}")

    def forward(self, input: Batch) -> Tensor:
        input = self._embedding(input)
        edge_hidden = self._edge_mlp(input.edge_attr)
        edge_values = self._edge_value(edge_hidden)
        edge_batch = input.batch[input.edge_index[0]]
        return self._get_value(edge_values, edge_batch)
