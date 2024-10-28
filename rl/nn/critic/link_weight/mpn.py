from torch import nn
from torch_scatter import scatter_mean

from rl.nn.hmpn import get_hmpn
from rl.nn.hmpn.common.latent_mlp import LatentMLP
from rl.nn.critic.link_weight._link_weight_critic import LinkWeightCritic
from utils.types import Batch, Tensor


class MPNLinkWeightCritic(LinkWeightCritic):
    """
    Critic network architecture for link weight prediction using the MPN design
    """
    def __init__(self,
                 nn_config: dict,
                 feature_counts: dict,
                 value_scope: str,
                 device: str):
        super(MPNLinkWeightCritic, self).__init__(device)
        self._embedding = get_hmpn(in_node_features=feature_counts["node"],
                                   in_edge_features=feature_counts["edge"],
                                   in_global_features=feature_counts["global"],
                                   latent_dimension=nn_config["latent_dimension"],
                                   base_config=nn_config["base"],
                                   unpack_output=False,
                                   device=self.device)
        self._node_mlp = LatentMLP(in_features=nn_config["latent_dimension"],
                                   latent_dimension=nn_config["latent_dimension"],
                                   config=nn_config["base"]["stack"]["mlp"]
                                   ).to(self.device)
        self._node_value = nn.Linear(in_features=nn_config["latent_dimension"], out_features=1,
                                     device=self.device)

        if value_scope == "graph":
            self._get_value = lambda node_values, node_batch: scatter_mean(node_values, node_batch, dim=0)
        else:
            raise ValueError(f"invalid value scope: {value_scope}")

    def forward(self, input: Batch) -> Tensor:
        input = self._embedding(input)
        node_hidden = self._node_mlp(input.x)
        node_values = self._node_value(node_hidden)
        return self._get_value(node_values, input.batch)
