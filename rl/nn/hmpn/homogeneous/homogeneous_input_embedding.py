from typing import Dict, Optional, Any

from torch_geometric.data.data import Data

from rl.nn.hmpn.abstract.abstract_input_embedding import AbstractInputEmbedding
from rl.nn.hmpn.common.embedding import Embedding


class HomogeneousInputEmbedding(AbstractInputEmbedding):
    def __init__(self,
                 *,
                 in_node_features: int,
                 in_edge_features: int,
                 in_global_features: Optional[int],
                 embedding_config: Optional[Dict[str, Any]],
                 latent_dimension: int):
        """
        Builds and returns an input embedding for a homogeneous graph.
        Args:
            in_node_features:
                number of input node features
            in_edge_features:
                number of input edge features
            in_global_features:
                number of input global features. None if no global features are used.
            latent_dimension:
                dimension of the latent space.
        """
        super().__init__(in_global_features=in_global_features, latent_dimension=latent_dimension,
                         embedding_config=embedding_config)

        self.node_input_embedding = Embedding(in_features=in_node_features,
                                              latent_dimension=latent_dimension,
                                              embedding_config=embedding_config)

        self.edge_input_embedding = Embedding(in_features=in_edge_features,
                                              latent_dimension=latent_dimension,
                                              embedding_config=embedding_config)

        if in_global_features is not None:
            self.global_input_embedding = Embedding(in_features=in_global_features,
                                                    latent_dimension=latent_dimension,
                                                    embedding_config=embedding_config)

    def forward(self, graph: Data):
        """
        Computes the forward pass for this homogeneous input embedding inplace
        Args:
            graph: torch_geometric.data.Batch, represents a batch of homogeneous graphs
        Returns:
            None
        """
        graph.__setattr__("x", self.node_input_embedding(graph.x))
        graph.__setattr__("edge_attr", self.edge_input_embedding(graph.edge_attr))

        super().forward(graph=graph)
