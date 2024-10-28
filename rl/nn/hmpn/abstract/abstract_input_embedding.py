from typing import Dict, Optional, Any

from torch import nn

from rl.nn.hmpn.common.embedding import Embedding
from torch_geometric.data.batch import Batch


class AbstractInputEmbedding(nn.Module):
    """
    Parent class to heterogeneous and homogeneous input embeddings. Embeds global features.
    """

    def __init__(self,
                 *,
                 in_global_features: Optional[int],
                 embedding_config: Optional[Dict[str, Any]],
                 latent_dimension: int):
        """
        Args:
            in_global_features: The number of input global features, None if no global features are used
            latent_dimension: The embedding dimension
        """
        super().__init__()

        if in_global_features is None:
            self.global_input_embedding = None
            self._maybe_embed_globals = lambda *args, **kwargs: None  # No-op
        else:
            self.global_input_embedding = Embedding(in_features=in_global_features,
                                                    latent_dimension=latent_dimension,
                                                    embedding_config=embedding_config)
            self._maybe_embed_globals = self._embed_globals

    def _embed_globals(self, graph: Batch):
        """
        Runs the MLP that takes the global features graph.u as input and writes the embedding back into graph.u
        Args:
            graph: torch_geometric.data.BaseData

        Returns:
            None
        """
        graph.u = self.global_input_embedding(graph.u)

    def forward(self, graph: Batch):
        """
        Computes the forward pass for this input embedding. Does this for the global information, as the
        other features depend on the kind of embedding (homogeneous or heterogeneous) that is used
        Args:
            graph: BaseData object of pytorch geometric. Represents a (batch of) of homogeneous or heterogeneous graph(s)

        Returns:
            None
        """
        self._maybe_embed_globals(graph=graph)
