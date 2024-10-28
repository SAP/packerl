from typing import Dict, Any, List, Callable

import torch
from torch_geometric.data.batch import Batch

from rl.nn.hmpn.abstract.abstract_modules import AbstractMetaModule
from rl.nn.hmpn.common.latent_mlp import LatentMLP


class HomogeneousMetaModule(AbstractMetaModule):
    """
    Base class for the homogeneous modules used in the GNN.
    They are used for updating node-, edge-, and global features.
    """

    def __init__(self, *,
                 in_features: int,
                 latent_dimension: int,
                 stack_config: Dict[str, Any],
                 scatter_reducers: List[Callable],
                 use_global_features: bool = False):
        """
        Args:
            in_features: Number of input features
            latent_dimension: Dimensionality of the internal layers of the mlp
            stack_config: Dictionary specifying the way that the gnn base should look like
            scatter_reducers: How to aggregate over the nodes/edges/globals. Can for example be [torch.scatter_mean]
            use_global_features: whether global features are used
        """
        super().__init__(scatter_reducers=scatter_reducers)
        mlp_config = stack_config.get("mlp")
        self._mlp = LatentMLP(in_features=in_features,
                              latent_dimension=latent_dimension,
                              config=mlp_config)

        self.in_features = in_features
        self.latent_dimension = latent_dimension
        self.use_global_features = use_global_features

        if use_global_features:
            self._maybe_concat_global = self._concat_global
        else:
            self._maybe_concat_global = lambda x, y: x

    def _concat_global(self, features, graph):
        raise NotImplementedError(f"Module {type(self)} needs to implement _concat_global(self,features,graph)")


class HomogeneousEdgeModule(HomogeneousMetaModule):
    """
    Module for computing edge updates of a step on a homogeneous message passing GNN. Edge inputs are concatenated:
    Its own edge features, the features of the two participating nodes and optionally,
    global features are also concatenated to the input.
    """

    def forward(self, graph: Batch):
        """
        Compute edge updates for the edges of the Module for homogeneous graphs in-place.
        An updated representation of the edge attributes for all edge_types is written back into the graph
        Args:
            graph: Data object of pytorch geometric. Represents a batch of homogeneous graphs
        Returns: None
        """
        source_indices, dest_indices = graph.edge_index
        edge_source_nodes = graph.x[source_indices]
        edge_dest_nodes = graph.x[dest_indices]

        aggregated_features = torch.cat([edge_source_nodes, edge_dest_nodes, graph.edge_attr], 1)
        aggregated_features = self._maybe_concat_global(aggregated_features, graph)

        graph.__setattr__("edge_attr", self._mlp(aggregated_features))

    def _concat_global(self, aggregated_features, graph):
        """
        computation and concatenation of global features
        Args:
            aggregated_features: so-far aggregated features
            graph: pytorch_geometric.data.Batch object

        Returns: aggregated_features with the global features appended
        """
        source_indices, _ = graph.edge_index
        global_features = graph.u[graph.batch[source_indices]]
        return torch.cat([aggregated_features, global_features], 1)


class HomogeneousNodeModule(HomogeneousMetaModule):
    """
    Module for computing node updates of a step on a homogeneous message passing GNN. Node inputs are concatenated:
    Its own Node features, the reduced features of all incoming edges and optionally,
    global features are also concatenated to the input.
    """

    def __init__(self, *,
                 in_features: int,
                 latent_dimension: int,
                 stack_config: Dict[str, Any],
                 scatter_reducers: List[Callable],
                 use_global_features: bool = False,
                 flip_edges_for_nodes: bool = False):

        super(HomogeneousNodeModule, self).__init__(in_features=in_features,
                                                    latent_dimension=latent_dimension,
                                                    stack_config=stack_config,
                                                    scatter_reducers=scatter_reducers,
                                                    use_global_features=use_global_features)

        # use the source indices for feat. aggregation if edges shall be flipped
        if flip_edges_for_nodes:
            self._get_edge_indices = lambda src_and_dest_indices: src_and_dest_indices[0]
        else:
            self._get_edge_indices = lambda src_and_dest_indices: src_and_dest_indices[1]


    def forward(self, graph: Batch):
        """
        Compute updates for each node feature vector
            graph: Batch object of pytorch_geometric.data, represents a batch of homogeneous graphs
        Returns: None. In-place operation
        """
        src_indices, dest_indices = graph.edge_index
        scatter_edge_indices = self._get_edge_indices((src_indices, dest_indices))

        aggregated_edge_features = self.multiscatter(features=graph.edge_attr, indices=scatter_edge_indices,
                                                     dim=0, dim_size=graph.x.shape[0])
        aggregated_features = torch.cat([graph.x, aggregated_edge_features], dim=1)
        aggregated_features = self._maybe_concat_global(aggregated_features, graph)

        # update
        graph.__setattr__("x", self._mlp(aggregated_features))

    def _concat_global(self, aggregated_features, graph):
        """
        computation and concatenation of global features
        Args:
            aggregated_features: so-far aggregated features
            graph: pytorch_geometric.data.Batch object

        Returns: aggregated_features with the global features appended
        """
        global_features = graph.u[graph.batch]
        return torch.cat([aggregated_features, global_features], dim=1)


class HomogeneousGlobalModule(HomogeneousMetaModule):
    """
    Module for computing updates of global features of a step on a homogeneous message passing GNN.
    Global feature network inputs are concatenated: Its own global features, the reduced features of all edges,
    and the reduced features of all nodes.
    """

    def forward(self, graph: Batch):
        """
        Compute updates for the global feature vector
            graph: Batch object of pytorch_geometric.data, represents a batch of homogeneous graphs
        Returns: None. in-place operation.
        """
        reduced_node_features = self.multiscatter(features=graph.x,
                                                  indices=graph.batch,
                                                  dim=0,
                                                  dim_size=graph.u.shape[0])
        source_indices, _ = graph.edge_index
        reduced_edge_features = self.multiscatter(features=graph.edge_attr,
                                                  indices=graph.batch[source_indices],
                                                  dim=0,
                                                  dim_size=graph.u.shape[0])
        aggregated_features = torch.cat([reduced_edge_features, reduced_node_features, graph.u], dim=1)
        graph.__setattr__("u", self._mlp(aggregated_features))
