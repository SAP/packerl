from copy import deepcopy

import torch
from torch import nn as nn, Tensor
from torch_geometric.data import Batch, Data

from rl.nn.hmpn import get_hmpn
from rl.nn.actor.next_hop._next_hop import NextHopActor


class FieldLinesActor(NextHopActor):
    """
    This next-hop actor architecture uses an MPN to provide per-edge values from the latent embedding.
    """
    def __init__(self, nn_config: dict, feature_counts: dict, device: str, use_spdist=False):
        super().__init__(device)
        self._embedding = get_hmpn(in_node_features=feature_counts["node"],
                                   in_edge_features=feature_counts["edge"],
                                   in_global_features=feature_counts["global"],
                                   latent_dimension=nn_config["latent_dimension"],
                                   base_config=nn_config["base"],
                                   unpack_output=False,
                                   device=self.device)
        self._score = MPNScoreModule(latent_dimension=nn_config["latent_dimension"],
                                     base_config=nn_config["base"],
                                     device=self.device,
                                     use_spdist=use_spdist
                                     )

    def forward(self, input: Batch) -> Tensor:
        input_with_embeddings = self._embedding(input)
        score = self._score(input_with_embeddings)
        return score


class MPNScoreModule(nn.Module):
    """
    The MPN module used in the FieldLinesActor to produce per-edge values from the latent embedding.
    This module computes scores per edge per destination by first creating a graph per destination node
    where the destination node's features are concatenated to all nodes' features.
    Then, the MPN is applied to each graph and the edge scores are aggregated from the resulting values.
    """
    def __init__(self, device, latent_dimension, base_config, use_spdist):
        super().__init__()
        self.device = device
        self.latent_dimension = latent_dimension
        self.use_spdist = use_spdist
        gnn_embedding_config = deepcopy(base_config)
        gnn_embedding_config['create_graph_copy'] = False  # since this module is not operating on graph leaves, it can't use this option
        node_in_dim = latent_dimension * 2 + 1 if self.use_spdist else latent_dimension * 2
        self.per_dest_embedding = get_hmpn(in_node_features=node_in_dim,
                                           in_edge_features=latent_dimension,
                                           in_global_features=latent_dimension,
                                           latent_dimension=latent_dimension,
                                           base_config=gnn_embedding_config,
                                           unpack_output=False,
                                           device=device)
        self.per_dest_embedding_to_score = nn.Linear(in_features=latent_dimension, out_features=1, device=device)
        self.activation = nn.LeakyReLU()

    def _prepare_graphs(self, input: Batch):
        """
        Creates batch_size * N graphs from the input graph, where the node features of
        the corresponding destination node are concatenated to the entire graph's node features.
        """
        aug_graphs = []
        graph_N = []  # [num_nodes for graph in batch]
        graph_NE = []  # [num_nodes*num_edges for graph in batch]

        # for each graph in the batch create N graphs with per-destination features
        for b in range(input.num_graphs):
            graph = input.get_example(b)
            N, E = graph.num_nodes, graph.num_edges
            graph_N.append(N)
            graph_NE.append(N*E)

            # create new per-node input features for N graphs, where for each segment i
            # the features of destination node i are concatenated to all nodes' features.
            aug_xs_features = [graph.x.repeat(N, 1), graph.x.repeat_interleave(N, dim=0)]

            # add shortest path distance as auxiliary feature if desired
            if self.use_spdist:
                aug_xs_features.append(graph.spdist.unsqueeze(-1))

            # chunk creates features into N segments and create N graphs with per-destination features.
            aug_xs = torch.cat(aug_xs_features, dim=-1).chunk(N, dim=0)
            b_aug_graphs = [Data(x=x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, u=graph.u)
                            for x in aug_xs]
            aug_graphs.extend(b_aug_graphs)
        return aug_graphs, graph_N, graph_NE

    def _score(self, embedding: Batch, graph_NE, graph_N):
        scores = self.per_dest_embedding_to_score(embedding.edge_attr)  # shape: [sum([num_nodes*num_edges for graph in batch]), 1]
        scores_flattened_per_graph = [cur_graph_scores.reshape(N, -1).t().flatten()
                                      for cur_graph_scores, N in
                                      zip(scores.split(graph_NE, dim=0), graph_N)]  # per graph g: [g.num_edges * g.num_nodes]
        return torch.cat(scores_flattened_per_graph, dim=0)

    def forward(self, graph: Batch) -> Tensor:
        aug_graphs, graph_N, graph_NE = self._prepare_graphs(graph)
        embedding = self.per_dest_embedding(Batch.from_data_list(aug_graphs))
        res = self._score(embedding, graph_NE, graph_N)
        return res
