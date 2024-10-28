from typing import List

import networkx as nx
import torch
from torch_geometric.data import Data as PygData
from torch_geometric.utils import to_networkx


def build_routing_graphs(actions: PygData) -> List[nx.DiGraph]:
    """
    Builds a list of routing graphs from the given actions.
    Each routing graph represents the routing paths from all nodes to a single destination.
    """

    action_edges = actions.edge_attr  # [num_edges, num_nodes]
    src_index, _ = actions.edge_index
    if torch.any(torch.logical_and(action_edges > 0, action_edges < 1)):
        raise ValueError(f"can use reward_function 'routing_loops' "
                         f"only with deterministic action values (given: {action_edges})")
    action_values_per_dest = list(action_edges.t())
    # set action values to zero for those action edges that originate from destination nodes,
    # because they're irrelevant to routing loop calculation
    action_values_per_dest = [torch.where(src_index == i, torch.zeros_like(av), av)
                              for i, av in enumerate(action_values_per_dest)]
    action_edge_index_per_dest = [actions.edge_index[:, action_values >= 1]
                                  for action_values in action_values_per_dest]
    routing_graphs_per_dest = [to_networkx(PygData(edge_index=action_edge_index, num_nodes=actions.num_nodes))
                               for action_edge_index in action_edge_index_per_dest]
    return routing_graphs_per_dest
