from collections import Counter

import networkx as nx
import torch

from reward.reward_functions.reward_utils import build_routing_graphs
from utils.utils import flatten_nested_list
from utils.types import Optional, Data as PygData, Tensor


def routing_loops_per_node(**reward_input) -> Tensor:
    """
    Returns the amount of routing loops per node in the provided routing.
    """
    actions = reward_input.get("actions", None)
    if actions is None:
        raise ValueError("invoking routing_loops reward function with actions=None")

    routing_graphs_per_dest = build_routing_graphs(actions)
    cycles = [list(nx.simple_cycles(routing_graph)) for routing_graph in routing_graphs_per_dest]
    # in deterministic routing nodes can be involved in loops multiple times
    involved_nodes = flatten_nested_list(cycles)
    involved_counts = Counter(involved_nodes)
    return torch.tensor([involved_counts.get(i, 0) for i in range(actions.num_nodes)])
