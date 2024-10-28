import networkx as nx
import numpy as np
from networkx import erdos_renyi_graph

from scenarios.config import BaseConfig, SingleOrChoice


class ErdosRenyiConfig(BaseConfig):
    """
    Configuration for Erdos-Renyi random graph generation.
    """
    num_attachments: SingleOrChoice[int]


def generate_erdos_renyi(num_attachments: int, num_nodes: int, rng: np.random.Generator, **kwargs):
    """
    Generates a random graph according to the Erdos-Renyi model.
    :param num_nodes: number of nodes
    :param num_attachments: average attachment count
    :param rng: random number generator
    """
    topology = erdos_renyi_graph(n=num_nodes, p=num_attachments / num_nodes, seed=rng)

    # ensure the graph is a single component by connecting all components to the largest one
    ccs = sorted(nx.connected_components(topology), key=len)
    if len(ccs) > 1:
        main_component = ccs.pop(-1)
        main_component = list(main_component)
        for cc in ccs:
            n_in = rng.choice(list(cc))
            n_out = rng.choice(main_component)
            topology.add_edge(n_in, n_out)
    return topology
