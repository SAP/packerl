from networkx import connected_watts_strogatz_graph
import numpy as np

from scenarios.config import BaseConfig, SingleOrChoice


class WattsStrogatzConfig(BaseConfig):
    """
    Configuration for Watts-Strogatz random graph generation.
    """
    num_attachments: SingleOrChoice[int]
    rewire_probability: SingleOrChoice[float]


def generate_watts_strogatz(num_attachments: int, rewire_probability: float, num_nodes: int,
                            rng: np.random.Generator, **kwargs):
    """
    Generates a random graph according to the Watts-Strogatz model.
    :param num_nodes: number of nodes
    :param num_attachments: 'Each node is joined with its k nearest neighbors in a ring graph' (NetworkX)
    :param rewire_probability: probability of rewiring each edge
    :param rng: random number generator
    """

    # use "connected_" variant to ensure the graph is a single component
    topology = connected_watts_strogatz_graph(n=num_nodes, k=num_attachments, p=rewire_probability, seed=rng)
    return topology
