from networkx import barabasi_albert_graph
import numpy as np

from scenarios.config import BaseConfig, SingleOrChoice


class BarabasiAlbertConfig(BaseConfig):
    """
    Configuration for Barabasi-Albert random graph generation.
    """
    num_attachments: SingleOrChoice[int]


def generate_barabasi_albert(num_attachments: int, num_nodes: int, rng: np.random.Generator, **kwargs):
    """
    Generates a random graph according to the Barabasi-Albert model.
    :param num_nodes: number of nodes
    :param num_attachments: Number of edges to attach from a new node to existing node
    :param rng: random number generator
    """

    # if no disconnected graph is given as initial graph,
    # this generator returns graph with 1 connected component by default
    topology = barabasi_albert_graph(n=num_nodes, m=num_attachments, seed=rng)
    return topology
