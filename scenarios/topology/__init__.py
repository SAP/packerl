"""
Topology generation module.
"""
import numpy as np

from .graph import generate_topology_graph, TopologyGraphConfig
from .attributes import generate_topology_attributes, TopologyAttributesConfig
from scenarios.config import BaseConfig


class TopologyConfig(BaseConfig):
    """
    Topology configuration. Contains the configuration for the topology graph and the topology attributes.
    """
    graph: TopologyGraphConfig
    attributes: TopologyAttributesConfig


def generate_topology(scenario_cfg: dict, rng: np.random.Generator, generated_network_rng_id: int):
    """
    Generate a topology graph with attributes.
    """
    G = generate_topology_graph(scenario_cfg, rng, generated_network_rng_id)
    G = generate_topology_attributes(G, scenario_cfg, rng)
    return G
