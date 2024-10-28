"""
Topology graph generation is done here, based on the configuration provided in the scenario configuration.
"""
from typing import Tuple, Dict, Any

import networkx as nx
import numpy as np

from scenarios.config import BaseConfig, SingleOrChoice
from .random import generate_watts_strogatz, WattsStrogatzConfig
from .random import generate_barabasi_albert, BarabasiAlbertConfig
from .random import generate_erdos_renyi, ErdosRenyiConfig
from .magnneto import generate_magnneto, MagnnetoDatasetConfig

graph_generators = {
    "erdos_renyi": generate_erdos_renyi,
    "watts_strogatz": generate_watts_strogatz,
    "barabasi_albert": generate_barabasi_albert,
    "magnneto": generate_magnneto,
}

graph_config_classes = {
    "erdos_renyi": ErdosRenyiConfig,
    "watts_strogatz": WattsStrogatzConfig,
    "barabasi_albert": BarabasiAlbertConfig,
    "magnneto": MagnnetoDatasetConfig,
}

graph_traffic_scalings = {
    "erdos_renyi": lambda G: np.maximum(1, 16 * np.sqrt(G.number_of_nodes()) - 30),
    "watts_strogatz": lambda G: np.maximum(1, 22 * np.sqrt(G.number_of_nodes() + 10) - 75),
    "barabasi_albert": lambda G: np.maximum(1, 22 * np.sqrt(G.number_of_nodes() + 10) - 75),
    "default": lambda G: 10,
}


class TopologyGraphConfig(BaseConfig):
    """
    Configuration for topology graph generation.
    """
    min_nodes: int
    max_nodes: int
    mode: SingleOrChoice[str]
    mode_configs: Dict[str, BaseConfig] = {}

    def __init__(self, **data):

        mode_configs = data.pop('mode_configs', {})
        super().__init__(**data)
        for graph_mode, cfg_data in mode_configs.items():
            if graph_mode not in graph_config_classes.keys():
                raise ValueError(f"Invalid graph mode: {graph_mode}")
            self.mode_configs[graph_mode] = graph_config_classes[graph_mode](**cfg_data)

    def sample(self, rng: np.random.Generator) -> Dict[str, Any]:
        sampled_cfg = super().sample(rng)
        min_nodes = sampled_cfg.pop('min_nodes')
        max_nodes = sampled_cfg.pop('max_nodes')
        sampled_cfg['num_nodes'] = rng.integers(min_nodes, max_nodes, endpoint=True)
        return sampled_cfg


def generate_topology_graph(scenario_cfg: dict, rng: np.random.Generator, rng_i: int) -> Tuple[nx.Graph, dict]:
    """
    Generate a random graph based on the given scenario configuration.
    """

    # preparation
    graph_cfg = scenario_cfg['topology']['graph']
    seed = scenario_cfg['seed']
    num_nodes = graph_cfg['num_nodes']
    chosen_mode = graph_cfg['mode']
    if chosen_mode not in graph_generators.keys():
        raise ValueError(f"Invalid chosen topology graph mode: {chosen_mode}")
    graph_generator_cfg = graph_cfg['mode_configs'][chosen_mode]
    graph_generator = graph_generators[chosen_mode]

    # generate graph depending on the chosen graph mode
    G = graph_generator(num_nodes=num_nodes, rng=rng, rng_i=rng_i, **graph_generator_cfg)

    # assign graph-dependent attributes
    scaling_method = graph_traffic_scalings.get(chosen_mode, graph_traffic_scalings['default'])
    G.graph['traffic_scaling'] = scaling_method(G)
    graph_name = f"{chosen_mode}-seed{seed}-i{rng_i}"
    if G.graph.get('name') is not None:
        graph_name += f"-{G.graph['name']}"
    G.graph['name'] = graph_name

    return G
