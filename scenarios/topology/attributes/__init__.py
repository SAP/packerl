"""
Topology attributes like node positions, edge delays, and edge datarates are generated here,
based on the configuration provided in the scenario configuration.
"""
import numpy as np
import networkx as nx

from scenarios.config import BaseConfig, SingleOrChoice
from utils.utils import rescale


class TopologyAttributesConfig(BaseConfig):
    """
    Configuration for generating topology attributes.
    """
    fixed_attributes: bool  # If set to true, all nodes and edges have the same weight (i.e. uniform)
    node_traffic_pos_factor: SingleOrChoice[float]  # Maximum possible pos. node weight = this factor times the minimum
    node_traffic_deg_factor: SingleOrChoice[float]  # Maximum possible deg. node weight = this factor times the minimum
    lambda_deg_weight: SingleOrChoice[float]  # 1.0: only use node deg. for node weight; 0.0: only use node centrality
    edge_weight_min: int  # Minimum edge weight in bps
    edge_weight_med: int  # Median edge weight in bps
    edge_weight_max: int  # Maximum edge weight in bps
    delay_avg: SingleOrChoice[float]  # in ms


def generate_topology_attributes(G, scenario_cfg: dict, rng: np.random.Generator):
    """
    Generate topology attributes like node positions, edge delays, and edge datarates.
    """

    attr_cfg = scenario_cfg['topology']['attributes']

    # get node positions (if not all nodes contain position information, generate new positions using spring layout)
    if not all('pos' in G.nodes[n] for n in G.nodes):
        pos = nx.spring_layout(G, seed=9001)  # nx doesn't support np.random.Generator yet, so just fix the seed
        nx.set_node_attributes(G, {n: {"pos": n_pos} for n, n_pos in pos.items()})
    else:
        pos = nx.get_node_attributes(G, 'pos')

    # if nodes were not yet given the 'active' attribute, set all nodes to active
    if "active" not in G.nodes[0]:
        nx.set_node_attributes(G, True, "active")

    # node traffic potentials derived from node positions and degrees (or fixed)
    if attr_cfg['fixed_attributes']:
        node_traffic_potentials = np.ones(G.number_of_nodes())
    else:
        pos_weights = [1.0 - (np.sqrt(v[0] ** 2 + v[1] ** 2) / np.sqrt(2)) for v in pos.values()]  # [0.0, 1.0]
        pos_weights = rescale(np.array(pos_weights), 1.0, attr_cfg['node_traffic_pos_factor'])
        degs = [d[1] for d in G.degree]
        deg_weights = rescale(np.array(degs), 1.0, attr_cfg['node_traffic_deg_factor'])
        lmb_deg = attr_cfg['lambda_deg_weight']
        node_traffic_potentials = lmb_deg * deg_weights + (1 - lmb_deg) * pos_weights
    # set potential of inactive nodes to 0 and make sure max potential is 1
    node_is_active = np.array(list(nx.get_node_attributes(G, "active").values()))
    node_traffic_potentials[np.logical_not(node_is_active)] = 0
    node_traffic_potentials /= np.max(node_traffic_potentials)  # [0.0, 1.0]
    nx.set_node_attributes(G, {n: {"traffic_potential": val} for n, val in enumerate(node_traffic_potentials)})

    # edge delays: fixed, proportional to the distances between nodes or simply keep existing datarates
    if list(G.edges) != list(nx.get_edge_attributes(G, "delay")):
        if not attr_cfg['fixed_attributes']:  # randomize edge weights
            edge_delays = (np.ones(G.number_of_edges()) * attr_cfg['delay_avg']).astype(int)
        else:
            edge_delays = np.array([(np.sqrt((pos[v][0] - pos[u][0]) ** 2 + (pos[v][1] - pos[u][1]) ** 2))
                                    for (u, v) in G.edges])
            edge_delays = edge_delays * rng.uniform(low=1 - scenario_cfg['rng_uniform_deviation'],
                                                    high=1 + scenario_cfg['rng_uniform_deviation'],
                                                    size=edge_delays.shape)
            edge_delays = edge_delays * (attr_cfg['delay_avg'] / np.average(edge_delays))
            edge_delays = np.maximum(edge_delays, np.ones_like(edge_delays)).astype(int)  # minimum delay is 1ms
        nx.set_edge_attributes(G, {e: edge_delays[i] for i, e in enumerate(G.edges)}, "delay")

    # edge datarates: fixed, proportional to max. incident node traffic potential, or from existing datarates
    if list(G.edges) != list(nx.get_edge_attributes(G, "datarate")):
        if attr_cfg['fixed_attributes']:
            edge_datarates = np.ones(G.number_of_edges()) * attr_cfg['edge_weight_med']
        else:
            node_tp = nx.get_node_attributes(G, 'traffic_potential')
            _, edge_weights = list(zip(*[((u, v), np.maximum(node_tp[u], node_tp[v])) for (u, v) in G.edges]))
            edge_dr_weights = np.array(edge_weights).astype(float)
            edge_dr_weights = edge_dr_weights * rng.uniform(low=1 - scenario_cfg['rng_uniform_deviation'],
                                                            high=1 + scenario_cfg['rng_uniform_deviation'],
                                                            size=edge_dr_weights.shape)
            edge_datarates = rescale(edge_dr_weights, attr_cfg['edge_weight_min'], attr_cfg['edge_weight_max'])
    else:
        edge_dr_weights = np.array(list(nx.get_edge_attributes(G, 'datarate').values())).astype(float)
        if G.graph.pop('keepDatarateProportions', False):
            edge_datarates = edge_dr_weights * (attr_cfg['edge_weight_min'] / np.min(edge_dr_weights))
        else:
            edge_datarates = rescale(edge_dr_weights, attr_cfg['edge_weight_min'], attr_cfg['edge_weight_max'])
    nx.set_edge_attributes(G, {e: edge_datarates[i] for i, e in enumerate(G.edges)}, "datarate")

    # save average datarate as graph attribute for later traffic generation
    G.graph['avg_datarate'] = np.average(np.array(list(nx.get_edge_attributes(G, 'datarate').values()))
                                         .astype(float))
    return G
