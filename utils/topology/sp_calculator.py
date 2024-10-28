import networkx as nx
import torch

from utils.topology.topology_utils import network_graphs_equal
from utils.types import Tensor, List, Optional


def get_shortest_path_calculator(sp_mode, ref_values: dict):
    """
    Factory function to create a ShortestPathCalculator instance based on the given sp_mode.
    """
    if sp_mode == "ospf":
        return OSPFShortestPathCalculator(ref_values['ospfw_ref_value'])
    elif sp_mode == "eigrp":
        return EIGRPShortestPathCalculator(ref_values['eigrp_ref_datarate'],
                                           ref_values['eigrp_ref_delay'],
                                           ref_values['eigrp_ref_multiplier'])
    else:
        raise ValueError(f"Invalid sp_mode: {sp_mode}, can't create ShortestPathCalculator")


class ShortestPathCalculator:
    """
    Class to calculate and cache shortest paths and node distances for a given network graph.
    Can be configured to use either OSPF or EIGRP routing weights.
    """
    relevant_sp_attrs = None
    sp_mode = None

    def __init__(self):
        self.cached_network_graph: Optional[nx.DiGraph] = None
        self.weights: Optional[List] = None
        self.node_distances = None
        self.shortest_paths = None
        self.sp_actions = None

    def _check_reset(self, network_graph: nx.DiGraph, force_reset: bool = False):
        """
        Check if the given network graph is different from the cached one,
        and reset the cached values if necessary.
        """
        should_reset = (force_reset
                        or self.cached_network_graph is None
                        or not network_graphs_equal(self.cached_network_graph, network_graph,
                                                    edge_attrs=self.relevant_sp_attrs))
        if should_reset:
            self.cached_network_graph = network_graph
            self._set_weights()
            self._set_node_dist_and_shortest_paths()
            self._set_sp_actions()

    def _set_weights(self):
        """
        Calculate and store routing weights for given network graph.
        """
        raise NotImplementedError

    def _set_node_dist_and_shortest_paths(self):
        """
        Calculate and store node distances and shortest paths for given network graph.
        """
        routing_weights = self.weights
        attr_name = f"{self.sp_mode}Weight"
        weighted_edges = dict(zip(self.cached_network_graph.edges, routing_weights))
        nx.set_edge_attributes(self.cached_network_graph, weighted_edges, name=attr_name)
        apsp = dict(nx.all_pairs_dijkstra(self.cached_network_graph, cutoff=None, weight=attr_name))
        node_distances, shortest_paths = dict(), dict()
        for node_id, (nd, sp) in apsp.items():
            node_distances[node_id] = nd
            shortest_paths[node_id] = sp
        self.node_distances = node_distances
        self.shortest_paths = shortest_paths

    def _set_sp_actions(self):
        """
        Get shortest paths for given network graph, and use them to create routing actions.
        """
        # TODO this can be optimized with some clever indexing and torch.scatter

        involved_edges_per_dst = {k: [] for k in self.cached_network_graph.nodes}
        for paths_per_src in self.shortest_paths.values():
            for dst, path in paths_per_src.items():
                involved_edges = zip(path, path[1:])
                involved_edges_per_dst[dst].extend(involved_edges)
        edges = list(self.cached_network_graph.edges)
        involved_edge_idx_per_dst = {k: [edges.index(e) for e in set(v)] for k, v in involved_edges_per_dst.items()}
        edge_values_per_dst = [torch.zeros((len(edges), 1)) for _ in self.cached_network_graph.nodes]
        for involved_edge_idx, edge_values in zip(involved_edge_idx_per_dst.values(), edge_values_per_dst):
            edge_values[involved_edge_idx] = 1

        self.sp_actions = torch.cat(edge_values_per_dst, dim=1)

    def get_node_distances(self, network_graph: nx.DiGraph, force_reset: bool = False):
        """
        Return {cached/calculated} node distances for given network graph.
        """
        self._check_reset(network_graph, force_reset)
        return self.node_distances

    def get_shortest_paths(self, network_graph: nx.DiGraph = None, force_reset: bool = False):
        """
        Return {cached/calculated} shortest paths for given network graph.
        """
        self._check_reset(network_graph, force_reset)
        return self.shortest_paths

    def get_sp_actions(self, network_graph: nx.DiGraph, force_reset: bool = False) -> Tensor:
        """
        Return {cached/calculated} routing actions for given network graph.
        """
        self._check_reset(network_graph, force_reset)
        return self.sp_actions


class OSPFShortestPathCalculator(ShortestPathCalculator):
    """
    OSPF routing calculator, using the OSPF reference datarate as a weight.
    """

    relevant_sp_attrs = ['channelDataRate']
    sp_mode = "ospf"

    def __init__(self, ospf_ref_datarate):
        self.ospf_ref_datarate = ospf_ref_datarate
        self.cached_network_graph: Optional[nx.DiGraph] = None
        super().__init__()

    def _set_weights(self):
        """
        Calculate and store OSPF weights for given network graph,
        where datarate is in bps and delay is in ms.
        """
        self.weights = [self.ospf_ref_datarate / datarate
                        for _, _, datarate in self.cached_network_graph.edges.data('channelDataRate')]


class EIGRPShortestPathCalculator(ShortestPathCalculator):
    """
    EIGRP routing calculator, using the EIGRP reference datarate and delay as weights.
    """

    relevant_sp_attrs = ['channelDataRate', 'channelDelay']
    sp_mode = "eigrp"

    def __init__(self, eigrp_ref_datarate, eigrp_ref_delay, eigrp_ref_multiplier):
        self.eigrp_ref_datarate = eigrp_ref_datarate
        self.eigrp_ref_delay = eigrp_ref_delay
        self.eigrp_ref_multiplier = eigrp_ref_multiplier
        super().__init__()

    def _set_weights(self):
        """
        Calculate and store EIGRP weights for given monitoring graph,
        where datarate is converted to kbps (*=0.001 from bps),
        and delay is converted to tens of µs (*=100 from ms).
        """

        eigrp_datarate_weights = [self.eigrp_ref_datarate / (datarate * 0.001) for
                                  _, _, datarate in self.cached_network_graph.edges.data('channelDataRate')]
        eigrp_delay_weights = [(delay * 100) / self.eigrp_ref_delay for
                               _, _, delay in self.cached_network_graph.edges.data('channelDelay')]
        self.weights = [self.eigrp_ref_multiplier * (dr_w + dl_w)
                        for dr_w, dl_w in zip(eigrp_datarate_weights, eigrp_delay_weights)]
