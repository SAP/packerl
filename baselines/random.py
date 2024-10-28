import torch
import torch_scatter
from torch_geometric.data import Data, Batch
import networkx as nx

from packerl_env import PackerlEnv
from utils.tensor import get_edge_dest_idx


def get_action_random_next_hop(obs: Data, **kwargs):
    """
    Returns a random selection of routing edges.
    """
    obs_batch = Batch.from_data_list([obs])
    edge_dest_idx = get_edge_dest_idx(obs_batch)
    random_values = torch.rand((obs.num_nodes * obs.num_edges,))
    _, selected_edge_dest_idx = torch_scatter.scatter_max(random_values, index=edge_dest_idx)
    value = torch.tensor(0)
    return (random_values, selected_edge_dest_idx), value


def get_action_random_link_weight(obs: Data, **kwargs):
    """
    Returns routing paths generated with random link weights
    """
    if 'env' not in kwargs:
        raise RuntimeError("Using the SPF baselines requires a PackerlEnv to be passed as a keyword argument.")
    env: PackerlEnv = kwargs['env']
    G = env.monitoring.copy()

    # assign random integer edge weights between 1 and 4 (inclusively) to the 'randomLinkWeight' attribute of G
    attr_name = 'randomLinkWeight'
    random_link_weights = torch.randint(1, 5, (G.number_of_edges(),), dtype=torch.int)
    for edge, weight in zip(G.edges, random_link_weights):
        G.edges[edge][attr_name] = weight.item()

    # compute shortest paths and the corresponding edge values
    apsp = dict(nx.all_pairs_dijkstra(G, cutoff=None, weight=attr_name))
    shortest_paths = {node_id: sp for node_id, (_, sp) in apsp.items()}
    involved_edges_per_dst = {node_id: [] for node_id in G.nodes}
    for paths_per_src in shortest_paths.values():
        for dst, path in paths_per_src.items():
            involved_edges = zip(path, path[1:])
            involved_edges_per_dst[dst].extend(involved_edges)
    edges = list(G.edges)
    involved_edge_idx_per_dst = {node_id: [edges.index(e) for e in set(involved_e)]
                                 for node_id, involved_e in involved_edges_per_dst.items()}
    edge_values_per_dst = [torch.zeros((len(edges), 1)) for _ in G.nodes]
    for involved_edge_idx, edge_values in zip(involved_edge_idx_per_dst.values(), edge_values_per_dst):
        edge_values[involved_edge_idx] = 1
    sp_actions = torch.cat(edge_values_per_dst, dim=1)
    selected_edge_dest_idx = sp_actions.flatten().nonzero(as_tuple=False).squeeze()

    # finalize
    action = (sp_actions.float(), selected_edge_dest_idx)
    value = torch.tensor(0.)
    return action, value
