import networkx as nx
import torch
import torch.nn as nn
from torch_geometric.utils import coalesce


from rl.nn.actor import build_link_weight_actor
from rl.nn.critic.link_weight import build_link_weight_critic
from rl.nn.actor_critic import ActorCritic
from utils.types import Tensor, Union, Data, Batch


class LinkWeightActorCritic(ActorCritic):
    """
    Abstract link-weight actor-critic class: Its actions denote link weights used to obtain routing decisions.
    It is up to the derived classes to determine how exactly these link weights are calculated.
    We obtain the link weights via computation on the line graph of the input graph, i.e. we modify the preprocessing
    step to convert the input graph to its line graph representation.
    """
    def __init__(self,
                 ac_config: dict,
                 nn_config: dict,
                 feature_counts: dict,
                 value_scope: str,
                 learning_rate: float,
                 device: str
                 ):
        super().__init__(ac_config, nn_config, feature_counts, value_scope, learning_rate, device)
        self.softplus = nn.Softplus()

    def _build_actor(self, nn_config, feature_counts, actor_mode, device):
        """
        Get edge-centric actor
        """
        return build_link_weight_actor(nn_config, feature_counts, actor_mode, device)

    def _build_critic(self, nn_config, feature_counts, critic_mode, value_scope, device):
        """
        Get edge-centric critic
        """
        return build_link_weight_critic(nn_config, feature_counts, critic_mode, value_scope, device)

    def _to_line_digraph(self, data: Data) -> Data:
        """
        Converts the input graph to a line digraph by taking the original graph's edges as new nodes,
        and drawing an edge between those new nodes that, as edges in the original graph,
        form a directed path of length two:
        E' = {(u, v), (w, x) | (u, v) in E; (w, x) in E; v = w}.
        The edge attributes of the original graph are used as node features in the line graph.
        """
        assert data.edge_index is not None
        edge_index, edge_attr = data.edge_index, data.edge_attr
        N, E = data.num_nodes, data.num_edges

        edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes=data.num_nodes)
        row, col = edge_index

        new_edge_index = []
        for i in range(E):
            new_col_i = torch.nonzero(row == col[i])
            new_row_i = i * torch.ones_like(new_col_i)
            new_edge_index.append(torch.cat([new_row_i, new_col_i], dim=1))
        new_edge_index = torch.cat(new_edge_index, dim=0).t()

        data.edge_index = new_edge_index
        data.x = edge_attr
        data.num_nodes = E
        data.edge_attr = torch.empty((new_edge_index.size(1), 0), dtype=edge_attr.dtype, device=edge_attr.device)
        return data

    def _preprocess_input(self, input: Union[Data, Batch]) -> Batch:
        """
        input preprocessing for link-weight actor-critic: Convert input graph to line digraph and batchify it.
        """
        _preprocessed_input: Batch = super()._preprocess_input(input)

        line_digraphs = []
        for i in range(_preprocessed_input.num_graphs):
            cur_graph: Data = _preprocessed_input.get_example(i)
            line_digraph = self._to_line_digraph(cur_graph)
            line_digraphs.append(line_digraph)
        preprocessed = Batch.from_data_list(line_digraphs)
        return preprocessed

    def _link_weights_to_routing(self, G: nx.DiGraph, attr_name: str) -> Tensor:
        """
        Given a directed graph G with link weights, this function calculates the shortest paths for all node pairs
        and, from that, the routing actions.
        """
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
        return selected_edge_dest_idx

    def _get_selected_edge_dest_idx(self, link_weights: Tensor, input: Union[Data, Batch]) -> Tensor:
        """
        Given the link weights for the edges of the input graph, this function calculates the shortest paths for all
        node pairs and returns the indices of the edges that are part of the shortest paths.
        """
        if not isinstance(input, Batch):
            input = Batch.from_data_list([input])
            link_weights = link_weights.unsqueeze(0)
        link_weights = self.softplus(link_weights)  # soft-limit to open lower bound 0

        # iterate over graphs in input batch, constructing nx.Graph and calculating shortest paths for each
        all_selected_edge_dest_idx = []
        for i in range(input.num_graphs):
            cur_input = input.get_example(i)
            cur_input_graph = nx.DiGraph()
            cur_input_graph.add_nodes_from(range(cur_input.num_nodes))
            cur_input_graph.add_edges_from(cur_input.edge_index.t().tolist())
            cur_link_weights = dict(zip(cur_input_graph.edges, link_weights[i]))
            nx.set_edge_attributes(cur_input_graph, cur_link_weights, name="inferredWeight")
            cur_selected_edge_dest_idx = self._link_weights_to_routing(cur_input_graph, "inferredWeight")
            all_selected_edge_dest_idx.append(cur_selected_edge_dest_idx)

        return torch.cat(all_selected_edge_dest_idx, dim=0)


    def _get_link_weights_deterministic(self, input: Batch) -> Tensor:
        raise NotImplementedError

    def _get_link_weights_sampled(self, input: Batch) -> (Tensor, Tensor):
        raise NotImplementedError

    def _evaluate_action(self, input: Batch, action: (Tensor, Tensor)) -> (Tensor, Tensor):
        """
        Returns the logprob and entropy of the given action after obtaining an action distribution from the input
        """
        raise NotImplementedError

    def get_deterministic_action(self, input: Union[Data, Batch]) -> ((Tensor, Tensor), Tensor):
        """
        Returns the action with the highest probability for each edge destination, obtained from
        the actor output
        """
        input_preprocessed = self._preprocess_input(input.clone())
        link_weights = self._get_link_weights_deterministic(input_preprocessed)
        selected_edge_dest_idx = self._get_selected_edge_dest_idx(link_weights.detach().clone(), input)
        value = self._critic(input_preprocessed)
        return (link_weights, selected_edge_dest_idx), value

    def get_sampled_action(self, input: Union[Data, Batch]) -> ((Tensor, Tensor), Tensor, Tensor):
        """
        Returns a sampled action for each edge destination, obtained from the actor output
        """
        input_preprocessed = self._preprocess_input(input.clone())
        link_weights, logprob = self._get_link_weights_sampled(input_preprocessed)
        selected_edge_dest_idx = self._get_selected_edge_dest_idx(link_weights.detach().clone(), input.clone())
        value = self._critic(input_preprocessed)
        return (link_weights, selected_edge_dest_idx), logprob, value

    def evaluate_action(self, input: Union[Data, Batch], action: (Tensor, Tensor)) -> (Tensor, Tensor, Tensor):
        """
        Returns the value, logprob and entropy of the given action after obtaining an action distribution from
        the input
        """
        input_preprocessed = self._preprocess_input(input.clone())
        logprob, entropy = self._evaluate_action(input_preprocessed, action)
        value = self._critic(input_preprocessed)
        return value, logprob, entropy
