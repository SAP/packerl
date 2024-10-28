import torch
import torch_scatter

from utils.types import Key, Dict, Tensor, List, Union, BaseData, ndarray, Batch


def detach(tensor: Union[Tensor, Dict[Key, Tensor], List[Tensor]]) -> \
        Union[ndarray, Dict[Key, ndarray], List[ndarray], BaseData]:
    """
    Tensor detach methods aggregated for multiple datatypes
    """
    if isinstance(tensor, dict):
        return {key: detach(value) for key, value in tensor.items()}
    elif isinstance(tensor, list):
        return [detach(value) for value in tensor]
    if tensor.is_cuda:
        return tensor.cpu().detach().numpy()
    else:
        return tensor.detach().numpy()


def all_tensors_equal(tensor_list):
    """
    Returns True if all tensors in the given list are equal, False otherwise.
    """
    if len(tensor_list) < 2:
        return True

    first_tensor = tensor_list[0]
    for tensor in tensor_list[1:]:
        if not torch.equal(first_tensor, tensor):
            return False
    return True


def get_edge_dest_idx(x: Batch):
    """
    Returns a flattened array of edge indices for an entire batch of graphs.
    """
    # TODO can we do this without the loop?
    expanded_idx = []
    idx_offset = 0
    for cur_graph in x.to_data_list():
        N, E = cur_graph.num_nodes, cur_graph.num_edges
        cur_source_idx = cur_graph.edge_index[0]
        expanded_source_idx = N * cur_source_idx.repeat_interleave(N)
        expanded_node_range = torch.arange(N, device=cur_source_idx.device).repeat(E)
        cur_expanded_idx = expanded_source_idx + expanded_node_range + idx_offset
        expanded_idx.append(cur_expanded_idx)
        idx_offset += N ** 2
    expanded_idx = torch.cat(expanded_idx)

    # concatenate edge destination indices
    return expanded_idx


def scatter_logsumexp(values, idx):
    """
    Calculates the logsumexp of values over the given indices. Used e.g. to safely convert unnormalized logits
    into normalized probabilities.
    """
    max_per_index, _ = torch_scatter.scatter_max(values, idx)
    values_exp = torch.exp(values - max_per_index[idx])  # subtract max per index for numerical stability
    values_sum_exp = torch_scatter.scatter_add(values_exp, idx)
    values_logsumexp = torch.log(values_sum_exp) + max_per_index  # re-add max per index
    return values_logsumexp


def train_stat_dict(val: Tensor, name):
    """
    Returns a dict of statistical scalars for the given tensor, with the given name as prefix.
    """
    return {
        f"{name}_mean": val.mean().item(),
        f"{name}_min": val.min().item(),
        f"{name}_max": val.max().item(),
        f"{name}_std": val.std().item(),
    }
