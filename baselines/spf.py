import torch

from packerl_env import PackerlEnv
from utils.topology.sp_calculator import ShortestPathCalculator


def get_action_spf(_, **kwargs):
    """
    Returns a selection of routing edges that corresponds to the shortest paths from sources to destinations,
    calculated by the sp_provider in the respective sp_mode (e.g. EIGRP or OSPF).
    """
    if 'env' not in kwargs:
        raise RuntimeError("Using the SPF baselines requires a PackerlEnv to be passed as a keyword argument.")
    if 'sp_provider' not in kwargs:
        raise RuntimeError("Using the SPF baselines requires an sp_provider to be passed as a keyword argument.")
    env: PackerlEnv = kwargs['env']
    sp_provider: ShortestPathCalculator = kwargs['sp_provider']

    monitoring = env.monitoring
    action_sp = sp_provider.get_sp_actions(monitoring)
    selected_edge_dest_idx = action_sp.flatten().nonzero(as_tuple=False).squeeze()
    action = (action_sp.float(), selected_edge_dest_idx)
    value = torch.tensor(0.)
    return action, value