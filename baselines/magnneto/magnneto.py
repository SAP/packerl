"""
This file contains a wrapper function to use the MAGNNETO baseline in the PackeRL environment.
"""
import tempfile
from pathlib import Path
import warnings

import tensorflow as tf
import torch
import numpy as np

from baselines.magnneto.environment import Environment as MagnnetoEnv
from baselines.magnneto.actor import Actor as MagnnetoActor
from packerl_env import PackerlEnv
from utils.serialization import serialize_topology_txt, serialize_tm
from utils.topology.topology_utils import network_graphs_equal


env_type = "PackeRL"
traffic_profile = "gravity"
routing = "sp"
max_simultaneous_actions = 10


def get_action_magnneto(obs, magnneto_model_path: Path, magnneto_container: dict, capacity_scaling, **kwargs):
    """
    Get the action from the MAGNNETO baseline.
    """
    if 'env' not in kwargs:
        raise RuntimeError("Using the MAGNNETO baseline requires a PackerlEnv to be passed as a keyword argument.")
    env: PackerlEnv = kwargs['env']
    monitoring = env.global_monitoring.copy()

    # calling MAGNNETO requires reading topology/traffic files from disk,
    # so use a temporary directory and serialize the topology/traffic there
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        graph_path = temp_path / env_type / traffic_profile
        graph_path.mkdir(parents=True)

        # write capacities file
        capacities_path = graph_path / "capacities"
        capacities_path.mkdir()
        capacity_fp = str((capacities_path / "graph.txt").resolve())
        with open(capacity_fp, 'w') as capacities_file:
            serialized_topology = serialize_topology_txt(monitoring,
                                                         capacity_scaling=capacity_scaling,
                                                         as_int=True,
                                                         directed=True
                                                         )
            capacities_file.write(serialized_topology)

        # write traffic file
        tm_path = graph_path / "TM"
        tm_path.mkdir()
        with open(str((tm_path / "TM-0").resolve()), 'w') as tm_file:
            serialized_traffic = serialize_tm(env.past_traffic, as_int=True)
            tm_file.write(serialized_traffic)

        # create MAGNNETO env
        magnneto_env = MagnnetoEnv(env_type=env_type,
                                   traffic_profile=traffic_profile,
                                   routing=routing,
                                   base_dir=temp_dir)

        # (re-)initialize the actor only at the start or when we see a new topology
        magnneto_actor = magnneto_container["actor"]
        cached_graph = magnneto_container["graph"]
        if magnneto_actor is None or cached_graph is None or not network_graphs_equal(monitoring, cached_graph):
            with warnings.catch_warnings():  # ignore TF initializer warnings since we load a model directly after
                warnings.simplefilter("ignore", category=UserWarning)
                magnneto_actor = MagnnetoActor(magnneto_env.G, num_features=magnneto_env.num_features)
                magnneto_actor.build()
                model = tf.keras.models.load_model(str(magnneto_model_path.resolve()), compile=False)
                for w_model, w_actor in zip(model.trainable_variables, magnneto_actor.trainable_variables):
                    w_actor.assign(w_model)
            magnneto_container["actor"] = magnneto_actor
            magnneto_container["graph"] = monitoring

        # reset MAGNNETO env (this is done every PackeRL step because in MAGNNETO,
        # an episode equals optimizing for a single TM)
        magnneto_env.initialize_environment(num_sample=0, random_env=False)
        magnneto_env.reset(change_sample=False)
        state = magnneto_env.get_state()

        # run one full optimization (imitates MAGNNETO's PPOAgent)
        horizon = int(np.floor(magnneto_env.n_links * 2.75))  # that's their default
        for i in range(horizon):
            logits = magnneto_actor(state)
            actions = [np.argmax(logits.numpy())]
            for action in actions:
                next_state, reward = magnneto_env.step(action)
            state = next_state

        # get result routing (i.e. all-pairs-shortest-paths result)
        result_routing = magnneto_env.sp_routing

    # post-process the result routing: get involved edges per destination and convert to action tensor
    monitoring = env.monitoring  # compute selected edges on the directed monitoring graph
    involved_edges_per_dst = {k: [] for k in monitoring.nodes}
    for src, paths_per_src in result_routing.items():
        if src not in monitoring.nodes:  # MAGNNETO includes a 'graph_data' item in the result that we don't use/need
            continue
        for dst, path in paths_per_src.items():
            involved_edges = zip(path, path[1:])
            involved_edges_per_dst[dst].extend(involved_edges)
    edges = list(monitoring.edges)
    involved_edge_idx_per_dst = {k: [edges.index(e) for e in set(v)] for k, v in involved_edges_per_dst.items()}
    edge_values_per_dst = [torch.zeros((len(edges), 1)) for _ in monitoring.nodes]
    for involved_edge_idx, edge_values in zip(involved_edge_idx_per_dst.values(), edge_values_per_dst):
        edge_values[involved_edge_idx] = 1
    action_sp = torch.cat(edge_values_per_dst, dim=1)
    selected_edge_dest_idx = action_sp.flatten().nonzero(as_tuple=False).squeeze()
    action = (action_sp.float(), selected_edge_dest_idx)
    value = torch.tensor(0.)
    return action, value
