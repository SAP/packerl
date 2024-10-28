import subprocess
import os
import time
from typing import Optional
from copy import deepcopy
from types import SimpleNamespace
from collections import defaultdict

import torch
import numpy as np
import py_interface as ns3ai
import networkx as nx
from torch_geometric.utils import from_networkx, degree
from torch_scatter import scatter_max, scatter_add

from features.feature_utils import is_performance_feature, aggregate_metrics
from reward.reward_module import RewardModule
from scenarios import Generator as ScenarioGenerator
from scenarios.events import LinkFailure, TrafficDemand
from utils.tensor import all_tensors_equal
from utils.topology.sp_calculator import ShortestPathCalculator
from utils.types import Data as PygData, GlobalMonitoring, RewardDict
from utils.shared_memory.interaction import place_shm, read_shm, check_shm
from utils.shared_memory.structs import PackerlEnvStruct, PackerlActStruct
from utils.logging import Logger
from utils.constants import NS3_DIR, SHM_SIZE


class PackerlEnv:
    """
    This environment is an interface for learning routing optimization in computer networks,
    using the packet-level network simulator ns-3 (https://www.nsnam.org) as a backend.
    The environment provides network states in the form of their current topology as well
    as network configuration (e.g. delays, datarates) and current load and performance
    metrics such as link utilization and queue load, or packet loss/avg. packet delay.
    The agents/policy shall provide routing actions for each participating router that
    get installed in the simulation and used as routing functions for the upcoming timestep.
    The reward correlates with the overall network performance, including metrics such as
    minimizing dropped packets, maximizing network throughput or minimizing flow completion time.

    This environment uses the 'scenarios' package to provide a wide range of different
    network scenarios to represent the diversity of networking conditions encountered in the wild.
    On each reset() call, it is consulted to provide a new network scenario (topology,
    configuration and traffic) which is installed on the interfaced ns-3 simulation.

    This environment is designed to be used as a scoped context manager, because certain
    resources (e.g. shared memory) need to be freed after the simulation is done.
    """
    def __init__(self, cfg: SimpleNamespace, logger: Logger,
                 sp_provider: ShortestPathCalculator, acceptable_features: dict):
        """
        Initializes the environment, which includes setting up the shared memory interface,
        the ns-3 simulation, the data generator, the reward module and some bookkeeping.
        """
        self.cfg = cfg
        self.logger = logger
        self.sp_provider = sp_provider
        self.acceptable_features = acceptable_features

        # ns-3 interfacing
        if self.cfg.mempool_key < 1001:
            raise ValueError("mempool_key has to be larger than 1000")
        ns3ai.Init(self.cfg.mempool_key, SHM_SIZE)
        self.logger.log_info("initialized shared memory pool")
        self.experiment = ns3ai.Experiment(self.cfg.mempool_key,
                                           SHM_SIZE,
                                           'packerl',
                                           NS3_DIR,
                                           self.cfg.build_ns3)
        self.sim = None  # initialized in reset()
        self.logger.log_info("crated ns3ai experiment")

        # data generator
        if not self.cfg.is_baseline_run:
            self.train_scenario_generator: ScenarioGenerator = self._setup_scenario_generator("train")
        self.eval_scenario_generator: ScenarioGenerator = self._setup_scenario_generator("eval")

        # reward module
        self._reward_module = RewardModule(self.cfg.reward_preset, self.cfg.global_reward_ratio)

        # misc and bookkeeping
        self.eval_enabled = False
        self.running = False
        self.ns3_run_id = 0  # used to control RNG in ns3. Can be left at 0 for now, since we don't do multiple runs per generated scenario
        self.t = 0
        self.training_step = 0
        self.chance_to_attach_given = False

    def __enter__(self):
        """
        Nothing special to do here, just return self
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Properly deletes the member attributes 'sim' and 'experiment' and calls
        a bash script to free shared memory
        :return:
        """
        self.logger.log_debug("deleting environment...")
        del self.sim
        del self.experiment
        subprocess.run(os.path.join(NS3_DIR, "contrib/ns3-ai/freeshm.sh"), check=True)

    @property
    def global_monitoring(self) -> GlobalMonitoring:
        """
        :return: the latest global monitoring
        """
        if len(self._global_monitorings) < 1:
            raise RuntimeError("global_monitoring() got called before any monitoring was received, "
                               "which should not happen!")
        return self._global_monitorings[-1]

    @property
    def monitoring(self):
        """
        :return: the latest monitoring (i.e. observation prior to conversion to pyg;
         global or local depending on selected obs mode)
        """
        if self.cfg.obs_mode == "global":
            return self.global_monitoring
        else:
            raise NotImplementedError("non-global obs mode not yet implemented")

    @property
    def global_observation(self):
        if len(self._global_observations) < 1:
            raise RuntimeError("global_observation() got called before any observation was received, "
                               "which should not happen!")

        if self.cfg.use_n_last_obs == 1:
            global_obs = self._global_observations[-1]

        else:  # self.cfg.use_n_last_obs > 1
            if len(self._global_observations) < self.cfg.use_n_last_obs:
                obs_list = ([self._global_observations[0]] * (self.cfg.use_n_last_obs - len(self._global_observations))
                            + self._global_observations)
            else:
                obs_list = self._global_observations[-self.cfg.use_n_last_obs:]

            # if the edge_index has changed (e.g. link failure),
            # adjust edge information of all observations to match the last one.
            # This works because for now we only delete edges, not add new ones.
            if not all_tensors_equal([o.edge_index for o in obs_list]):
                last_obs = obs_list[-1]
                last_obs_edge_list = last_obs.edge_index.t().tolist()
                last_obs_edge_set = set(map(tuple, last_obs_edge_list))
                filtered_obs_list = []

                for obs in obs_list[:-1]:
                    obs_edge_list = obs.edge_index.t().tolist()
                    obs_edge_set = set(map(tuple, obs_edge_list))
                    common_edge_set = obs_edge_set.intersection(last_obs_edge_set)
                    mask = torch.tensor([tuple(edge) in common_edge_set for edge in obs_edge_list])

                    filtered_obs = obs.clone()
                    filtered_obs.__setattr__("edge_index", obs.edge_index[:, mask])
                    filtered_obs.__setattr__("edge_attr", obs.edge_attr[mask])
                    filtered_obs_list.append(filtered_obs)

                # Append the last graph as it is
                filtered_obs_list.append(last_obs)
                obs_list = filtered_obs_list

            global_obs = self._global_observations[-1].clone()
            global_obs.__setattr__("x", torch.cat([o.x for o in obs_list], dim=1))
            global_obs.__setattr__("edge_attr", torch.cat([o.edge_attr for o in obs_list], dim=1))
            global_obs.__setattr__("u", torch.cat([o.u for o in obs_list], dim=1))

        return global_obs

    @property
    def observation(self):
        """
        :return: the latest observation (global or local depending on selected obs mode)
        """
        if self.cfg.obs_mode == "global":
            return self.global_observation
        else:
            raise NotImplementedError("non-global obs mode not yet implemented")

    @property
    def global_reward(self):
        """
        :return: the latest global reward
        """
        if len(self._global_rewards) < 1:
            raise RuntimeError("global_reward() got called before any reward was received, "
                               "which should not happen!")
        return self._global_rewards[-1]

    @property
    def global_ep_reward(self):
        """
        :return: the sum of all global rewards received so far
        """
        if len(self._global_rewards) < 1:
            raise RuntimeError("global_ep_reward() got called before any reward was received, "
                               "which should not happen!")
        return torch.sum(torch.stack(self._global_rewards))

    @property
    def local_reward(self):
        """
        :return: the latest local reward
        """
        if len(self._local_rewards) < 1:
            raise RuntimeError("local_reward() got called before any reward was received, "
                               "which should not happen!")
        return self._local_rewards[-1]

    @property
    def mixed_reward(self):
        """
        :return: the latest mixed reward (global and local reward combined via global_reward_ratio)
        """
        if len(self._mixed_rewards) < 1:
            raise RuntimeError("mixed_reward() got called before any reward was received, "
                               "which should not happen!")
        return self._mixed_rewards[-1]

    @property
    def past_events(self):
        """
        :return: the events that happened in the past time step
        """
        events = self.scenario.events
        events_horizon = len(events)
        past_t = self.t - 1
        if self.t > events_horizon:
            raise ValueError(f"current time step t ({self.t}) exceeds scenario events length ({events_horizon})")
        elif self.t == 0:
            return []
        else:
            return events[past_t]

    @property
    def upcoming_events(self):
        """
        :return: the events that are scheduled for the upcoming time step
        """
        events = self.scenario.events
        events_horizon = len(events)
        if self.t > events_horizon:
            raise ValueError(f"current time step t ({self.t}) exceeds scenario events length ({events_horizon})")
        elif self.t == events_horizon:
            return []
        else:
            return events[self.t]

    @property
    def past_traffic(self):
        """
        :return: the traffic demands that were observed in the past time step
        """
        traffic = self._observed_traffic
        if len(traffic) < 1:
            raise RuntimeError("past_traffic() got called before any traffic was observed, which should not happen!")
        return traffic[-1]

    @property
    def _scenario_generator(self):
        """
        return the data generator for the current mode (train or eval)
        """
        if self.eval_enabled:
            return self.eval_scenario_generator
        else:
            return self.train_scenario_generator

    def _setup_scenario_generator(self, mode: str):
        """
        Creates a data generator for the given mode (train or eval) and returns it.
        :param mode: The mode (train or eval)
        """
        scenario_presets = deepcopy(self.cfg.scenario_presets[mode])
        scenario_custom_cfg = self.cfg.scenario_custom_cfg[mode] | {
            'seed': self.cfg.seed if mode == "train" else self.cfg.eval_seed,  # different data for eval and train
            'packet_size': self.cfg.packet_size,
            'scenario_length': self.cfg.ep_length,
            'ms_per_step': self.cfg.ms_per_step,
        }
        generator = ScenarioGenerator(presets=scenario_presets, custom_cfg=scenario_custom_cfg)
        return generator

    def reset(self, **kwargs):
        """
        The actual reset method resets the simulator, creates a new scenario
        and does the necessary bookkeeping.
        :return: Observations and state of the new scenario prios to the first action
        """

        # bookkeeping
        self.running = True
        self.t = 0
        self.ns3_run_id = 0  # used to control RNG in ns3. Can be left at 0 for now, since we don't do multiple runs per generated scenario
        if self.eval_enabled:
            self.logger.log_info("Upcoming episode is evaluation episode")
        self._global_monitorings = []
        self._global_observations = []
        self._observed_traffic = []
        self._global_rewards = []
        self._stored_actions = []
        self._local_rewards = []
        self._mixed_rewards = []
        self._ep_metrics = dict()

        # use suitable data generator to create scenario for this episode (reset rng if desired)
        cur_scenario_generator = self._scenario_generator
        if kwargs.get("reset_rng", False):
            new_rng_seed = kwargs.get("new_rng_seed", None)
            cur_scenario_generator.reset_rng(seed=new_rng_seed)
        self.scenario = cur_scenario_generator.generate_scenario()

        # link-weight policies can use the last link weights as input. Therefore, we initialize them here
        # (2*E for directed edges). They will get updated from the training algorithm or the evaluation loop
        if self.cfg.link_weights_as_input:
            if self.cfg.link_weights_as_input_mode == "random":
                self.last_link_weights = torch.rand(self.scenario.network.number_of_edges() * 2, dtype=torch.float) * 3 + 1  # random weights in [1, 4]
            else:
                self.last_link_weights = torch.ones(self.scenario.network.number_of_edges() * 2, dtype=torch.float)  # constant weights

        self.logger.log_debug(f"upcoming scenario: {self.scenario.network}")
        self.logger.log_logic(f"{self.scenario.network.nodes}")
        self.logger.log_logic(f"{self.scenario.network.edges}")
        self.logger.log_debug(f" - its traffic demands per step: {[len([e for e in evs if isinstance(e, TrafficDemand)]) for evs in self.scenario.events]}")
        self.logger.log_debug(f" - link failures per step: {[len([e for e in evs if isinstance(e, LinkFailure)]) for evs in self.scenario.events]}")

        # initialize and start ns-3 simulator, and link shared memory block with ns3
        self.experiment.reset()
        memblock_key = 0  # when implementing vectorized envs at some later point, use range(0, num_envs-1)
        self.sim = ns3ai.Ns3AIRL(uid=memblock_key, EnvType=PackerlEnvStruct, ActType=PackerlActStruct)
        experiment_setting = {
            'configFilePath': self.cfg.config_fp,
            "run": self.ns3_run_id,  # acts as rng seed for ns3 and thus ensures that every episode is different
            "outDir": self.cfg.base_event_dir,
            "memblockKey": memblock_key
        }
        self.logger.log_info(f"mempool key: {hex(self.cfg.mempool_key)} {memblock_key=}")
        self.experiment.run(setting=experiment_setting,
                            show_output=True,
                            log_modules=self.cfg.ns3_log_modules,
                            log_level=self.cfg.log_level,
                            profile_ns3=self.cfg.profiling_cpp
                            )

        # check whether ns3 is ready to interact via shared memory
        check_shm(self.sim, self.logger)
        self.logger.log_info(f"started sim experiment")

        # give user a chance to attach debugger to ns3 process
        if self.cfg.debug_cpp and not self.chance_to_attach_given:
            time.sleep(0.5)  # this just makes the following line appear at the bottom of your console
            self.logger.log_uncond(f"\nGiving you a chance to attach your debugger to the ns3 process (10 seconds).")
            time.sleep(10)

        # Initialize episode interaction by communicating network graph and reading initial monitoring
        place_shm(self.sim, self.cfg.sim_timeout, self.logger, "network_graph", self.scenario.network)
        if self.sim.isFinish():
            raise RuntimeError("ns-3 simulation finished before it even started. This should not happen.")
        self._get_monitoring_and_obs()
        return self.observation

    def step(self, actions):
        """
        This env's step method places the given actions alongside the latest traffic demands
        into the shared memory so that the simulation installs them and simulates the specified
        amount of time, monitoring the network performance and sending that back to this env
        via the shared memory.
        :param actions: The actions to be taken
        :return: (Observations, Rewards, Terminateds, Truncateds, Infos)
        """
        self.logger.log_function(f"PackerlEnv.step()")
        if not self.running:
            raise RuntimeError("Can't run step() if experiment is not running. Call reset() first.")

        # start step timer (we include shm communication time)
        step_start = time.time()

        # place upcoming events (traffic demands, link failures etc.) and actions in shared memory.
        # After having placed actions completely, the simulation runs a timestep.
        place_shm(self.sim, self.cfg.sim_timeout, self.logger, "events", self.upcoming_events)
        place_shm(self.sim, self.cfg.sim_timeout, self.logger, "actions", actions)

        # ========================================
        # ... meanwhile, ns3 runs the sim step ...
        # ========================================

        self._stored_actions.append(actions.detach().cpu())
        self.t += 1
        if not self.eval_enabled:
            self.training_step += 1

        # obtain results: _get_monitoring_and_obs() is a blocking operation until ns3 has finished the sim step.
        self._get_monitoring_and_obs()

        # the actual step is finished; measure step time and process results
        step_time_ms = (time.time() - step_start) * 1000
        obs = self.observation
        all_rewards = self._collect_reward(actions)
        reward = self.global_reward
        terminated = False  # there is no terminal state, our ns3 simulations end with our last timestep.
        truncated = self.t >= self.cfg.ep_length or self.sim.isFinish()
        done = terminated or truncated
        infos = {"done": done, "step_time_ms": step_time_ms}
        place_shm(self.sim, self.cfg.sim_timeout, self.logger, "done", done)
        reward_dict_str = {k: str(round(reward, 4)) for k, reward in all_rewards.items()}
        self.logger.log_info(f"PackerlEnv.step(): t={self.t}, r={reward_dict_str}")

        # handle end of episode
        if done:
            self.running = False

            # create episode summary
            ep_summary = {
                "global_monitorings": self._global_monitorings,
                "global_ep_reward": self.global_ep_reward,
                "ep_metrics": self._get_ep_metrics(),
                "ep_scenario_stats": self.scenario.get_stats(),
                "ep_action_stats": self._get_ep_action_stats(),
                "ep_step_stats": self._get_ep_step_stats(),
                "graph_name": self.scenario.network.graph['name'],
                "graph_node_pos": nx.get_node_attributes(self.scenario.network, "pos"),
                "events": self.scenario.events,
                "observed_traffic": self._observed_traffic,
            }
            infos.update(ep_summary=ep_summary)

        return (obs, reward, terminated, truncated, infos)

    def _monitoring_to_obs(self, input_monitoring: nx.DiGraph) -> PygData:
        """
        1. create a copy of the networkx graph that only keeps the node,
        edge and global features of a list called accepted_features.
        2. convert to Pytorch geometric data object
        :param input_monitoring: the input network monitoring in networkx format
        :return: The converted Pytorch geometric data object
        """
        monitoring = input_monitoring.copy()

        # if we store past link weights, add them as edge feature using nx.set_edge_attributes()
        if self.cfg.link_weights_as_input:
            last_link_weights_dict = {(u, v): self.last_link_weights[i].item()
                                      for i, (u, v) in enumerate(monitoring.edges())}
            nx.set_edge_attributes(monitoring, last_link_weights_dict, name="linkWeight")

        # calculate shortest path distance between all node pairs (spdist) if desired
        node_distances = self.sp_provider.get_node_distances(monitoring)
        node_distance_lists = [[sv for k, sv in sorted(spd.items())] for k, spd in sorted(node_distances.items())]
        spdist = torch.tensor(node_distance_lists).flatten()  # shape: [N**2, ]

        # explicitly ADD global features
        global_features = [value for global_feat, value in list(monitoring.graph.items())
                           if global_feat in self.acceptable_features["global"]]

        # remove globals from networkx graph so that they don't overwrite node/edge features in conversion.
        # The global features are added back after conversion
        monitoring.graph = {}

        # explicitly REMOVE node features that are not accepted
        for _, node_feat in monitoring.nodes(data=True):
            for feat in list(node_feat.keys()):
                if feat not in self.acceptable_features["node"]:
                    del node_feat[feat]

        # explicitly REMOVE edge features that are not accepted
        for _, _, edge_feat in monitoring.edges(data=True):
            for feat in list(edge_feat.keys()):
                if feat not in self.acceptable_features["edge"]:
                    del edge_feat[feat]

        # DIY for node features (PyG conversion somehow doesn't work for node features)
        node_attrs = self.acceptable_features["node"]
        data_x = None
        if node_attrs:
            data_dict = defaultdict(list)
            for i, (_, feat_dict) in enumerate(monitoring.nodes(data=True)):
                if set(feat_dict.keys()) != set(node_attrs):
                    raise ValueError('Not all nodes contain the same attributes')
                for key, value in feat_dict.items():
                    data_dict[str(key)].append(value)
            xs = [data_dict[key] for key in node_attrs]
            data_x = torch.tensor(xs, dtype=torch.float).t()

        # convert the networkx graph to a pytorch geometric data object
        data = from_networkx(monitoring, group_node_attrs=None,
                             group_edge_attrs=self.acceptable_features["edge"] or None)

        # finalize features, re-adding globals and making sure everything is float
        data.__setattr__("spdist", spdist.float())
        if data.edge_attr is None:
            data.__setattr__("edge_attr", torch.empty(size=(data.num_edges, 0), dtype=torch.float))
        else:
            data.__setattr__("edge_attr", data.edge_attr.float())
        if data_x is None:
            data.__setattr__("x", torch.empty(size=(data.num_nodes, 0), dtype=torch.float))
        else:
            data.__setattr__("x", data_x)
        data.__setattr__("u", torch.tensor(global_features, dtype=torch.float).unsqueeze(0))
        return data

    def _get_monitoring_and_obs(self):
        """
        This method reads the shared memory block that contains the monitoring data
        and converts it to a Pytorch geometric data object.
        The first is stored in a list of global monitorings, the latter in a list of global observations.
        """
        global_monitoring, past_matrices, dropped_packets_per_reason, dropped_bytes_per_reason \
            = read_shm(self.sim, self.cfg.sim_timeout, self.logger, "monitoring")
        self._global_monitorings.append(global_monitoring)
        self._observed_traffic.append(past_matrices['sent_bytes'])

        # collect metrics for episode statistics
        cur_ep_metrics = global_monitoring.graph | dropped_bytes_per_reason | dropped_packets_per_reason
        for metric, value in cur_ep_metrics.items():
            if is_performance_feature(metric, "global"):
                if metric in self._ep_metrics:
                    self._ep_metrics[metric].append(value)
                else:
                    self._ep_metrics[metric] = [value]

        global_observation = self._monitoring_to_obs(global_monitoring)
        self._global_observations.append(global_observation)

    def _collect_reward(self, actions: Optional[PygData]) -> RewardDict:
        """
        Calculate the reward for the agents based on the current global network state and local observations.
        :param actions: The actions taken in the current timestep
        :return: The calculated reward
        """
        reward_input = {
            "global_monitoring": self.global_monitoring,
            "global_observation": self.global_observation,
            "local_monitoring": None,  # TODO implement
            "local_observation": None,  # TODO implement
            "actions": actions
        }
        rewards = self._reward_module.collect_reward(reward_input)

        self._global_rewards.append(rewards["global"])
        self._local_rewards.append(rewards["local"])
        self._mixed_rewards.append(rewards["mixed"])
        return rewards["all"]

    def _get_ep_metrics(self):
        """
        :return: A dict containing the episode's statistics
        """
        ep_metrics = {m_name: aggregate_metrics(m_values, metric_name=m_name)
                      for (m_name, m_values) in self._ep_metrics.items()}
        # NOTE: this disregards packets that are neither received nor dropped -> should handle those in ns-3
        received_and_dropped_bytes = ep_metrics["receivedBytes"] + ep_metrics["droppedBytes"]
        if received_and_dropped_bytes > 0:
            drop_ratio = ep_metrics["droppedBytes"] / received_and_dropped_bytes
        else:
            drop_ratio = 0.0
        max_sent_bytes_per_step = aggregate_metrics(self._ep_metrics["sentBytes"], aggregator_str="max")
        max_received_bytes_per_step = aggregate_metrics(self._ep_metrics["receivedBytes"], aggregator_str="max")
        ep_metrics.update(dropRatio=drop_ratio, maxSentBytesPerStep=max_sent_bytes_per_step,
                          maxReceivedBytesPerStep=max_received_bytes_per_step)
        return ep_metrics

    def _get_ep_step_stats(self):
        """
        :return: A dict containing the episode's step-wise statistics
        """
        graph_ep_stats = ['avgPacketDelay', 'maxPacketDelay', 'avgPacketJitter',
                          'sentPackets', 'receivedPackets', 'droppedPackets', 'retransmittedPackets',
                          'sentBytes', 'receivedBytes', 'droppedBytes', 'retransmittedBytes',
                          'maxLU', 'avgTDU']
        cumsum_stats = ['sentPackets', 'receivedPackets', 'droppedPackets', 'retransmittedPackets',
                        'sentBytes', 'receivedBytes', 'droppedBytes', 'retransmittedBytes']
        ep_stats = {step_stat: [] for step_stat in graph_ep_stats}
        for g in self._global_monitorings:
            for step_stat in graph_ep_stats:
                ep_stats[step_stat].append(g.graph[step_stat])
        for cs_stat in cumsum_stats:
            ep_stats[cs_stat] = np.cumsum(ep_stats[cs_stat])
        ep_stats['global_rewards'] = [np.nan] + self._global_rewards
        ep_stats['local_rewards'] = [np.nan] + self._local_rewards
        ep_stats['mixed_rewards'] = [np.nan] + self._mixed_rewards
        return ep_stats

    def _get_ep_action_stats(self):
        """
        :return: A dict containing the episode's step-wise action statistics
        esrc = "edge source", edst = "edge destination", ddst = "demand destination"
        "next_hops" = an [N, N] tensor per timestep that contains the next_hop node ID
         per pair of current routing node and demand destination node.
        """
        initial_obs = self._global_observations[0]
        N, T = initial_obs.num_nodes, len(self._stored_actions)  # assumes that our node set doesn't change
        if N < 2:
            raise ValueError("Can't calculate action stats for networks with less than 2 nodes")
        next_hops = -torch.ones(N, N, T, dtype=torch.long)
        node_degrees = torch.zeros(N, T, dtype=torch.long)
        for t, actions_t in enumerate(self._stored_actions):
            esrc_t = actions_t.edge_index[0]
            node_degrees[:, t] = degree(esrc_t, num_nodes=N)
            _, next_hops_t = scatter_max(actions_t.edge_attr, dim=0, index=esrc_t, dim_size=N)  # shape: (N, N)
            next_hops[:, :, t] = next_hops_t

        # oscillation: 0 when routing next_hops never change in between timesteps,
        # 1 when next_hops change every timestep
        next_hop_switches = torch.abs(next_hops[:, :, 1:] - next_hops[:, :, :-1])
        next_hop_switch_counts = (next_hop_switches != 0).sum(dim=-1)  # shape: [N, N]
        oscillation_ratio = next_hop_switch_counts.float() / (T - 1)  # shape: [N, N]
        oscillation_ratio_global = oscillation_ratio.mean()  # shape: [N, N]

        # next_hop spread: 0 when routing nodes use the same next_hop for all destinations,
        # 1 when next_hops follow a uniform spread per routing node
        next_hop_probs = scatter_add(torch.ones_like(next_hops), index=next_hops, dim=1).float() / N  # shape: [N, E, T]
        next_hop_logits = torch.clamp(torch.log(next_hop_probs), min=torch.finfo(next_hop_probs.dtype).min)
        next_hop_entropy = -(next_hop_probs * next_hop_logits).sum(dim=1)  # shape: [N, T]
        next_hop_spread = 1 - (next_hop_entropy / np.log(N)).mean(dim=1)  # shape: [N]
        next_hop_spread_global = next_hop_spread.mean()

        return {
            "oscillation_ratio_vis": oscillation_ratio.numpy(),  # per esrc-ddst pair
            "oscillation_ratio_global": oscillation_ratio_global.item(),
            "next_hop_spread_vis": next_hop_spread.numpy(),  # per esrc node
            "next_hop_spread_global": next_hop_spread_global.item(),
        }
