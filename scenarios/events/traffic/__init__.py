"""
Traffic demands are generated here, based on the configuration provided in the scenario configuration.
"""
from typing import List

import numpy as np
import networkx as nx

from scenarios.config import BaseConfig, SingleOrChoice
from .traffic_demand import TrafficDemand
from ..event import Event

INTERARRIVAL_SAMPLE_SIZE = 10_000  # number of samples to generate at a time for the interarrival time distribution


class TrafficConfig(BaseConfig):
    """
    Configuration for traffic demand event generation.
    """
    prob_tcp: SingleOrChoice[float]
    traffic_scaling: SingleOrChoice[float]
    flow_size_tradeoff: SingleOrChoice[float]  # from 0.0 (more but smaller flows) to 1 (fewer but larger flows)
    max_demand_size: int  # maximum size of a demand in bytes
    max_demand_interarrival_time: int  # maximum time between two demands in seconds
    demand_size_distribution_scale: float  # scale parameter of the demand size distribution
    demand_size_distribution_shape_base: float  # shape parameter of the demand size distribution
    demand_interarrival_distribution_shape: float  # shape parameter of the demand interarrival time distribution


def get_datarates(rng: np.random.Generator, demand_sizes: np.array, demand_is_tcp: np.array):
    """
    Generate data rates for the traffic demands based on the demand sizes and the demand types.
    """
    data_rates = np.zeros_like(demand_sizes)

    # Apply conditions, excluding where demand_is_tcp is True (because these will adjust their data rates themselves)
    small_udp_demand = ~demand_is_tcp & (demand_sizes < 1e5)
    large_udp_demand = ~demand_is_tcp & (demand_sizes >= 1e5)

    # Generate data rates based on conditions
    data_rates[small_udp_demand] = 1e9 * np.ones(np.sum(small_udp_demand))
    data_rates[large_udp_demand] = rng.uniform(1e6, 5e6, size=np.sum(large_udp_demand))

    return data_rates.astype(int)


def generate_traffic(G, events: List[List[Event]], scenario_cfg: dict, rng: np.random.Generator, check=False):
    """
    Generate traffic demand events based on the configuration provided in the scenario configuration.
    """

    # preparation
    traffic_cfg = scenario_cfg['events']['traffic']
    traffic_scaling = traffic_cfg['traffic_scaling']
    prob_tcp = traffic_cfg['prob_tcp']
    flow_size_tradeoff = traffic_cfg["flow_size_tradeoff"]
    ms_per_step = scenario_cfg['ms_per_step']
    total_time = scenario_cfg['scenario_length'] * ms_per_step
    max_interarrival_time = traffic_cfg['max_demand_interarrival_time']

    # obtain "traffic matrix", i.e. expected traffic per src-dst pair
    node_traffic_potentials = list(nx.get_node_attributes(G, 'traffic_potential').values())
    node_traffic_potentials = np.array(node_traffic_potentials)[:, np.newaxis]
    if check:
        assert np.all(node_traffic_potentials >= 0)
    tm = np.matmul(node_traffic_potentials, node_traffic_potentials.T)  # create potential matrix
    tm = tm * rng.uniform(low=1 - scenario_cfg['rng_uniform_deviation'],
                          high=1 + scenario_cfg['rng_uniform_deviation'],
                          size=tm.shape)  # random perturbation
    np.fill_diagonal(tm, 0)
    tm = (total_time * tm) / np.sum(tm)  # normalize to sum=total_time
    if check:
        assert np.all(tm >= 0)

    # log-logistic interarrival distribution: Uses uniform samples and a quantile function, then caps interarrival time
    did_shape = traffic_cfg["demand_interarrival_distribution_shape"]
    did_scale = (flow_size_tradeoff + 0.2) / (traffic_scaling * G.graph['traffic_scaling'])

    def sample_interarrivals(size):
        uniform_samples = rng.uniform(0, 1, size)
        log_logistic_samples = did_scale * ((uniform_samples / (1 - uniform_samples)) ** (1 / did_shape))
        return np.minimum(log_logistic_samples, max_interarrival_time)

    # pareto demand size distribution: Samples from Lomax distribution (called 'pareto' in np) and post-processes
    dsd_shape = traffic_cfg['demand_size_distribution_shape_base'] + np.log(np.power(0.1 + flow_size_tradeoff, -1 / 37))
    dsd_scale = traffic_cfg['demand_size_distribution_scale']

    def sample_demand_sizes(size):
        lomax_samples = rng.pareto(dsd_shape, size)
        pareto_samples = dsd_scale * (lomax_samples + 1)
        return np.minimum(pareto_samples, traffic_cfg['max_demand_size']).astype(int)

    # get interarrival times (for all src-dst pairs at once, we'll assign them to specific src-dst pairs later)
    interarrival_times = sample_interarrivals(INTERARRIVAL_SAMPLE_SIZE)
    cumsum_time = np.sum(interarrival_times)
    while cumsum_time < np.sum(tm):
        new_samples = sample_interarrivals(INTERARRIVAL_SAMPLE_SIZE)
        interarrival_times = np.append(interarrival_times, new_samples)
        cumsum_time += np.sum(new_samples)
    if check:
        assert np.all(interarrival_times >= 0)

    # get number of demands and splitting indices, depending on the traffic matrix
    ia_time_cumsum = np.cumsum(interarrival_times)  # [0.0 ... smth. >= (np.sum(tm) * total_time)]
    tmf = tm.flatten()
    cum_demand_split_idx = np.searchsorted(ia_time_cumsum, np.cumsum(tmf), side='right')
    num_demands_per_tm_entry = np.diff(cum_demand_split_idx, prepend=0)
    if check:
        assert np.all(num_demands_per_tm_entry >= 0)
    num_demands = np.sum(num_demands_per_tm_entry)

    # remove extra samples and overflowing split indices and thresholds
    # (this happens whenever we generated exactly the number of demands needed)
    ia_time_cumsum = ia_time_cumsum[:num_demands]
    pruned_tmf = tmf[cum_demand_split_idx < num_demands]
    cum_demand_split_idx = cum_demand_split_idx[cum_demand_split_idx < num_demands]

    # get demand arrival times by subtracting the cumulative arrival time of the corresponding traffic matrix entry
    arrival_time_sub = np.zeros_like(ia_time_cumsum)
    arrival_time_sub_idx, inv_idx = np.unique(cum_demand_split_idx, return_inverse=True)
    arrival_time_sub_vals = np.bincount(inv_idx, weights=pruned_tmf)
    arrival_time_sub[arrival_time_sub_idx] = arrival_time_sub_vals
    arrival_time_sub = np.cumsum(arrival_time_sub)
    demand_arrival_times = ia_time_cumsum - arrival_time_sub
    demand_arrival_times = (total_time * demand_arrival_times) / np.repeat(tmf, num_demands_per_tm_entry)

    # the resulting arrival times should be within the total time,
    # and monotonically increasing within the splits per src-dst pair
    if check:
        assert np.all(demand_arrival_times < total_time)
        assert (np.sum(
            np.concatenate([np.diff(m, prepend=0) for m in np.split(demand_arrival_times, cum_demand_split_idx)]) < 0)
                == 0)

    # get demand sizes, types and datarates
    demand_sizes = sample_demand_sizes(num_demands)
    demand_is_tcp = rng.random(num_demands) < prob_tcp
    demand_datarates = get_datarates(rng, demand_sizes, demand_is_tcp)

    # get source and destination per demand
    coordinates = np.indices(tm.shape).flatten().reshape(2, -1).T  # shape: [n*n, 2]
    demand_coordinates = np.repeat(coordinates, num_demands_per_tm_entry, axis=0)  # shape: [num_demands, 2]

    # create demand events from the generated data and insert them into the provided event container
    for i in range(len(demand_coordinates)):
        src, dst = demand_coordinates[i]
        arrival_time = demand_arrival_times[i]
        new_demand = TrafficDemand(t=arrival_time, src=src, dst=dst, amount=demand_sizes[i],
                                   datarate=demand_datarates[i], is_tcp=demand_is_tcp[i])
        demand_timestep = int(arrival_time / ms_per_step)
        events[demand_timestep].append(new_demand)
