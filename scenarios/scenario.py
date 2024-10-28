from typing import List

import numpy as np

from scenarios.events import Event, LinkFailure, TrafficDemand, EventsConfig
from scenarios.topology import TopologyConfig
from scenarios.config import BaseConfig


class ScenarioConfig(BaseConfig):
    """
    Configuration for a network scenario. Contains the configuration for the topology and the events.
    """
    seed: int
    packet_size: int
    scenario_length: int  # number of timesteps in the scenario
    ms_per_step: float  # duration per simulation step, also scales traffic volume
    rng_uniform_deviation: float  # sample uniform from [1-deviation, 1+deviation]
    topology: TopologyConfig
    events: EventsConfig


class NetworkScenario:
    """
    A network scenario consists of a network topology and a sequence of events that occur in the network.
    These events can be traffic demands or link failures.
    """

    def __init__(self, scenario_cfg, network, events):
        self.scenario_cfg = scenario_cfg
        self.network = network
        self.events: List[List[Event]] = events

        # check if traffic and events are of correct length
        if len(self.events) != self.scenario_cfg["scenario_length"]:
            raise ValueError("Number event lists must be equal to the scenario length")

    def get_event_stats(self):
        """
        Returns statistics about the events in the scenario.
        """
        num_events, num_traffic_demands, num_link_failures = 0, 0, 0
        demand_sizes = []

        for events_t in self.events:
            for e in events_t:
                num_events += 1
                if isinstance(e, TrafficDemand):
                    num_traffic_demands += 1
                    demand_sizes.append(e.amount)
                elif isinstance(e, LinkFailure):
                    num_link_failures += 1

        ds_p10, ds_25, ds_50, ds_75, ds_90, ds_max = np.percentile(demand_sizes, [10, 25, 50, 75, 90, 100])

        event_stats = {
            "num_events": num_events,
            "traffic": {
                "num_traffic_demands": num_traffic_demands,
                "num_link_failures": num_link_failures,
                "demand_sizes_avg": np.average(demand_sizes),
                "demand_sizes_p10": ds_p10,
                "demand_sizes_p25": ds_25,
                "demand_sizes_p75": ds_75,
                "demand_sizes_p90": ds_90,
                "demand_size_max": ds_max,
            }
        }
        return event_stats

    def get_topology_stats(self):
        """
        Returns statistics about the topology in the scenario.
        """
        topology_stats = {
            "graph": {
                "num_nodes": self.network.number_of_nodes(),
                "num_edges": self.network.number_of_edges(),
            },
        }
        return topology_stats

    def get_stats(self):
        """
        Returns statistics about the scenario.
        """
        return {"topology": self.get_topology_stats(), "events": self.get_event_stats()}
