"""
This module contains the logic to generate events for a scenario.
Events include traffic demands and link failures.
"""
from typing import List

import numpy as np

from scenarios.config import BaseConfig
from .event import Event
from .link_failures import LinkFailureConfig, generate_link_failures, LinkFailure
from .traffic import TrafficConfig, generate_traffic, TrafficDemand


class EventsConfig(BaseConfig):
    """
    Configuration for event generation. Contains configurations for traffic and link failure event generation.
    """
    traffic: TrafficConfig
    link_failures: LinkFailureConfig


def generate_events(G, scenario_cfg: dict, rng: np.random.Generator, check=False) -> List[List[Event]]:
    """
    Generate events for a scenario based on the provided configuration.
    """

    T = scenario_cfg['scenario_length']
    ms_per_step = scenario_cfg['ms_per_step']
    events_per_step = [[] for _ in range(T)]

    # generate events and place them into the provided events container
    generate_traffic(G, events_per_step, scenario_cfg, rng, check)
    generate_link_failures(G, events_per_step, scenario_cfg, rng)

    # sort events by time (can we do this quicker? E.g. events within a timestep might already be sorted)
    events_per_step = [sorted(events, key=lambda x: x.t) for events in events_per_step]

    if check:
        assert len(events_per_step) == T
        for i, demand_events_this_step in enumerate(events_per_step):
            for j, e in enumerate(demand_events_this_step):
                if j == 0:
                    assert e.t >= ms_per_step * i
                if j > 0:
                    assert e.t >= demand_events_this_step[j - 1].t
                if j == len(demand_events_this_step) - 1:
                    assert e.t < (i + 1) * ms_per_step

    return events_per_step
