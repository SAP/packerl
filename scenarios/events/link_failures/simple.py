from typing import List

import numpy as np
import networkx as nx

from scenarios.config import BaseConfig, SingleOrChoice
from .link_failure import LinkFailure
from ..event import Event


class SimpleLinkFailureConfig(BaseConfig):
    """
    Configuration for simple link failure events.
    """
    num_failures: SingleOrChoice[int]


def generate_simple_link_failures(G_original, events: List[List[Event]], scenario_length: int,
                                  ms_per_step: float, rng: np.random.Generator, num_failures, **kwargs):
    """
    Generate simple link failure events. These are evenly distributed over the scenario length.
    """

    G = G_original.copy()

    # first, generate the desired number of link failures
    desired_event_times = np.linspace(0, scenario_length * ms_per_step, num_failures + 2)[1:-1]

    # for each desired event time, choose a random edge that is not a cut edge (if possible)
    for t in desired_event_times:
        non_cut_edges = sorted(G.edges - nx.bridges(G))
        if len(non_cut_edges) == 0:
            break
        fst, snd = rng.choice(non_cut_edges)

        link_failure_event = LinkFailure(t, fst, snd)
        timestep = int(t / ms_per_step)
        events[timestep].append(link_failure_event)

        # remove edge from graph so that future failures happen on other, still non-cut edges
        G.remove_edge(fst, snd)
