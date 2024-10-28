from typing import List

import numpy as np
import networkx as nx

from scenarios.config import BaseConfig, SingleOrChoice
from .link_failure import LinkFailure
from ..event import Event


class WeibullLinkFailureConfig(BaseConfig):
    """
    Configuration for Weibull-distributed link failure events.
    """
    scale: SingleOrChoice[float]
    shape: SingleOrChoice[float]


def generate_weibull_link_failures(G_original, events: List[List[Event]], scenario_length: int,
                                   ms_per_step: float, rng: np.random.Generator, scale, shape, **kwargs):
    """
    Generate link failure events based on a Weibull distribution.
    """

    # preparation
    G = G_original.copy()
    non_cut_edges = sorted(G.edges - nx.bridges(G))
    if len(non_cut_edges) == 0:
        return []

    # generate failure probabilities for all initially non-cut edges using a Weibull distribution
    failure_probs = rng.weibull(shape, size=len(non_cut_edges)) * scale

    for t in range(scenario_length):
        failure_samples = rng.random(size=len(non_cut_edges)) < failure_probs
        failed_non_cut_edges = [edge for edge, failed in zip(non_cut_edges, failure_samples) if failed]

        # iterate over all failed edges and remove them from the graph if they are still non-cut
        # (Some edges might become cut edges due to other edges being removed before them)
        for fst, snd in failed_non_cut_edges:

            # if the edge has become a cut edge in the meantime, we can skip it
            if (fst, snd) not in non_cut_edges:
                continue

            # else: create the event and remove the edge from the graph
            link_failure_t = t * ms_per_step
            link_failure_event = LinkFailure(link_failure_t, fst, snd)
            timestep = int(link_failure_t / ms_per_step)
            events[timestep].append(link_failure_event)

            edges_to_remove_from_failure_probs = [non_cut_edges.index((fst, snd))]
            G.remove_edge(fst, snd)

            # we have to update the set of non-cut edges and disregard all edges that have become cut edges.
            # (If there are no non-cut edges left, we can directly return)
            new_non_cut_edges = sorted(G.edges - nx.bridges(G))
            if len(new_non_cut_edges) == 0:
                return
            for edge in non_cut_edges:
                if edge not in new_non_cut_edges:
                    edges_to_remove_from_failure_probs.append(non_cut_edges.index(edge))

            # update the failure probabilities and the set of non-cut edges
            failure_probs = np.delete(failure_probs, edges_to_remove_from_failure_probs)
            non_cut_edges = new_non_cut_edges
