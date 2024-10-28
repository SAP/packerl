"""
Link failure events are generated here, based on the configuration provided in the scenario configuration.
"""
from typing import List, Dict

import numpy as np

from .link_failure import LinkFailure
from .simple import SimpleLinkFailureConfig, generate_simple_link_failures
from .weibull import WeibullLinkFailureConfig, generate_weibull_link_failures
from ..event import Event
from scenarios.config import BaseConfig, SingleOrChoice


link_failure_generators = {
    "simple": generate_simple_link_failures,
    "weibull": generate_weibull_link_failures,
}

link_failure_config_classes = {
    "simple": SimpleLinkFailureConfig,
    "weibull": WeibullLinkFailureConfig,
}


class LinkFailureConfig(BaseConfig):
    """
    Configuration for link failure events.
    """
    mode: SingleOrChoice[str]
    mode_configs: Dict[str, BaseConfig]

    def __init__(self, **data):
        super().__init__(**data)
        mode_configs = {}
        for mode, cfg_data in data.pop('mode_configs', {}).items():
            if mode not in link_failure_config_classes:
                raise ValueError(f"Unknown event mode: {mode}")
            mode_configs[mode] = link_failure_config_classes[mode](**cfg_data)
        self.mode_configs = mode_configs


def generate_link_failures(G, events: List[List[Event]], scenario_cfg: dict, rng: np.random.Generator):
    """
    Generate link failure events based on the configuration provided in the scenario configuration.
    """

    # preparation
    link_failure_cfg = scenario_cfg['events']['link_failures']
    scenario_length = scenario_cfg['scenario_length']
    ms_per_step = scenario_cfg['ms_per_step']
    chosen_mode = link_failure_cfg['mode']
    if chosen_mode == 'none':
        return
    elif chosen_mode not in link_failure_generators.keys():
        raise ValueError(f"Link failure mode {chosen_mode} not supported")
    chosen_mode_cfg = link_failure_cfg['mode_configs'][chosen_mode]

    # generate link failure events
    _generate_link_failures = link_failure_generators[chosen_mode]
    _generate_link_failures(G, events, scenario_length, ms_per_step, rng, **chosen_mode_cfg)
