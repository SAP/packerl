import yaml
from pathlib import Path

import numpy as np

from scenarios.scenario import NetworkScenario, ScenarioConfig
from scenarios.topology import generate_topology
from scenarios.events import generate_events
from utils.utils import merge_dicts


PRESETS_PATH = Path(__file__).parent / 'presets'


def get_category_config(preset_category: str, preset_value: str):
    """
    Load the default configuration for a category and merge it with the requested preset configuration.
    """

    with open(PRESETS_PATH / preset_category / 'default.yaml', 'r') as default_file:
        category_config = yaml.safe_load(default_file)
    requested_preset = PRESETS_PATH / preset_category / f'{preset_value}.yaml'
    if requested_preset.exists():
        with open(requested_preset, 'r') as preset_file:
            preset_cfg = yaml.safe_load(preset_file)
            merge_dicts(category_config, preset_cfg)
    else:
        raise ValueError(f"Requested preset {preset_value} for category {preset_category} does not exist")
    return category_config


class Generator:
    """
    The Generator class is used to generate network scenarios based on a given configuration.
    It holds the current random number generator (rng) and the current scenario configuration set.
    From this configuration set, it can generate a network scenario by sampling concrete values for the configuration.
    """

    rng = None
    generated_network_rng_id = 0  # the i-th generated graph since the rng last got reset. Used for naming.
    scenario_cfg_set: ScenarioConfig = None

    def __init__(self, presets: dict, custom_cfg: dict):
        """
        Initializes the generator with a given set of presets and custom configuration.
        """

        # load presets (using defaults if not specified) and create scenario cfg
        with open(PRESETS_PATH / 'default.yaml', 'r') as default_file:
            cfg_dict = yaml.safe_load(default_file)
        if "topology" not in cfg_dict:
            cfg_dict["topology"] = {}
        g_preset = presets['topology']['graph'] if (
                    'topology' in presets and 'graph' in presets['topology']) else "default"
        cfg_dict['topology']['graph'] = get_category_config('topology/graph', g_preset)
        a_preset = presets['topology']['attributes'] if (
                    'topology' in presets and 'attributes' in presets['topology']) else "default"
        cfg_dict['topology']['attributes'] = get_category_config('topology/attributes', a_preset)
        if "events" not in cfg_dict:
            cfg_dict["events"] = {}
        t_preset = presets['events']['traffic'] if (
                    'events' in presets and 'traffic' in presets['events']) else "default"
        cfg_dict['events']['traffic'] = get_category_config('events/traffic', t_preset)
        lf_preset = presets['events']['link_failures'] if (
                    'events' in presets and 'link_failures' in presets['events']) else "default"
        cfg_dict['events']['link_failures'] = get_category_config('events/link_failures', lf_preset)

        merge_dicts(cfg_dict, custom_cfg)
        self.scenario_cfg_set = ScenarioConfig(**cfg_dict)

        # reset rng
        self.reset_rng()

    def reset_rng(self, seed: int = None):
        """
        Resets the rng
        """
        if seed is not None:
            self.scenario_cfg_set.seed = seed
        self.rng = np.random.default_rng(self.scenario_cfg_set.seed)
        self.generated_network_rng_id = 0

    def generate_scenario(self):
        """
        Generates a network scenario by sampling a configuration from the current configuration set,
        generating the network topology and events, and returning a NetworkScenario object.
        """

        # sample scenario config
        scenario_cfg = self.scenario_cfg_set.sample(self.rng)

        # create network topology (= topology graph and attributes), and events (traffic and e.g. link failures)
        G = generate_topology(scenario_cfg, self.rng, self.generated_network_rng_id)
        events = generate_events(G, scenario_cfg, self.rng)

        # finalize and return
        scenario = NetworkScenario(scenario_cfg, G, events)
        self.generated_network_rng_id += 1
        return scenario
