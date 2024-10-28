import sys
sys.path.append(".")

import numpy as np
import pytest
from pathlib import Path
from itertools import product

from scenarios import Generator
from scenarios.topology import generate_topology_attributes, generate_topology_graph
from scenarios.events import generate_traffic, generate_link_failures


def test_init_pure():
    """
    Test if the generator can be initialized without any presets or custom configurations.
    """
    _ = Generator(presets={}, custom_cfg={})


PRESETS_PATH = Path(__file__).parent.parent / 'scenarios' / 'presets'


def get_presets(category: str):
    """
    Get a list of presets for a given category.
    """
    return [pn.stem for pn in sorted((PRESETS_PATH / category).glob('*.yaml'))]


def combine_presets(gps, aps, tps, lfps):
    """
    Combine presets from different categories.
    """
    combined_presets = []
    for (gp, ap, tp, lfp) in product(gps, aps, tps, lfps):
        p = {
            'topology': {
                'graph': gp,
                'attributes': ap,
            },
            'events': {
                'traffic': tp,
                'link_failures': lfp,
            },
        }
        combined_presets.append(p)
    return combined_presets


"""
The following tests are parameterized to test possible combinations of the following presets.
"""
g_preset_list = get_presets('topology/graph')
a_preset_list = get_presets('topology/attributes')
t_preset_list = [p for p in get_presets('events/traffic') if "tm_" not in p]
lf_preset_list = get_presets('events/link_failures')
all_presets = combine_presets(g_preset_list, a_preset_list, t_preset_list, lf_preset_list)


@pytest.mark.parametrize("preset_param", all_presets)
def test_init_presets(preset_param):
    """
    Test if the generator can be initialized with different presets.
    """
    _ = Generator(presets=preset_param, custom_cfg={})


@pytest.mark.parametrize("preset_param", all_presets)
def test_sample(preset_param):
    """
    Test if the generator can sample a scenario configuration.
    """
    generator = Generator(presets=preset_param, custom_cfg={})
    for _ in range(20):
        _ = generator.scenario_cfg_set.sample(generator.rng)


standard_preset = {'topology': {'graph': 'random_XL', 'attributes': 'default'},
                   'events': {'traffic': 'default', 'link_failures': 'simple_1'}
                   }

custom_cfgs = [{'seed': 1235},
               {'seed': 1235, 'events': {'traffic': {'traffic_scaling': 2.0}}},
               ]


@pytest.mark.parametrize("custom_cfg_param", custom_cfgs)
def test_init_custom_cfg(custom_cfg_param):
    """
    Test if the generator can be initialized with custom configurations.
    """
    _ = Generator(presets=standard_preset, custom_cfg=custom_cfg_param)


selected_g_presets = ["random_XS", "random_M"]
selected_a_presets = ["default"]
selected_t_presets = ["default"]
selected_lf_presets = ["simple_1", "weibull_easy"]
selected_presets = combine_presets(selected_g_presets, selected_a_presets, selected_t_presets, selected_lf_presets)


@pytest.mark.parametrize("preset_param", [{}] + all_presets)
def test_topology_graph(preset_param):
    """
    Test if the generator can generate a topology graph.
    """
    generator = Generator(presets=preset_param, custom_cfg={})

    scenario_cfg = generator.scenario_cfg_set.sample(generator.rng)
    _ = generate_topology_graph(scenario_cfg, generator.rng, generator.generated_network_rng_id)


@pytest.mark.parametrize("preset_param", [{}] + all_presets)
def test_topology_attributes(preset_param):
    """
    Test if the generator can generate topology attributes.
    """
    generator = Generator(presets=preset_param, custom_cfg={})
    scenario_cfg = generator.scenario_cfg_set.sample(generator.rng)
    G = generate_topology_graph(scenario_cfg, generator.rng, generator.generated_network_rng_id)
    _ = generate_topology_attributes(G, scenario_cfg, generator.rng)


@pytest.mark.parametrize("preset_param", [{}] + selected_presets)
def test_traffic(preset_param):
    """
    Test if the generator can generate traffic events.
    """
    for graph_preset in selected_g_presets:
        if 'topology' not in preset_param:
            preset_param['topology'] = {}
        preset_param['topology']['graph'] = graph_preset
        generator = Generator(presets=preset_param, custom_cfg={})
        for _ in range(5):
            scenario_cfg = generator.scenario_cfg_set.sample(generator.rng)
            G = generate_topology_graph(scenario_cfg, generator.rng, generator.generated_network_rng_id)
            G = generate_topology_attributes(G, scenario_cfg, generator.rng)
            T = scenario_cfg['scenario_length']
            events_per_step = [[] for _ in range(T)]
            generate_traffic(G, events_per_step, scenario_cfg, generator.rng, check=True)


@pytest.mark.parametrize("fs_tradeoff", list(np.linspace(0.0, 1.0, 11).round(1)))
def test_traffic_flow_lambda(fs_tradeoff):
    """
    Test if the generator can generate traffic events with different flow size trade-offs.
    """
    custom_cfg = {"events": {"traffic": {"flow_size_tradeoff": fs_tradeoff}}}
    generator = Generator(presets={"topology": {'graph': "random_M"}}, custom_cfg=custom_cfg)
    for _ in range(10):
        scenario_cfg = generator.scenario_cfg_set.sample(generator.rng)
        G = generate_topology_graph(scenario_cfg, generator.rng, generator.generated_network_rng_id)
        G = generate_topology_attributes(G, scenario_cfg, generator.rng)
        T = scenario_cfg['scenario_length']
        events_per_step = [[] for _ in range(T)]
        generate_traffic(G, events_per_step, scenario_cfg, generator.rng, check=True)


@pytest.mark.parametrize("preset_param", [{}] + selected_presets)
def test_link_failures(preset_param):
    """
    Test if the generator can generate link failure events.
    """
    for graph_preset in selected_g_presets:
        if 'topology' not in preset_param:
            preset_param['topology'] = {}
        preset_param['topology']['graph'] = graph_preset
        generator = Generator(presets=preset_param, custom_cfg={})
        for _ in range(5):
            scenario_cfg = generator.scenario_cfg_set.sample(generator.rng)
            G = generate_topology_graph(scenario_cfg, generator.rng, generator.generated_network_rng_id)
            G = generate_topology_attributes(G, scenario_cfg, generator.rng)
            T = scenario_cfg['scenario_length']
            events_per_step = [[] for _ in range(T)]
            generate_traffic(G, events_per_step, scenario_cfg, generator.rng, check=True)
            generate_link_failures(G, events_per_step, scenario_cfg, generator.rng)


@pytest.mark.parametrize("preset_param", [{}] + all_presets)
def test_scenario(preset_param):
    """
    Test if the generator can generate a complete scenario.
    """
    generator = Generator(presets=preset_param, custom_cfg={})
    scenario = generator.generate_scenario()
    _ = scenario.get_stats()
