"""
We obtain baseline routing algorithms here, using the following sources:
- Random routing: random_link_weight, random_next_hop
- Shortest Path First (SPF) routing: spf
- MAGNNETO models: magnneto1, magnneto2, ...
"""

from baselines.random import get_action_random_next_hop, get_action_random_link_weight
from baselines.spf import get_action_spf

baseline_modes = ["random_link_weight", "random_next_hop", "spf",
                  "magnneto1", "magnneto2", "magnneto3", "magnneto4",
                  "magnneto5", "magnneto6", "magnneto7", "magnneto8",
                  "magnneto9", "magnneto10", "magnneto11", "magnneto12",
                  "magnneto13", "magnneto14", "magnneto15", "magnneto16"
                  ]


def is_baseline_mode(routing_mode):
    """
    returns True if the given routing_mode is a baseline mode, False otherwise.
    """
    return routing_mode in baseline_modes


def get_baseline_action_func(config):
    """
    returns a baseline (i.e. an action provider function) depending on the given routing_mode.
    """
    if not is_baseline_mode(config.routing_mode):
        raise RuntimeError(f"Unknown actor mode: {config.routing_mode} (not a baseline mode)")
    if config.routing_mode == "random_next_hop":
        return get_action_random_next_hop
    elif config.routing_mode == "random_link_weight":
        return get_action_random_link_weight
    elif config.routing_mode == "spf":
        return get_action_spf
    elif "magnneto" in config.routing_mode and config.routing_mode in baseline_modes:
        # extract the seed from the routing
        magnneto_seed = int(config.routing_mode.replace("magnneto", ""))
        # check whether the corresponding magnneto model exists
        from pathlib import Path
        magnneto_model_path = Path(__file__).parent / "magnneto" / "saved_models" / f"seed{magnneto_seed}" / "actor"
        if not magnneto_model_path.exists():
            raise RuntimeError(f"Could not find magnneto model for seed {magnneto_seed}. "
                               f"Please make sure that the model exists at the correct location.")
        # check whether tensorflow is installed (import here to avoid tf warning spam when not using MAGNNETO)
        try:
            import tensorflow as tf
        except ImportError as e:
            raise RuntimeError(f"Error importing tensorflow: {e.msg}. Please make sure you've got it installed.")
        from baselines.magnneto.magnneto import get_action_magnneto
        from functools import partial

        return partial(get_action_magnneto,
                       magnneto_model_path=magnneto_model_path,
                       magnneto_container={"actor": None, "graph": None},  # need to put this stuff in a container to persist across calls
                       capacity_scaling=0.001 * config.ms_per_step
                       )
    else:
        raise NotImplementedError(f"Baseline mode {config.routing_mode} not registered in get_action_func()")
