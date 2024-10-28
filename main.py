from types import SimpleNamespace
from pathlib import Path
import yaml
import random
import os

from cw2 import experiment, cw_error
from cw2 import cluster_work
from cw2.cw_data import cw_logging
import torch
import numpy as np

from baselines import get_baseline_action_func
from packerl_env import PackerlEnv
from features.feature_utils import get_acceptable_features
from rl.algorithm.on_policy.ppo import PPO
from utils.topology.sp_calculator import get_shortest_path_calculator
from utils.logging import Logger
from utils.shared_memory.structs import ALL_SHM_SIZES
from utils.constants import DEFAULT_CONFIG_FP, ICMP_HEADER_SIZE, IPV4_HEADER_SIZE, RUN_CONFIG_FILENAME
from utils.evaluation import evaluate
from utils.serialization import yaml_dump_quoted
from utils.utils import deep_update, ensure_sysctl_value


def setup_and_run(config_dict):
    """
    The actual heart of the framework. Here, we set up the needed components,
    run the training algorithm if desired, and evaluate the setup.
    """
    config = SimpleNamespace(**config_dict)

    # seeding
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # initialize the list of monitoring features that will be considered in the run
    acceptable_features = get_acceptable_features(config)

    # initialize a shortest-path calculator that either provides auxiliary shortest-path distances,
    # or takes over the role of a policy during evaluation
    sp_ref_values = {
        "ospfw_ref_value": config.ospfw_ref_value,
        "eigrp_ref_datarate": config.eigrp_ref_datarate,
        "eigrp_ref_delay": config.eigrp_ref_delay,
        "eigrp_ref_multiplier": config.eigrp_ref_multiplier,
    }
    sp_provider = get_shortest_path_calculator(config.sp_mode, sp_ref_values)

    # run!
    with Logger(config_dict, acceptable_features) as logger:

        logger.log_info(f"shm sizes: {ALL_SHM_SIZES}")

        with PackerlEnv(config, logger, sp_provider, acceptable_features) as env:

            if config.is_baseline_run:
                eval_get_action_func = get_baseline_action_func(config)
                final_eval_summary = evaluate(config, env, sp_provider, logger,
                                              step_func=env.step,
                                              reset_func=env.reset,
                                              get_action_func=eval_get_action_func,
                                              is_final=True)
            else:
                algorithm = PPO(config, acceptable_features, logger, env, sp_provider)
                final_eval_summary = algorithm.train_and_evaluate()

        logger.log_final_evaluation(final_eval_summary)


# ==============================================================================


def main(cw2_config_dict: dict):
    """
    Main entry point for the PackeRL framework. Here, we mostly handle configuration.
    """

    # extract default config as dict
    with open(str(Path(DEFAULT_CONFIG_FP).resolve()), 'r') as default_config_file:
        config_dict = dict(yaml.safe_load(default_config_file))

    # override default config dict with provided parameters
    deep_update(config_dict, cw2_config_dict)

    # update config with derived values
    run_pid = os.getpid()
    steps_per_rollout = config_dict['episodes_per_rollout'] * config_dict['ep_length']
    minibatch_size = steps_per_rollout // config_dict['num_minibatches']
    config_dict_update = {
        "mempool_key": run_pid + 1000,  # mempool_keys for the shared memory provided by ns3-ai have to start from 1001
        "packet_size_with_headers": config_dict['packet_size'] + IPV4_HEADER_SIZE + ICMP_HEADER_SIZE,
        "steps_per_rollout": steps_per_rollout,
        "minibatch_size": minibatch_size,
        "is_baseline_run": config_dict['routing_mode'] != "learned",
        "seed": config_dict['seed'] + config_dict['seed_offset']
    }
    config_dict.update(**config_dict_update)

    # config feasibility checks
    if steps_per_rollout % config_dict['num_minibatches'] != 0:
        raise ValueError(f"Number of steps per rollout ({steps_per_rollout})"
                         f" = ep_length ({config_dict['ep_length']})"
                         f" * eps. per rollout ({config_dict['episodes_per_rollout']})"
                         f" must be divisible by number of minibatches ({config_dict['num_minibatches']})")
    if minibatch_size < 2:
        raise ValueError("Need at least two steps per minibatch")
    if config_dict['episodes_per_rollout'] < 1:
        raise ValueError("Need at least one rollout episode")
    if config_dict['ep_length'] < 1:
        raise ValueError("Need at least one step per episode")
    if config_dict["device"] == "cuda" and not torch.cuda.is_available():
        print("cuda not available, using cpu")
        config_dict["device"] = "cpu"
    if config_dict["link_weights_as_input"] and "link_weight" not in config_dict["actor_critic_mode"]:
        print(f"WARNING: can't use link weights as input feature "
              f"for non-link-weight actor-critic mode '{config_dict['actor_critic_mode']}'. Disabling...")
        config_dict["link_weights_as_input"] = False

    # write config to file in new event dir
    base_event_dir = Path(config_dict['base_event_dir'])
    base_event_dir.mkdir(parents=True, exist_ok=True)
    config_dict_fp = str((Path(config_dict['base_event_dir']) / RUN_CONFIG_FILENAME).resolve())
    config_dict["config_fp"] = config_dict_fp
    with open(config_dict_fp, 'w') as config_dict_file:
        yaml_dump_quoted(config_dict, config_dict_file)

    # for C++ profiling, we use linux's 'perf' for profiling the ns3 code and simply change
    # the command to call ns3 via ns3-ai. This requires special privileges in
    # 'kernel.{kernel.perf_event_paranoid,kptr_restrict}'.
    if config_dict['profiling_cpp']:
        print("requested C++ profiling -> checking kernel permissions...")
        ensure_sysctl_value('kernel.perf_event_paranoid', -1)
        ensure_sysctl_value('kernel.kptr_restrict', 0)

    # for python profiling, we simply wrap the run statement in a cProfile object,
    # and extract stats after the run.
    if config_dict['profiling_py']:
        import cProfile
        import pstats
        with cProfile.Profile() as profile:
            setup_and_run(config_dict)
            results = pstats.Stats(profile)
        results.sort_stats(pstats.SortKey.TIME)
        results.dump_stats(os.path.join(config_dict['base_event_dir'], "profiling_py.prof"))

    # if not using python profiling, we just start the run.
    else:
        setup_and_run(config_dict)


# ==============================================================================


class PackeRL(experiment.AbstractExperiment):
    """
    A cw2 experiment wrapper for the PackeRL framework.
    This lets us do parallel PackeRL runs on any compute machine using cw2.
    See documentation of cw2 at https://github.com/ALRhub/cw2/tree/master/doc for further information.
    """
    def initialize(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        pass

    def run(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:

        config['params']['seed'] = rep  # use rep numbers for random seeds
        config['params']['base_event_dir'] = config.get('_rep_log_path')
        config['params']['event_id'] = config.get('_experiment_name')
        config['params']['group_id'] = config.get('name')
        # execute main method with provided config
        main(config['params'])

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        pass


# ==============================================================================


if __name__ == "__main__":
    # Set up the cw object and distribute the runs to the chosen scheduler.
    cw = cluster_work.ClusterWork(PackeRL)
    cw.run()
