import time

import torch
from tqdm import tqdm

from utils.utils import list_of_dict_to_dict_of_lists


def evaluate(config, env, sp_provider, logger, step_func, reset_func, get_action_func, it=-1, is_final=False):
    """
    Evaluation loop. If training, this is also called after every N training iterations.
    This needs the step, reset and get_action functions to be provided because learnable policies might also
    need to handle/use their normalizers.
    """
    eval_episode_summaries = []
    if is_final:
        num_episodes = config.final_eval_episodes
        eval_log_msg = "final evaluation"
    else:
        num_episodes = config.evaluation_episodes
        eval_log_msg = f"evaluation #{it}"
    eval_range = range(num_episodes * config.ep_length)
    if logger.level_equals("debug"):
        eval_range = tqdm(eval_range, desc=eval_log_msg)
    else:
        logger.log_uncond(f"=== {eval_log_msg} ({num_episodes} ep) ===")

    # --- evaluation loop ---
    env.eval_enabled = True
    env.running = False  # force initial env.reset() in upcoming evaluation loop
    obs = None
    step_logs_per_ep = []
    eval_ep_number = 0
    visualize_this_ep = False
    for i in eval_range:

        if not env.running:
            if is_final and i == 0:
                logger.log_uncond("Resetting data generator for final evaluation.")
                obs = reset_func(reset_rng=True, new_rng_seed=config.final_eval_seed)
            else:
                obs = reset_func()
                eval_ep_number += 1
            if config.visualize_at_all:
                visualize_this_ep = (is_final and eval_ep_number <= config.vis_first_n_final_evals or
                                     (not is_final and eval_ep_number <= config.vis_first_n_evals))

        # get deterministic action. Env needs to be provided because the shortest path calculators need the raw obs.
        start_time = time.time()
        action, value = get_action_func(obs, env=env, sp_provider=sp_provider)
        inference_time_ms = round((time.time() - start_time) * 1000, 3)

        # reshaping results for logging and execution
        N, E = obs.num_nodes, obs.num_edges
        action_values, selected_edge_dest_idx = action
        action_values = action_values.cpu()
        selected_edge_dest_idx = selected_edge_dest_idx.cpu()
        if "next_hop" in config.actor_critic_mode:
            action_raw = action_values.reshape(E, N)
        else:
            action_raw = action_values.flatten()

        # link-weight based policies can use the past link weights as input feature -> store them in env
        if config.link_weights_as_input:
            env.last_link_weight = action_values

        # prepare action for env and execute it
        action_one_hot = torch.zeros(E * N).scatter_(0, selected_edge_dest_idx, 1).reshape(E, N)
        action_for_env = obs.clone()  # copy obs to get the same graph structure
        action_for_env.__setattr__("edge_attr", action_one_hot)
        obs, _, _, _, infos = step_func(action_for_env)

        # log step results
        step_log = infos | {"inference_time_ms": inference_time_ms}
        if visualize_this_ep:
            step_log |= {
                "state_pyg": obs,
                "action_raw": action_raw,
                "action_discretized": action_one_hot,
            }
        step_logs_per_ep.append(step_log)

        # collect episode summary
        done = float(infos["done"])
        if done:
            ep_summary = step_log.pop("ep_summary")  # remove ep_summary from step_log before converting to dict
            if not visualize_this_ep:
                del ep_summary["global_monitorings"]
                del ep_summary["ep_step_stats"]
                del ep_summary["graph_name"]
                del ep_summary["graph_node_pos"]
                del ep_summary["events"]
                del ep_summary["observed_traffic"]
            ep_summary |= list_of_dict_to_dict_of_lists(step_logs_per_ep)
            eval_episode_summaries.append(ep_summary)
            step_logs_per_ep = []

    # --- evaluation loop end ---
    env.eval_enabled = False
    env.running = False  # force initial env.reset() in next iteration
    return eval_episode_summaries
