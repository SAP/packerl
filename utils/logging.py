from typing import Union, List
from flatdict import FlatDict
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import wandb
from PIL import Image

from features.feature_utils import highlight_metrics
from utils.visualization.vis_graph import visualize_monitoring_nx
from utils.visualization.vis_utils import pad_images_to_largest
from utils.visualization.visualization import (visualize_step, visualize_traffic_events,
                                               visualize_ep_performance, visualize_traffic_matrices)
from utils.constants import DEFAULT_OUT_DIR
from utils.utils import list_of_dict_to_dict_of_lists, dict_of_lists_to_dict_of_stats


def colored(text: str, color: int = 189):
    """
    Colors a string to display it fancily in the terminal.
    """
    return(f"\033[38;5;{color}m{text}\033[0;0m")

"""
Defines the logging severity levels as they are defined in ns3. Assigns a number to each level for comparison.
"""
LOG_LEVELS = {
    "none": 7,
    "error": 6,
    "warn": 5,
    "debug": 4,
    "info": 3,
    "function": 2,
    "logic": 1,
    "all": 0
}


class Logger:
    """
    A helper class that mimics the logging methods provided by ns3,
    as well as handling the logging of training and evaluation outcomes to Weights and Biases.

    It is designed as a scoped context manager, initializing a W&B run on enter and finalizing it on exit.
    """
    def __init__(self, config_dict: dict, acceptable_features: dict):
        """
        Initializes the logger by setting the desired logging level.
        Everything with a value lower than that severity level will not be printed.
        """
        self.cfg = SimpleNamespace(**config_dict)
        self.log_level = LOG_LEVELS[self.cfg.log_level]
        self.acceptable_features = acceptable_features
        self.log_path = Path(self.cfg.base_event_dir)

        self._wandb_run = None
        # initialize W&B run if desired
        if self.cfg.use_wandb:
            wandb_config = dict(FlatDict(config_dict, delimiter="."))
            (Path(DEFAULT_OUT_DIR) / "wandb").mkdir(exist_ok=True)
            wandb_dir = os.environ.get("WANDB_DIR", str(Path(DEFAULT_OUT_DIR).resolve()))
            wandb_entity = None if self.cfg.wandb_entity == "" else self.cfg.wandb_entity
            self._wandb_run = wandb.init(config=wandb_config, project="packerl", group=self.cfg.group_id,
                                         entity=wandb_entity, job_type="packerl", name=self.cfg.event_id,
                                         dir=wandb_dir)

        self.metric_stat_keys = None
        self.highlight_stat_keys = None
        self.training_scalar_keys = None
        self.rollout_metrics_all_it_fp = str((self.log_path / "rollout_metrics_all_iterations.csv").resolve())
        self.rollout_highlights_all_it_fp = str((self.log_path / "rollout_highlights_all_iterations.csv").resolve())
        self.training_scalars_all_it_fp = str((self.log_path / "training_scalars_all_iterations.csv").resolve())
        self.eval_metrics_all_it_fp = str((self.log_path / "eval_metrics_all_iterations.csv").resolve())
        self.eval_highlights_all_it_fp = str((self.log_path / "eval_highlights_all_iterations.csv").resolve())
        self.final_metrics_all_ep_fp = str((self.log_path / "final_metrics_all_ep.csv").resolve())
        self.final_metrics_per_ep_fp = str((self.log_path / "final_metrics_per_ep.csv").resolve())
        self.final_highlights_all_ep_fp = str((self.log_path / "final_highlights_all_ep.csv").resolve())
        self.final_highlights_per_ep_fp = str((self.log_path / "final_highlights_per_ep.csv").resolve())

    def __enter__(self):
        """
        Nothing to do here since the W&B run is already initialized in the constructor.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        When exiting the context, the W&B run is finalized.
        """
        self.log_debug("Finalizing logger.")

        if self._wandb_run is not None:
            self._wandb_run.finish(exit_code=0, quiet=True)


    def level_at_least(self, level: str):
        """
        Returns true if the given log level is at least as high as the one the logger is assigned.
        """
        if level not in LOG_LEVELS:
            raise ValueError("given log level invalid")
        return self.log_level >= LOG_LEVELS[level]

    def level_equals(self, level: str):
        """
        Returns true if the given log level is equal to the one the logger is assigned.
        """
        if level not in LOG_LEVELS:
            raise ValueError("given log level invalid")
        return self.log_level == LOG_LEVELS[level]
        
    def _log_line(self, severity: int, prefix: str, msg: Union[str, List[str]]):
        """
        Conditional print of given message msg: only if given severity level is at least as high as the one
        the logger is assigned.
        :param severity: Severity level of the message
        :param msg: message
        """
        if severity >= self.log_level:
            if isinstance(msg, str):
                print(f"{prefix} {msg}")
            elif isinstance(msg, list):
                for line in msg:
                    print(f"{prefix} {line}")

    def log_uncond(self, msg: Union[str, List[str]]):
        self._log_line(LOG_LEVELS["none"], f"{colored('[py:UC]')}", msg)

    def log_error(self, msg: Union[str, List[str]]):
        self._log_line(LOG_LEVELS["error"], f"{colored('[py:ER]')}", msg)

    def log_warn(self, msg: Union[str, List[str]]):
        self._log_line(LOG_LEVELS["warn"], f"{colored('[py:WR]')}", msg)

    def log_debug(self, msg: Union[str, List[str]]):
        self._log_line(LOG_LEVELS["debug"], f"{colored('[py:DB]')}", msg)

    def log_info(self, msg: Union[str, List[str]]):
        self._log_line(LOG_LEVELS["info"], f"{colored('[py:IN]')}", msg)

    def log_function(self, msg: Union[str, List[str]]):
        self._log_line(LOG_LEVELS["function"], f"{colored('[py:FN]')}", msg)

    def log_logic(self, msg: Union[str, List[str]]):
        self._log_line(LOG_LEVELS["logic"], f"{colored('[py:LG]')}", msg)

    # ==============================================================================================

    def log_iteration(self, it, iteration_log):
        """
        Logs the entire content of the given iteration log to disk (and W&B if desired).
        """
        self.log_debug(f"=== iteration {it}: logging ===")
        it_path = self.log_path / f"it{it:03d}"
        it_path.mkdir(parents=True)

        # log rollout to disk and W&B if desired
        self._log_episodes(iteration_log["rollout"], "rollout", it,
                           str((self.log_path / f"it{it:03d}" / "rollout_metrics_per_ep.csv").resolve()),
                           str((self.log_path / f"it{it:03d}" / "rollout_highlights_per_ep.csv").resolve()),
                           self.rollout_metrics_all_it_fp,
                           self.rollout_highlights_all_it_fp
                           )

        # if available, log evaluation to disk and W&B if desired
        if "evaluation" in iteration_log:
            self._log_episodes(iteration_log["evaluation"], "eval", it,
                               str((self.log_path / f"it{it:03d}" / "eval_metrics_per_ep.csv").resolve()),
                               str((self.log_path / f"it{it:03d}" / "eval_highlights_per_ep.csv").resolve()),
                               self.eval_metrics_all_it_fp,
                               self.eval_highlights_all_it_fp
                               )

        # log training step to disk and W&B if desired.
        # This also commits the log (the others log without committing).
        self.log_training_step(iteration_log["training_step"], it)

    def log_training_step(self, training_scalars, it):
        """
        Logs training scalars to disk and W&B if desired.
        """
        if self.training_scalar_keys is None:
            self.training_scalar_keys = list(training_scalars.keys())
        if not os.path.isfile(self.training_scalars_all_it_fp):
            with open(self.training_scalars_all_it_fp, "w") as f:
                f.write(f"{','.join(self.training_scalar_keys)}\n")
        with open(self.training_scalars_all_it_fp, "a") as f:
            scalars_comma_separated = ','.join([str(training_scalars[k]) for k in self.training_scalar_keys])
            f.write(f"{scalars_comma_separated}\n")

        if self._wandb_run is not None:
            self._wandb_run.log({f"training/{k}": v for k, v in training_scalars.items()},
                                step=it, commit=True)

    def log_final_evaluation(self, final_eval_log):
        """
        Logs final evaluation results to disk (and W&B if desired).
        """
        self.log_debug(f"=== final evaluation: logging ===")
        self._log_episodes(final_eval_log, "final_eval", -1,
                           self.final_metrics_per_ep_fp, self.final_highlights_per_ep_fp,
                           self.final_metrics_all_ep_fp, self.final_highlights_all_ep_fp)

    def _log_episodes(self, episodes_log, step_log_prefix, it,
                      metric_list_fp, highlight_list_fp,
                      metric_stats_fp, highlight_stats_fp):
        """
        Log episodes to disk and W&B if desired
        """
        num_ep = len(episodes_log)
        eps_metrics, eps_highlights, eps_scenario_stats = [], [], []

        if step_log_prefix == "rollout":
            vis_dir = None
            num_vis_ep = 0
            ep_names = [f"{step_log_prefix}_it{it:03d}_ep{i:02d}" for i in range(len(episodes_log))]
        elif step_log_prefix == "eval":  # only visualize first episode
            num_vis_ep = max(0, min(num_ep, self.cfg.vis_first_n_evals))
            vis_dir = str((self.log_path / f"it{it:03d}" / f"{step_log_prefix}_vis").resolve())
            Path(vis_dir).mkdir(parents=True)
            ep_names = [f"{step_log_prefix}_it{it:03d}_ep{i:02d}" for i in range(len(episodes_log))]
        else:   # step_log_prefix == "final_eval": shortened ep_name, visualize all episodes
            num_vis_ep = max(0, min(num_ep, self.cfg.vis_first_n_final_evals))
            vis_dir = str((self.log_path / f"{step_log_prefix}_vis").resolve())
            Path(vis_dir).mkdir(parents=True)
            ep_names = [f"{step_log_prefix}_ep{i:02d}" for i in range(len(episodes_log))]

        if not self.cfg.visualize_at_all:
            num_vis_ep = 0
        eps_with_vis = [True] * num_vis_ep + [False] * (num_ep - num_vis_ep)

        for i, (ep_summary, ep_name, ep_with_vis) in enumerate(zip(episodes_log, ep_names, eps_with_vis)):

            # gather episode metrics, converting bytes to MB for better readability
            ep_metrics = {}
            ep_highlights = {"global_ep_reward": ep_summary['global_ep_reward'].item()}
            if "inference_time_ms" in ep_summary:
                ep_highlights.update(inference_time_ms=np.mean(ep_summary["inference_time_ms"]))
            if "step_time_ms" in ep_summary:
                ep_highlights.update(step_time_ms=np.mean(ep_summary["step_time_ms"]))

            # set up regular metrics
            for metric_name, metric in ep_summary['ep_metrics'].items():
                if "Bytes" in metric_name:
                    metric /= 1e6
                    metric_name = metric_name.replace("Bytes", "MB")
                if "Delay" in metric_name:
                    metric *= 1e3
                    metric_name = metric_name.replace("Delay", "Delay_ms")
                if "Jitter" in metric_name:
                    metric *= 1e3
                    metric_name = metric_name.replace("Jitter", "Jitter_ms")
                if metric_name in highlight_metrics:
                    ep_highlights[metric_name] = metric
                else:
                    ep_metrics[metric_name] = metric

            # add global action stats to regular metrics
            for metric_name, metric in ep_summary['ep_action_stats'].items():
                if "vis" not in metric_name:
                    if metric_name in highlight_metrics:
                        ep_highlights[metric_name] = metric
                    else:
                        ep_metrics[metric_name] = metric

            eps_metrics.append(ep_metrics)
            eps_highlights.append(ep_highlights)
            eps_scenario_stats.append(dict(FlatDict(ep_summary['ep_scenario_stats'], delimiter='/')))

            # create and log visualizations directly to save memory
            if ep_with_vis:
                self._log_ep_vis(ep_summary, ep_name, vis_dir, step_log_prefix)

        # log episode metrics and highlights to disk, and W&B if desired
        self._log_metrics(eps_metrics, eps_highlights, eps_scenario_stats,
                          step_log_prefix, it,
                          metric_list_fp, highlight_list_fp,
                          metric_stats_fp, highlight_stats_fp
                          )

    def _log_metrics(self, metrics, highlights, scenario_stats,
                     mode, it,
                     metric_list_fp, highlight_list_fp,
                     metric_stats_fp, highlight_stats_fp):
        """
        Log metrics of multiple episodes to disk and W&B if desired
        """

        metrics = list_of_dict_to_dict_of_lists(metrics)
        highlights = list_of_dict_to_dict_of_lists(highlights)
        scenario_stats = list_of_dict_to_dict_of_lists(scenario_stats)
        metric_stats = dict_of_lists_to_dict_of_stats(metrics)
        highlight_stats = dict_of_lists_to_dict_of_stats(highlights)

        # log metric lists to device (values per episode)
        metric_keys = list(metrics.keys())
        with open(metric_list_fp, "w") as f:
            f.write(f"{','.join(metric_keys)}\n")
            for i in range(len(metrics[metric_keys[0]])):
                f.write(f"{','.join([str(metrics[k][i]) for k in metric_keys])}\n")

        # log highlight lists to device (values per episode)
        highlight_keys = list(highlights.keys())
        with open(highlight_list_fp, "w") as f:
            f.write(f"{','.join(highlight_keys)}\n")
            for i in range(len(highlights[highlight_keys[0]])):
                f.write(f"{','.join([str(highlights[k][i]) for k in highlight_keys])}\n")

        # log metric stats per iteration to device (mean, median, min, max, std)
        if self.metric_stat_keys is None:
            self.metric_stat_keys = list(metric_stats.keys())
        if not os.path.isfile(metric_stats_fp):
            with open(metric_stats_fp, "w") as f:
                f.write(f"{','.join(self.metric_stat_keys)}\n")
        with open(metric_stats_fp, "a") as f:
            metrics_comma_separated = ','.join([str(metric_stats[k]) for k in self.metric_stat_keys])
            f.write(f"{metrics_comma_separated}\n")

        # log highlight stats per iteration to device (mean, median, min, max, std)
        if self.highlight_stat_keys is None:
            self.highlight_stat_keys = list(highlight_stats.keys())
        if not os.path.isfile(highlight_stats_fp):
            with open(highlight_stats_fp, "w") as f:
                f.write(f"{','.join(self.highlight_stat_keys)}\n")
        with open(highlight_stats_fp, "a") as f:
            highlights_comma_separated = ','.join([str(highlight_stats[k]) for k in self.highlight_stat_keys])
            f.write(f"{highlights_comma_separated}\n")

        # log to W&B if desired
        if self._wandb_run is not None:
            wandb_logs = {}
            for metric_name, metric_value in metric_stats.items():
                wandb_logs[f"{mode}/metrics/{metric_name}"] = metric_value
            for highlight_name, highlight_value in highlight_stats.items():
                wandb_logs[f"{mode}/highlights/{highlight_name}"] = highlight_value
            for scenario_stat_name, scenario_stat_values in scenario_stats.items():
                wandb_logs[f"{mode}/scenario_stats/{scenario_stat_name}"] = wandb.Histogram(scenario_stat_values)

            if it >= 0:  # log iterations with step number but without committing because training_step will commit
                self._wandb_run.log(wandb_logs, commit=False, step=it)
            else:  # log final evaluation without step number (i.e. at the end), but with commit
                self._wandb_run.log(wandb_logs, commit=True)

    def _log_ep_vis(self, ep_summary, ep_name, vis_dir, step_log_prefix):
        """
        Logs visualizations of a single episode to disk and W&B if desired
        """

        self.log_info(f"visualizing episode {ep_name} (this takes some time...)")
        ep_step_stats = ep_summary['ep_step_stats']
        ep_action_stats = ep_summary["ep_action_stats"]
        ep_vis = dict()
        monitorings = ep_summary["global_monitorings"]  # from before t=0 to before t=T (i.e. T+1 items)
        states_pyg = ep_summary["state_pyg"]  # from before t=1 to before t=T (i.e. T items)
        actions_raw = ep_summary["action_raw"]  # from before t=1 to before t=T (i.e. T items)
        actions_discretized = ep_summary["action_discretized"]  # from before t=1 to before t=T (i.e. T items)
        observed_traffic = ep_summary["observed_traffic"]
        render_node_pos = ep_summary["graph_node_pos"]

        # rendered performance plots
        self.log_logic(f"visualizing episode {ep_name}: ep_performance")
        ep_plots = visualize_ep_performance(ep_name, len(monitorings), ep_step_stats,
                                            self.cfg.rendering['plot_annotation_fontsize'])
        for plot_name, plot_arr in ep_plots.items():
            ep_vis[f'{ep_name}/ep_plots/{plot_name}'] = plot_arr

        # rendered traffic demands
        self.log_logic(f"visualizing episode {ep_name}: traffic")
        ep_vis[f'{ep_name}/traffic'] = visualize_traffic_events(ep_summary['events'])

        # rendered monitorings: # list of Images
        self.log_logic(f"visualizing episode {ep_name}: monitorings")
        rendered_monitorings = []
        for t, M in enumerate(monitorings):
            self.log_logic(f"visualizing episode {ep_name}: monitoring {t=}")
            rendered_M: Image.Image = visualize_monitoring_nx(M, render_node_pos, self.cfg.rendering, ep_name, t,
                                                              self.cfg.packet_size_with_headers,
                                                              self.cfg.use_flow_control)
            rendered_monitorings.append(rendered_M)
        if self.cfg.rendering['full_step_vis']:
            ep_vis[f'{ep_name}/monitoring'] = pad_images_to_largest(rendered_monitorings)

        # rendered steps (monitoring, state_pyg, action_raw, action_discretized): list of Images
        visualized_steps = []  # list of Images
        self.log_logic(f"visualizing episode {ep_name}: steps")
        for t, (rendered_M, state_pyg, action_raw, action_discretized) \
                in enumerate(zip(rendered_monitorings[:-1], states_pyg, actions_raw, actions_discretized)):
            self.log_logic(f"visualizing episode {ep_name}: step {t=}")
            visualized_step: Image.Image = visualize_step(self.cfg.rendering, self.acceptable_features, render_node_pos,
                                                          rendered_M, state_pyg, action_raw, action_discretized)
            visualized_steps.append(visualized_step)
        ep_vis[f'{ep_name}/step'] = pad_images_to_largest(visualized_steps)

        # rendered action stats
        self.log_logic(f"visualizing episode {ep_name}: action stats")
        for action_stat, vis_data in ep_action_stats.items():
            if "vis" in action_stat:
                if vis_data.ndim == 2:
                    ep_vis[f'{ep_name}/{action_stat}'] = wandb.Image(vis_data)
                else:
                    ep_vis[f'{ep_name}/{action_stat}'] = vis_data

        # rendered observed traffic
        self.log_logic(f"visualizing episode {ep_name}: observed traffic")
        ep_vis[f'{ep_name}/observed_traffic'] = visualize_traffic_matrices(observed_traffic)

        # log visualizations to disk and wandb if desired
        wandb_vis_dict = {}
        for vis_name, vis_data in ep_vis.items():

            # single PIL Image
            if isinstance(vis_data, Image.Image):
                vis_path = (Path(vis_dir) / vis_name)
                vis_path_parent, vis_filename = vis_path.parent, f"{str(vis_path.stem)}.jpg"
                vis_path_parent.mkdir(parents=True, exist_ok=True)
                vis_data.save(vis_path_parent / vis_filename, optimize=True, quality=85)

                if self._wandb_run is not None:
                    wandb_vis_dict[f"{step_log_prefix}/{vis_name}"] = wandb.Image(vis_data)

            # lists of PIL images -> create gif and log as Wandb.Image
            elif isinstance(vis_data, list):
                vis_path_parent = (Path(vis_dir) / vis_name)
                vis_type_name = vis_path_parent.parent.stem
                vis_path_parent.mkdir(parents=True, exist_ok=True)
                for i, vis in enumerate(vis_data):
                    vis_fp = str((vis_path_parent / f"{vis_type_name}{i:03d}.jpg").resolve())
                    vis.save(vis_fp, optimize=True, quality=85)
                vis_anim_fp = str((vis_path_parent / f"{vis_type_name}_anim.gif").resolve())
                vis_data[0].save(vis_anim_fp, save_all=True, append_images=vis_data[1:],
                                 duration=500, loop=0)

                if self._wandb_run is not None:
                    wandb_vis_dict[f"{step_log_prefix}/{vis_name}"] = wandb.Video(vis_anim_fp)

        # since metrics will always be logged thereafter, this does not commit
        if self._wandb_run is not None:
            self._wandb_run.log(wandb_vis_dict, commit=False)
