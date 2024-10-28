from typing import Optional
from pathlib import Path

from tqdm import tqdm
import torch
import numpy as np
from torch_scatter import scatter_mean, scatter_max

from rl.algorithm.on_policy.ppo_util import get_policy_loss, get_entropy_loss, get_value_loss
from rl.algorithm.on_policy.buffers import get_buffer
from rl.normalizer.observation import get_obs_normalizer, ObservationNormalizer
from rl.normalizer.reward import get_reward_normalizer, RewardNormalizer
from rl.nn.actor_critic import get_actor_critic
from utils.utils import list_of_dict_to_dict_of_lists
from utils.evaluation import evaluate
from utils.tensor import detach, train_stat_dict

class PPO:
    """
    The class implements the PPO algorithm. It holds the required actor-critic component, and replay 'buffer'
    (only contains the experience of the current rollout since it's on-policy). It also uses normalizers.
    """
    def __init__(self, config, acceptable_features, logger, env, sp_provider):
        """
        Initializes the PPO algorithm with the provided configuration and components.
        """
        self._config = config
        if self._config.spf_pretraining > self._config.training_iterations:
            self._config.spf_pretraining = self._config.training_iterations
            logger.log_info(f"setting spf_pretraining (was {self._config.spf_pretraining}) "
                            f"to training_iterations ({self._config.training_iterations})")
        self._logger = logger
        # if multiple obs are stacked and provided to the policy, we need to stack the acceptable features accordingly
        self._acceptable_features = {k: v * self._config.use_n_last_obs for k, v in acceptable_features.items()}
        self._feature_counts = {k: len(v) for k, v in self._acceptable_features.items()}

        self._env = env
        self._sp_provider = sp_provider

        self._buffer = get_buffer(self._config.value_scope,
                                buffer_size=self._config.steps_per_rollout,
                                gae_lambda=self._config.gae_lambda,
                                discount_factor=self._config.discount_factor,
                                device=self._config.device)
        actor_critic_config = {
            "actor_critic_mode": self._config.actor_critic_mode,
            "actor_mode": self._config.actor_mode,
            "critic_mode": self._config.critic_mode,
            "concat_rev_edges": self._config.concat_rev_edges,
            "initial_exploration_coeff": self._config.initial_exploration_coeff,
            "epsilon_decay": self._config.epsilon_decay,
            "min_epsilon": self._config.min_epsilon,
        }
        self._policy = get_actor_critic(actor_critic_config, self._config.nn, self._feature_counts,
                                        self._config.value_scope, self._config.learning_rate, self._config.device)
        self._policy.train(False)  # policy will remain in eval mode except for when in training_step()

        # load components if specified
        if self._config.load_experiment != "":
            exp_load_dir = self._config.base_event_dir.replace(self._config.group_id, self._config.load_experiment)
            # check whether we find a model in the exact directory (indicating that we match training configuration).
            if Path(exp_load_dir).exists():
                self.load_components(exp_load_dir)
            # else, check whether we find a single model (per rep/seed) in the load_experiment directory
            # (indicating that we use a single training run for all eval configurations).
            else:
                exp_load_dir_parts = Path(exp_load_dir).parts
                eld_idx = exp_load_dir_parts.index(self._config.load_experiment)
                single_run_dir_parts = exp_load_dir_parts[:eld_idx + 1] + exp_load_dir_parts[eld_idx + 2:]
                single_run_dir = Path(*single_run_dir_parts)
                if single_run_dir.exists():
                    self.load_components(single_run_dir)
                else:
                    raise FileNotFoundError(f"no matching model found in {exp_load_dir},"
                                            f"and no general model found in {single_run_dir}")
        else:
            self._obs_normalizer = get_obs_normalizer(self._config, self._acceptable_features, self._feature_counts)
            self._reward_normalizer = get_reward_normalizer(self._config)

    @property
    def policy(self):
        return self._policy

    def _reset_env_and_normalizers(self, **kwargs):
        """
        Resetting the env requires resetting the normalizers, too.
        """
        obs = self._env.reset(**kwargs)
        obs = self._obs_normalizer.reset(obs)
        self._reward_normalizer.reset()
        return obs

    def _eval_step(self, action):
        """
        For the evaluation step we also need to normalize the resulting observation.
        """
        obs, reward, terminated, truncated, infos = self._env.step(action)
        obs = self._obs_normalizer.normalize(obs)
        return obs, reward, terminated, truncated, infos

    def _eval_get_action(self, obs, **kwargs):
        """
        For the evaluation step we use the policy to get a deterministic action.
        """
        with torch.no_grad():
            action, value = self._policy.get_deterministic_action(obs.clone().to(self._policy.device))
        return action, value

    def load_components(self, exp_load_dir):
        """
        Loads the state dict of our model as well as normalizer statistics from the disk.
        """
        model_state_dict_path = Path(exp_load_dir) / f"{self._config.load_mode}_model_state_dict.pth"
        self._policy.load_state_dict(torch.load(model_state_dict_path))

        obs_normalizer_path = Path(exp_load_dir) / f"{self._config.load_mode}_obs_normalizer.pkl"
        self._obs_normalizer = ObservationNormalizer.load(obs_normalizer_path)

        reward_normalizer_path = Path(exp_load_dir) / f"{self._config.load_mode}_reward_normalizer.pkl"
        self._reward_normalizer = RewardNormalizer.load(reward_normalizer_path)

    def save_components(self, file_prefix, msg="saving components..."):
        """
        Stores the state dict of our model as well as normalizer statistics on the disk.
        """
        self._logger.log_info(msg)

        model_state_dict_path = Path(self._config.base_event_dir) / f"{file_prefix}_model_state_dict.pth"
        torch.save(self._policy.state_dict(), model_state_dict_path)

        obs_normalizer_path = Path(self._config.base_event_dir) / f"{file_prefix}_obs_normalizer.pkl"
        self._obs_normalizer.save(obs_normalizer_path)

        reward_normalizer_path = Path(self._config.base_event_dir) / f"{file_prefix}_reward_normalizer.pkl"
        self._reward_normalizer.save(reward_normalizer_path)

    def _update_epoch(self, rollout_data):
        """
        A single update epoch on a given minibatch. Prepares the rollout data, evaluates the taken actions,
        calculates the losses and performs an optimization step.
        """

        train_step_scalars = {}

        # unpack rollout data
        observations = rollout_data.observations.to(self._config.device).detach()
        action_values, selected_idx = rollout_data.actions
        actions = action_values.to(self._config.device).detach(), selected_idx.to(self._config.device).detach()
        old_logprobs = rollout_data.old_log_probabilities.to(self._config.device).detach()
        old_values = rollout_data.old_values.to(self._config.device).detach()
        advantages = rollout_data.advantages.to(self._config.device).detach()
        returns = rollout_data.returns.to(self._config.device).detach()

        # evaluate policy and value function
        values, logprobs, entropies = self._policy.evaluate_action(observations, actions)
        values = values.squeeze(dim=-1)  # flattened list of one value, either per node or per graph

        # normalize advantages
        normalized_advantages = ((advantages - torch.mean(advantages)) / (torch.std(advantages) + 1e-8)).detach()

        # ratios between old and new policy, should be one at the first iteration
        ratios = torch.exp(logprobs - old_logprobs)  # [sum([num_nodes**2 per graph]),]

        train_step_scalars |= train_stat_dict(logprobs, "logprob")
        train_step_scalars |= train_stat_dict(old_logprobs, "old_logprob")

        if self._config.value_scope == "node":
            raise NotImplementedError("node value function scope not yet implemented")
        elif self._config.value_scope == "edge":
            raise NotImplementedError("edge value function scope not yet implemented")
        elif self._config.value_scope == "graph":
            # if the value function acts on full graphs, we need to obtain the mean ratio over the respective graphs
            if "next_hop" in self._config.actor_critic_mode:  # ratio consists of N**2 values per graph
                node_counts = torch.bincount(observations.batch)
                ratio_idx = torch.repeat_interleave(torch.arange(len(node_counts), device=ratios.device), node_counts ** 2)
            else:  # ratio consists of E values per graph
                edge_counts = torch.bincount(observations.batch[observations.edge_index[0]])
                ratio_idx = torch.repeat_interleave(torch.arange(len(edge_counts), device=ratios.device), edge_counts)
            ratios = scatter_mean(ratios, ratio_idx, dim=0)
        else:
            raise ValueError(f"invalid value_scope {self._config.value_scope} in config")

        # calculate losses
        policy_loss = get_policy_loss(normalized_advantages=normalized_advantages,
                                      ratio=ratios,
                                      clip_range=self._config.clip_range)
        value_loss = get_value_loss(returns=returns,
                                    values=values,
                                    old_values=old_values,
                                    clip_range=self._config.value_function_clip_range)
        entropy_loss = get_entropy_loss(entropy=entropies,
                                        log_probabilities=logprobs)
        total_loss = (policy_loss
                      + self._config.value_function_coefficient * value_loss
                      + self._config.entropy_coefficient * entropy_loss)

        # optimization step
        self._policy.optimizer.zero_grad()
        total_loss.backward()
        unclipped_gradients = torch.cat([p.grad.flatten() for p in self._policy.parameters() if p.grad is not None])
        train_step_scalars |= train_stat_dict(unclipped_gradients, "unclipped_gradient")
        torch.nn.utils.clip_grad_norm_(self._policy.parameters(), self._config.max_grad_norm)
        clipped_gradients = torch.cat([p.grad.flatten() for p in self._policy.parameters() if p.grad is not None])
        train_step_scalars |= train_stat_dict(clipped_gradients, "clipped_gradient")
        self._policy.optimizer.step()

        # create train scalars (e.g. for logging)
        with torch.no_grad():
            log_ratio = logprobs - old_logprobs
            lr_exp = (torch.exp(log_ratio) - 1) - log_ratio
            approx_kl_div = detach(torch.mean(lr_exp))
            ratio_outliers = torch.abs(ratios - 1) > self._config.clip_range

        train_step_scalars |= {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "total_loss": total_loss.item(),
            "policy_kl": approx_kl_div.item(),
            "policy_clip_fraction": torch.mean((ratio_outliers).float()).item()
        }
        train_step_scalars |= train_stat_dict(values, "value")
        train_step_scalars |= train_stat_dict(advantages, "advantage")
        train_step_scalars |= train_stat_dict(normalized_advantages, "nrm_advantage")
        train_step_scalars |= train_stat_dict(ratios, "ratios")
        train_step_scalars |= train_stat_dict(entropies, "entropy")

        return total_loss.item(), train_step_scalars

    def training_step(self, it, is_spf_pretraining):
        """
        A training step consists of multiple training epochs on the entire rollout 'buffer' (i.e. the rollout phase's
        experience), where in turn an epoch executes multiple update steps on a minibatch.
        The number of epochs is specified in the config.
        """

        self._policy.train(True)
        old_policy_parameters = torch.cat([param.flatten() for param in self._policy.parameters()])

        train_scalars = {}
        train_step_scalar_dicts = list()
        total_losses = list()

        training_step_range = range(self._config.update_epochs)
        training_step_pbar: Optional[tqdm] = None
        is_spf_pretraining_str = " (SPF pretraining)" if is_spf_pretraining else ""
        if self._logger.level_equals("debug"):
            training_step_range = tqdm(training_step_range,
                                       desc=f"training #{it}{is_spf_pretraining_str}"
                                            f" ({self._config.update_epochs} epochs)")
            training_step_pbar = training_step_range
        else:
            self._logger.log_uncond(f"=== iteration {it}{is_spf_pretraining_str}: "
                                    f"training step ({self._config.update_epochs} epochs) ===")

        # --- training epochs loop ---
        for epoch in training_step_range:
            self._logger.log_info(f"update epoch {epoch}")

            # Do a complete pass on the rollout buffer
            for rollout_data in self._buffer.get(self._config.minibatch_size):
                total_loss, train_step_scalars = self._update_epoch(rollout_data)
                total_losses.append(total_loss)
                train_step_scalar_dicts.append(train_step_scalars)

                if training_step_pbar is not None:
                    training_step_pbar.set_postfix({"total_loss": total_loss})

        new_policy_parameters = torch.cat([param.flatten() for param in self._policy.parameters()])
        param_differences = torch.abs(old_policy_parameters - new_policy_parameters)

        # logging
        if len(train_step_scalar_dicts) > 0:
            train_scalars |= train_step_scalar_dicts[-1]
        train_scalars["mean_network_weight_difference"] = np.mean(detach(param_differences))
        train_scalars["max_network_weight_difference"] = np.max(detach(param_differences))
        train_scalars["value_explained_variance"] = self._buffer.explained_variance
        train_scalars["mean_total_loss"] = np.mean(total_losses)
        train_scalars |= self._policy.get_log()  # actor-critic parameters

        self._policy.train(False)
        return train_scalars

    def _bootstrap_terminal_reward(self, next_observation, reward):
        """
        Bootstraps the reward for the last step of the rollout, if the environment is done due to a terminal state.
        time limit dones, i.e., environment terminations that are due to a time limit rather than
        policy failure are bootstrapped for the advantage estimate. Rather than "stopping" the advantage
        at this step, the reward is bootstraped to include the value function estimate of the current
        observation as an estimate of how the episode *should/could* have continued.
        Args:
            next_observation: The observation after the terminal state
            reward: The reward for the terminal state

        Returns: The bootstrapped reward
        """
        with torch.no_grad():
            terminal_value = self._policy.get_value(next_observation.clone().to(self._policy.device))

        if self._config.value_scope == "graph":
            terminal_value = terminal_value.item()
        else:
            raise NotImplementedError("other value function scopes not yet implemented")
        # elif self._value_function_scope == "node":
            # aggregate over evaluations per node to get one evaluation per graph.
            # Here, we only have one graph, so we can simply take the mean
            # Terminal_value = terminal_value.mean(dim=0)  # TODO is this correct?

        reward += self._buffer.discount_factor * terminal_value
        return reward

    def _policy_sample_step(self, obs, is_last_rollout_step, is_spf_pretraining):
        """
        Performs a single step of the policy, i.e. samples an action, executes it in the environment
        and returns the experience.
        """
        N, E = obs.num_nodes, obs.num_edges

        # sample action and postprocess it. Obs is normalized already
        with torch.no_grad():
            action, logprob, value = self._policy.get_sampled_action(obs.clone().to(self._policy.device))
        action_values, selected_edge_dest_idx = action
        action_values = action_values.cpu()
        selected_edge_dest_idx = selected_edge_dest_idx.cpu()

        # if in SPF pretraining phase, override action for env and vis with SPF actions, and logprobs accordingly.
        if is_spf_pretraining:
            # get SPF action
            monitoring = self._env.monitoring
            action_sp = self._sp_provider.get_sp_actions(monitoring).clone()  # clone is important

            # SPF action does not specify action for pairs of the same node -> randomly select outgoing edges for that
            _, randomly_selected_self_idx = scatter_max(torch.rand(obs.num_edges),
                                                    obs.edge_index[0],
                                                    dim_size=obs.num_nodes)
            action_sp[randomly_selected_self_idx, range(obs.num_nodes)] = 1

            # override action indices and logprob (keep action_values the same for buffer)
            selected_edge_dest_idx = action_sp.flatten().nonzero(as_tuple=False).squeeze()
            logprob = torch.log(action_values[selected_edge_dest_idx])
            action_vis = action_sp.reshape(E, N)

        # if in proper training, we use the policy's sampled action/logprob/edge values for visualization:
        # shape [E, N] for next-hop-centric routing, [E] for link-weight-based routing
        elif "next_hop" in self._config.actor_critic_mode:
            action_vis = action_values.reshape(E, N)
        else:
            action_vis = action_values.flatten()

        # link-weight based policies can use the past link weights as input feature -> store them in env
        if self._config.link_weights_as_input:
            self._env.last_link_weight = action_values

        # prepare action for env and execute it
        action_for_env = obs.clone()
        action_one_hot = torch.zeros(E * N).scatter_(0, selected_edge_dest_idx, 1).reshape(E, N)
        action_for_env.__setattr__("edge_attr", action_one_hot)
        next_obs, reward, _, truncated, infos = self._env.step(action_for_env)

        # immediate postprocessing
        done = float(infos["done"])
        next_obs = self._obs_normalizer.update_and_normalize(next_obs)
        reward = self._reward_normalizer.update_and_normalize(reward)

        # bootstrap reward if episode terminated due to time limit and not policy failure
        if truncated:
            reward = self._bootstrap_terminal_reward(next_obs, reward)

        # store transition in buffer and log
        self._buffer.add(observation=obs,
                         action=(action_values, # always the policy's output, [num_edges * num_nodes,]
                                 selected_edge_dest_idx  # either policy or SPF output, [num_nodes ** 2,]
                                 ),
                         reward=reward,
                         done=done,
                         value=value.flatten(),  # [1] for value_scope == "graph"
                         log_probabilities=logprob  # calculated from either policy or SPF indices [num_nodes ** 2,]
                         )

        # reset
        if done and not is_last_rollout_step:
            obs = self._reset_env_and_normalizers()
        else:
            obs = next_obs

        return infos, done, obs

    def collect_rollouts(self, it: int, is_spf_pretraining: bool):
        """
        Rollout phase: Collects experience by executing the policy in the environment for a specified number
        of episodes.
        """
        self._policy.train(False)
        self._buffer.reset()

        rollout_episode_summaries = []
        rollout_range = range(0, self._config.steps_per_rollout)
        last_rollout_step = rollout_range[-1]
        rollout_pbar: Optional[tqdm] = None
        is_spf_pretraining_str = " (SPF pretraining)" if is_spf_pretraining else ""
        if self._logger.level_equals("debug"):
            rollout_range = tqdm(rollout_range, desc=f"rollout #{it}{is_spf_pretraining_str} "
                                                     f"({self._config.episodes_per_rollout} ep)")
            rollout_pbar = rollout_range
            rollout_pbar.set_postfix({"t": self._env.t})
        else:
            self._logger.log_uncond(f"=== iteration {it}: rollout{is_spf_pretraining_str} "
                                    f"({self._config.episodes_per_rollout} ep) ===")

        # --- rollout loop ---
        obs = None
        step_logs_per_ep = []
        for step in rollout_range:

            if rollout_pbar is not None:
                rollout_pbar.set_postfix({"ep": step // self._config.ep_length, "t": step % self._config.ep_length})

            if not self._env.running:
                obs = self._reset_env_and_normalizers()

            step_log, done, obs = self._policy_sample_step(obs,
                                                           step == last_rollout_step,
                                                           is_spf_pretraining)  # resets env and normalizer if done but rollout not yet finished
            step_logs_per_ep.append(step_log)

            if done:
                ep_summary = step_log.pop("ep_summary")  # remove ep_summary from step_log before converting to dict
                ep_summary |= list_of_dict_to_dict_of_lists(step_logs_per_ep)
                rollout_episode_summaries.append(ep_summary)
                step_logs_per_ep = []

        # finalize rollout
        last_value = self._policy.get_value(obs.clone().to(self._policy.device))
        last_value = last_value.squeeze(dim=-1).flatten()
        self._buffer.compute_returns_and_advantage(last_value=last_value)
        self._env.running = False  # force initial reset for the next env invocation
        return rollout_episode_summaries

    def train_and_evaluate(self):
        """
        Main training loop, executing a specified number of training iterations, each consisting of a rollout phase
        followed by a training step and, optionally, an evaluation phase.
        After training, a final evaluation is performed.
        """
        best_training_loss = np.inf
        best_rollout_global_ep_reward = -np.inf
        best_validation_global_ep_reward = -np.inf
        self.save_components("initial", "saving initial model...")

        for it in range(1, self._config.training_iterations + 1):

            # prep: calculate actual iteration number for policy (needed e.g. for epsilon decay in eps-greedy)
            policy_it = max(it - self._config.spf_pretraining, 0)
            self.policy.set_training_iteration(policy_it)
            is_spf_pretraining = policy_it == 0

            # collect rollouts and execute training step
            rollout_episode_summaries = self.collect_rollouts(it, is_spf_pretraining)
            train_scalars = self.training_step(it, is_spf_pretraining)
            iteration_log = {"rollout": rollout_episode_summaries, "training_step": train_scalars}

            # save model if it improved on the mean total loss over all update epochs
            new_loss = train_scalars["mean_total_loss"]
            if new_loss < best_training_loss:
                best_training_loss = new_loss
                self.save_components("best_training",
                                     "training loss improved -> saving model...")

            # save model if it improved on the rollout episode reward (global_ep_reward)
            rollout_global_ep_reward = np.mean([ep["global_ep_reward"] for ep in rollout_episode_summaries])
            if rollout_global_ep_reward > best_rollout_global_ep_reward:
                best_rollout_global_ep_reward = rollout_global_ep_reward
                self.save_components("best_rollout",
                                     "rollout improved -> saving model...")

            # evaluate after every N training iterations
            if it % self._config.evaluate_every == 0:
                eval_summary = evaluate(self._config, self._env, self._sp_provider, self._logger,
                                        step_func=self._eval_step,
                                        reset_func=self._reset_env_and_normalizers,
                                        get_action_func=self._eval_get_action,
                                        it=it)
                iteration_log["evaluation"] = eval_summary
                validation_global_ep_reward = np.mean([ep_summary["global_ep_reward"] for ep_summary in eval_summary])
                if validation_global_ep_reward > best_validation_global_ep_reward:
                    best_validation_global_ep_reward = validation_global_ep_reward
                    self.save_components("best_validation",
                                         "validation improved -> saving model...")

            # log iteration
            self._logger.log_iteration(it, iteration_log)

        # final save
        self.save_components("last", "saving final model...")

        # After having trained, if final evaluation should be done with a model other than the latest, load it
        if self._config.training_iterations > 0 and self._config.load_mode != "last":
            self.load_components(self._config.base_event_dir)
        final_eval_summary = evaluate(self._config, self._env, self._sp_provider, self._logger,
                                      step_func=self._eval_step,
                                      reset_func=self._reset_env_and_normalizers,
                                      get_action_func=self._eval_get_action,
                                      is_final=True)
        return final_eval_summary
