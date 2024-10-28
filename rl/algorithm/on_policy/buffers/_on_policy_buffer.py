import abc
from typing import NamedTuple, Optional, Union, Generator, Tuple

import numpy as np
import torch
from torch_geometric.data import Data, Batch

from utils.types import InputBatch, Tensor


def explained_variance_function(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    :param y_pred: the prediction
    :param y_true: the expected value
    :return: explained variance of ypred and y
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


class RolloutBufferSamples(NamedTuple):
    observations: InputBatch
    actions: Tuple[torch.Tensor, torch.Tensor]  # (edge_dest_values, selected_edge_dest_idx)
    old_log_probabilities: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class AbstractMultiAgentOnPolicyBuffer(abc.ABC):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    For multi-agent RL, these transitions may either be collected for each node/agent per timestep, or for each
    full graph/set of agents. This class gives a generic interface for both and is inherited by NodeOnPolicyBuffer
    and GraphOnPolicyBuffer for this distinction.

    The buffer only persists for a single algorithm iteration, i.e., it is used to store recent (on-policy) transitions.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state (again node or graph-wise)
    and the log probability of the taken actions (always node-wise).


        The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not actions selection.
    """

    def __init__(self, buffer_size: int, gae_lambda: float, discount_factor: float,
                 device: Optional[torch.device] = None, **kwargs):
        """

        Args:
            buffer_size: Maximum number of element in the buffer
            gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
                Equivalent to classic advantage when set to 1.
            discount_factor: Discount factor/gamma of the environment
        """
        self.buffer_size = buffer_size
        self.gae_lambda = gae_lambda
        self.discount_factor = discount_factor

        # initialize all the values to be stored in the buffer
        self.current_position = None

        self.observations = None
        self.actions = None
        self.log_probabilities = None
        self.rewards = None
        self.dones = None

        self.values = None
        self.advantages = None
        self.returns = None

        # reset all values to their default value to get ready to collect data
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.current_position = 0
        self.observations = []
        self.actions = []
        self.log_probabilities = []
        self.rewards = np.empty(self.buffer_size)
        self.dones = np.empty(self.buffer_size)
        self._reset_values()

    def _reset_values(self):
        raise NotImplementedError("Abstract MultiAgentOnPolicyBuffer does not implement '_reset_values()'")

    def add(self, *, observation: Data, action: (torch.Tensor, torch.Tensor), reward: Union[float, np.array],
            done: float, value: torch.Tensor, log_probabilities: torch.Tensor, **kwargs) -> None:
        """

        Args:
            observation: Graph observation
            action: Action
            reward: Reward returned by the environment after taking the action
            done: Whether this step was the last step of the episode
            value: Value function evaluation for the current state-action pair
            log_probabilities: Log probability of each action for the current policy

        Returns:

        """
        self.observations.append(observation.to(self.device))
        self.actions.append((action[0].to(self.device), action[1].to(self.device)))
        self.log_probabilities.append(log_probabilities.to(self.device))

        # directly adding scalar values to pre-allocated arrays to save on compute
        self._add_values(value)
        self._add_reward(reward)
        self.dones[self.current_position] = done

        # keeping track of the number of entries
        self.current_position += 1
        if self.current_position == self.buffer_size:  # convert scalar entries to Tensors once full
            self._convert_dones()
            self._convert_rewards()

    def _add_values(self, current_value):
        raise NotImplementedError("Abstract MultiAgentOnPolicyBuffer does not implement '_add_values()'")

    def _add_reward(self, reward: float):
        """
        Add a reward to the buffer.
        Args:
            reward: Reward to add to the buffer
        """
        self.rewards[self.current_position] = reward

    def _convert_dones(self):
        """
        Cast scalar entries (one scalar per timestep/graph) into a graph for easier handling.
        Depending on whether this is a graph- or node-based policy buffer, the values are either per graph or per node
        and are handled accordingly
        Returns:

        """
        self.dones = torch.tensor(self.dones, device=self.device)

    def _convert_rewards(self):
        raise NotImplementedError

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        """
        Get a generator yielding samples from the buffer.
        """
        assert self.full, "Need to provide full rollout buffer before accessing it"
        indices = np.random.permutation(self.buffer_size)  # get random permutation of data

        start_idx = 0
        while start_idx < self.buffer_size:  # generate a piece of data for each call until all indices have been used
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def _get_obs_and_actions(self, batch_indices: np.ndarray) -> (Batch, Tensor, Tensor):
        """
        Get the observations and actions for the given batch indices.
        This method is necessary because we need to adjust action indices when creating batches.
        """
        observations = []
        edge_dest_values = []
        selected_edge_dest_idx = []
        idx_offset = 0
        for index in batch_indices:
            cur_obs = self.observations[index]
            observations.append(cur_obs)
            cur_edge_dest_values, cur_selected_edge_dest_idx = self.actions[index]
            edge_dest_values.append(cur_edge_dest_values)
            selected_edge_dest_idx.append(cur_selected_edge_dest_idx + idx_offset)
            idx_offset += cur_obs.num_nodes * cur_obs.num_edges

        observations = Batch.from_data_list(observations)
        actions = (torch.cat(edge_dest_values, dim=0), torch.cat(selected_edge_dest_idx, dim=0))
        return observations, actions

    def _get_samples(self, batch_indices: np.ndarray) -> RolloutBufferSamples:
        """
        Get the samples for the given batch indices.
        """
        observations, actions = self._get_obs_and_actions(batch_indices)

        log_probabilities = [self.log_probabilities[index] for index in batch_indices]
        log_probabilities = torch.cat(log_probabilities, dim=0)

        values, advantages, returns = self._get_values_advantages_returns(batch_indices)

        # these tensors are implemented by child methods, so we move to the current device to be safe
        values: torch.Tensor = values.to(self.device)
        advantages: torch.Tensor = advantages.to(self.device)
        returns = returns.to(self.device)
        data = (
            observations,
            actions,
            log_probabilities,  # flattened log probabilities per action, one per *node* over batch_size many *graphs*
            values,
            advantages,
            returns
        )
        return RolloutBufferSamples(*data)

    def _get_values_advantages_returns(self, batch_indices: np.array) -> Tuple[torch.Tensor,
    torch.Tensor,
    torch.Tensor]:
        """
        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438) to compute the advantage as
        GAE: h_t^V = r_t + y * V(s_t+1) - V(s_t)
        with
        V(s_t+1) = {0 if s_t is terminal (where V(s) may be bootstrapped in the algorithm sampling for terminal dones)
                   {V(s_t+1) if s_t not terminal
        where T is the last step of the rollout.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        Args:
        Returns:

        """
        raise NotImplementedError("Abstract MultiAgentOnPolicyBuffer does not "
                                  "implement '_get_values_advantages_returns()'")

    def compute_returns_and_advantage(self, last_value: torch.Tensor) -> None:
        raise NotImplementedError("Abstract MultiAgentOnPolicyBuffer does not "
                                  "implement 'compute_returns_and_advantage()'")

    @property
    def explained_variance(self):
        raise NotImplementedError("Abstract MultiAgentOnPolicyBuffer does not implement 'explained_variance'")

    @property
    def full(self):
        return self.current_position == self.buffer_size
