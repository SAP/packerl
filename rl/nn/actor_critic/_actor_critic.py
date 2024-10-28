import torch
import torch.nn as nn
import torch.optim as optim

from utils.types import *


class ActorCritic(nn.Module):
    """
    Abstract actor-critic class
    """
    def __init__(self,
                 ac_config: dict,
                 nn_config: dict,
                 feature_counts: dict,
                 value_scope: str,
                 learning_rate: float,
                 device: str
                 ):
        super().__init__()

        for k, v in ac_config.items():
            setattr(self, k, v)

        if self.concat_rev_edges:
            feature_counts['edge'] *= 2
        self.device = device

        self._actor = self._build_actor(nn_config=nn_config,
                                        feature_counts=feature_counts,
                                        actor_mode=self.actor_mode,
                                        device=self.device)

        self._critic = self._build_critic(nn_config=nn_config,
                                          feature_counts=feature_counts,
                                          critic_mode=self.critic_mode,
                                          value_scope=value_scope,
                                          device=self.device)

        self.exploration_coeff = nn.Parameter(torch.tensor(self.initial_exploration_coeff, device=self.device))
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.it = 0

    def _build_actor(self, nn_config, feature_counts, actor_mode, device):
        raise NotImplementedError

    def _build_critic(self, nn_config, feature_counts, critic_mode, value_scope, device):
        raise NotImplementedError

    # TODO we should save this in addition to the model state_dict if we want continue training at a later point
    def set_training_iteration(self, it):
        self.it = it

    def get_log(self):
        """
        Returns logging information
        """
        return {"exploration_coeff": self.exploration_coeff.item()}

    def _preprocess_input(self, input: Union[Data, Batch]) -> Batch:
        """
        Batchify input if necessary and add reverse edges if desired.
        """
        if not isinstance(input, Batch):
            input = Batch.from_data_list([input])
        if self.concat_rev_edges:
            edge_attr = input.edge_attr
            s, d = input.edge_index.tolist()
            edge_idx_tuples = zip(s, d)
            edge_rev_idx_tuples = list(zip(d, s))
            rev_edge_attr = edge_attr[[edge_rev_idx_tuples.index(e) for e in edge_idx_tuples]]
            input.__setattr__("edge_attr", torch.cat([edge_attr, rev_edge_attr], dim=-1))

        return input

    def forward(self, input: Union[Data, Batch]):
        return self.get_sampled_action(input)

    def get_deterministic_action(self, input: Union[Data, Batch]) -> ((Tensor, Tensor), Tensor):
        """
        Returns the action with the highest probability.
        """
        raise NotImplementedError

    def get_sampled_action(self, input: Union[Data, Batch]) -> ((Tensor, Tensor), Tensor, Tensor):
        """
        Returns a sampled action.
        """
        raise NotImplementedError

    def evaluate_action(self, input: Union[Data, Batch], action) -> (Tensor, Tensor, Tensor):
        """
        Returns the value, logprob and entropy of the given action,
        after obtaining an action distribution from the input
        """
        raise NotImplementedError

    def get_value(self, input: Union[Data, Batch]) -> Tensor:
        """
        Returns the value of the given input
        """
        input_preprocessed = self._preprocess_input(input.clone())
        value = self._critic(input_preprocessed.clone())
        return value
