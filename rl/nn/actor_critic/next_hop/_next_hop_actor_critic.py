import torch_scatter
from torch.distributions.distribution import Distribution

from rl.nn.actor_critic._actor_critic import ActorCritic
from rl.nn.actor.next_hop import build_next_hop_actor
from rl.nn.critic.next_hop import build_next_hop_critic
from utils.types import *
from utils.tensor import get_edge_dest_idx


class NextHopActorCritic(ActorCritic):
    """
    Abstract next-hop actor-critic class: Its actions describe routing decisions per node,
    i.e. each node selects a next-hop edge per possible destination node.
    """
    def _build_actor(self, nn_config, feature_counts, actor_mode, device):
        """
        Get next-hop actor
        """
        return build_next_hop_actor(nn_config, feature_counts, actor_mode, device)

    def _build_critic(self, nn_config, feature_counts, critic_mode, value_scope, device):
        """
        Get next-hop critic
        """
        return build_next_hop_critic(nn_config, feature_counts, critic_mode, value_scope, device)

    def _select_edges(self, edge_dest_values, edge_dest_idx) -> (Tensor, Tensor):
        """
        Given the actor output per edge per destination and an index tensor, this method actually selects the
        edge actions. The derived classes need to implement this method, e.g. by sampling or selecting the max.
        """
        raise NotImplementedError

    def _sample_action(self, edge_dest_actor_output, edge_dest_idx) -> ((Tensor, Tensor), Tensor):
        """
        Given the actor output per edge per destination and an index tensor, this method probabilistically
        selects the edge actions
        """
        raise NotImplementedError

    def _evaluate_action(self, edge_dest_actor_output, edge_dest_idx, action) -> (Tensor, Tensor):
        """
        Given the actor output per edge per destination, an index tensor and a previously computed action,
        this method evaluates the given action wrt. its probability and entropy.
        """
        raise NotImplementedError

    def _get_actor_output_and_value(self, input: Union[Data, Batch]):
        """
        Necessary preprocessing for the task: preprocess input, obtain a flat edge-destination index tensor
        for later computations, and get actor and critic output for further action selection/evaluation.
        """
        input_preprocessed = self._preprocess_input(input.clone())

        # get unrolled edge-dest indices to properly index the edge_dest_actor_output tensor
        #  Caution: this assumes that every node has at least one outgoing edge!
        edge_dest_idx = get_edge_dest_idx(input_preprocessed)

        # get edge_dest_actor_output from the actor, which are interpreted depending on the sampling mode
        edge_dest_actor_output = self._actor(input_preprocessed.clone())  # 1-dim tensor, numel=sum([g.E*g.N for graph g in batch])

        # get value
        value = self._critic(input_preprocessed.clone())

        return edge_dest_actor_output, edge_dest_idx, value

    def forward(self, input: Union[Data, Batch]) -> (Tensor, Distribution, Tensor):
        return self.get_sampled_action(input)

    def get_deterministic_action(self, input: Union[Data, Batch]) -> ((Tensor, Tensor), Tensor):
        """
        Returns the action with the highest probability for each edge destination, obtained from
        the actor output
        """
        edge_dest_actor_output, edge_dest_idx, value = self._get_actor_output_and_value(input)
        _, selected_edge_dest_idx = torch_scatter.scatter_max(edge_dest_actor_output, index=edge_dest_idx)  # edge_dest_values, selected_edge_dest_idx
        return (edge_dest_actor_output, selected_edge_dest_idx), value

    def get_sampled_action(self, input: Union[Data, Batch]) -> ((Tensor, Tensor), Tensor, Tensor):
        """
        Returns a sampled action for each edge destination, obtained from the actor output
        """
        edge_dest_actor_output, edge_dest_idx, value = self._get_actor_output_and_value(input)
        action, logprob = self._sample_action(edge_dest_actor_output, edge_dest_idx)  # (edge_dest_values, selected_edge_dest_idx), logprob
        return action, logprob, value

    def evaluate_action(self, input: Union[Data, Batch], action: (Tensor, Tensor)) -> (Tensor, Tensor, Tensor):
        """
        Returns the value, logprob and entropy of the given action after obtaining an action distribution from
        the input
        """
        edge_dest_actor_output, edge_dest_idx, value = self._get_actor_output_and_value(input)
        logprob, entropy = self._evaluate_action(edge_dest_actor_output, edge_dest_idx, action)
        return value, logprob, entropy
