import torch
import torch_scatter
from torch.distributions.categorical import Categorical

from rl.nn.actor_critic.next_hop._next_hop_actor_critic import NextHopActorCritic
from utils.tensor import scatter_logsumexp
from utils.types import Tensor


class EpsGreedyNextHopActorCritic(NextHopActorCritic):
    """
    The EpsGreedyNextHopActorCritic interprets the actor output as logits for a next-hop selection distribution.
    However, as opposed to the SoftmaxNextHopActorCritic, the resulting next-hop selection probability distribution
    is only used to calculate the logprobs. Instead, the greedy action simply selects the max-prob. next-hop,
    while in the exploration case we select a random edge.
    """
    def set_training_iteration(self, it):
        self.it = it
        self.epsilon = self.min_epsilon + ((1 - self.min_epsilon) * self.epsilon_decay ** self.it)

    def get_log(self):
        return {"epsilon": self.epsilon} | super().get_log()

    def _actor_output_to_probs(self, edge_dest_actor_output, edge_dest_idx) -> Tensor:
        """
        Converts the actor's output, i.e. edge selection logits, to a probability distribution over edges.
        Note that the resulting probabilities are similar but not identical
        to the softmax of the exponents of the logits (the usual way to obtain probs from logits), but we have to
        calculate them using the Log-Sum-Exp trick since otherwise we run into numerical overflows.
        """
        edge_dest_logits = - self.exploration_coeff * edge_dest_actor_output
        edge_dest_logits_logsumexp = scatter_logsumexp(edge_dest_logits, edge_dest_idx)
        return torch.exp(edge_dest_logits - edge_dest_logits_logsumexp[edge_dest_idx])

    def _get_probs_and_positions(self, edge_dest_values, edge_dest_idx,
                                 get_sample_from_probs, get_factored_prob):
        # We assume src_dst_pair_count corresponds to sum([N**2 for N in num_nodes_per_graph]),
        # which is the overall count of valid source-destination node pairs in the batch.
        src_dst_pair_count = int(edge_dest_idx.max().item()) + 1
        sampled_probs = torch.zeros(src_dst_pair_count)
        sampled_positions = torch.zeros(src_dst_pair_count, dtype=torch.long)

        # We don't have anything like scatter_sample() at the moment, so sampling is looped.
        for i in range(src_dst_pair_count):
            mask: Tensor = edge_dest_idx == i
            cur_probs = edge_dest_values[mask]
            cur_options = cur_probs.nelement()
            if cur_options == 0:
                raise ValueError("there is no outgoing edge for some src-dst pair! This should not happen...")
            s = get_sample_from_probs(cur_probs)
            sampled_probs[i] = get_factored_prob(cur_probs[s], cur_options)
            cur_position_candidates = torch.nonzero(mask).view(-1)
            sampled_positions[i] = cur_position_candidates[s]

        return sampled_probs, sampled_positions

    def _select_edges(self, edge_dest_values, edge_dest_idx) -> (Tensor, Tensor):
        """
        edge_dest_scores are interpreted as unnormalized logits for a distribution, from which we sample the edges.
        The actor's output is scaled by the negative exploration coefficient C before being treated as a logit.
        However, with a probability of epsilon, we simply choose random edges.
        All in all, the input probs for the logprob operation are adjusted to account for eps.-greedy exploration.
        """
        # exploration: Sample random edges and set uniform probs
        if torch.rand(1) < self.epsilon:
            get_sample_from_probs = lambda cur_probs: Categorical(probs=torch.ones_like(cur_probs)).sample()
            get_factored_prob = lambda raw_prob, cur_options: self.epsilon / cur_options

        # greedy action: sample from categorical distribution spanned by probs and use according probs
        else:
            get_sample_from_probs = lambda cur_probs: Categorical(probs=cur_probs).sample()
            get_factored_prob = lambda raw_prob, cur_options: (1-self.epsilon) * raw_prob + self.epsilon / cur_options

        return self._get_probs_and_positions(edge_dest_values, edge_dest_idx,
                                             get_sample_from_probs, get_factored_prob)

    def _sample_action(self, edge_dest_actor_output, edge_dest_idx) -> ((Tensor, Tensor), Tensor):
        """
        edge_dest_scores are interpreted as unnormalized logits for a distribution, from which we sample the edges.
        The actor's output is scaled by the negative exploration coefficient C before being treated as a logit.
        """
        edge_probs = self._actor_output_to_probs(edge_dest_actor_output, edge_dest_idx)
        selected_probs, selected_edge_dest_idx = self._select_edges(edge_probs, edge_dest_idx)  # 1-dim tensors, numel=sum([g.N**2 for graph g in batch])
        logprob = torch.log(selected_probs)  # 1-dim tensor, numel=sum([g.N**2 for graph g in batch])

        return (edge_probs, selected_edge_dest_idx), logprob

    def _evaluate_action(self, edge_dest_actor_output, edge_dest_idx, action) -> (Tensor, Tensor):
        edge_probs = self._actor_output_to_probs(edge_dest_actor_output, edge_dest_idx)
        action_edge_dest_probs, selected_edge_dest_idx = action
        logprob = torch.log(edge_probs)[
            selected_edge_dest_idx]  # 1-dim tensor, numel=sum([g.N**2 for graph g in batch])

        # calculate entropy from categorical distribution over edges
        action_edge_p_logp = action_edge_dest_probs * torch.log(action_edge_dest_probs)
        entropy = -torch_scatter.scatter_add(action_edge_p_logp,
                                             edge_dest_idx)  # 1-dim tensor, numel=sum([g.E*g.N for graph g in batch])

        return logprob, entropy
