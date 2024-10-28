import torch
import torch_scatter
from torch.distributions.categorical import Categorical

from rl.nn.actor_critic.next_hop._next_hop_actor_critic import NextHopActorCritic
from utils.tensor import scatter_logsumexp
from utils.types import Tensor


class SoftmaxNextHopActorCritic(NextHopActorCritic):
    """
    The SoftmaxNextHopActorCritic interprets the actor output as logits for a categorical next-hop selection
    distribution per routing node. This means that we're doing Boltzmann exploration here
    """

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

    def _select_edges(self, edge_dest_values, edge_dest_idx) -> (Tensor, Tensor):
        """
        :param edge_dest_values: Interpreted as selection probabilities for each edge per destination node.
        """
        # We assume src_dst_pair_count corresponds to sum([N**2 for N in num_nodes_per_graph]),
        # which is the overall count of valid source-destination node pairs in the batch.
        src_dst_pair_count = int(edge_dest_idx.max().item()) + 1
        sampled_probs = torch.zeros(src_dst_pair_count, device=self.device)
        sampled_positions = torch.zeros(src_dst_pair_count, dtype=torch.long, device=self.device)

        # We don't have anything like scatter_sample() at the moment, so sampling is looped.
        for i in range(src_dst_pair_count):
            mask: Tensor = edge_dest_idx == i
            cur_probs = edge_dest_values[mask]
            if cur_probs.nelement() == 0:
                raise ValueError("there is no outgoing edge for some src-dst pair! This should not happen...")
            s = Categorical(probs=cur_probs).sample()
            sampled_probs[i] = cur_probs[s]
            cur_position_candidates = torch.nonzero(mask).view(-1)
            sampled_positions[i] = cur_position_candidates[s]

        return sampled_probs, sampled_positions

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
        logprob = torch.log(edge_probs)[selected_edge_dest_idx]  # 1-dim tensor, numel=sum([g.N**2 for graph g in batch])

        # calculate entropy from categorical distribution over edges
        action_edge_p_logp = action_edge_dest_probs * torch.log(action_edge_dest_probs)
        entropy = -torch_scatter.scatter_add(action_edge_p_logp, edge_dest_idx)  # 1-dim tensor, numel=sum([g.E*g.N for graph g in batch])

        return logprob, entropy
