import torch
import torch_scatter
from torch.distributions.normal import Normal

from rl.nn.actor_critic.next_hop._next_hop_actor_critic import NextHopActorCritic
from utils.types import Tensor


class ScoreNextHopActorCritic(NextHopActorCritic):
    """
    The ScoreNextHopActorCritic interprets the actor output as score values for next-hop preferences.
    In the deterministic case the edge with maximal actor-provided score is chosen as a node's next-hop edge,
    in the sampling case the value is treated as the mean for a gaussian distribution,
    from which the actual value is sampled.
    """

    def _select_edges(self, edge_dest_values, edge_dest_idx) -> (Tensor, Tensor):
        """
        Given the actor output per edge per destination and an index tensor, this method selects the edge actions
        deterministically by taking the edge with the highest score.
        """
        max_probs, max_positions = torch_scatter.scatter_max(edge_dest_values, index=edge_dest_idx)
        return max_probs, max_positions

    def _sample_action(self, edge_dest_actor_output, edge_dest_idx) -> ((Tensor, Tensor), Tensor):
        """
        edge_dest_scores are interpreted as means for a normal distribution with scale=actor_logstd,
        from which we will sample scores and take the maximum score over all concerned edges.
        The exploration coefficient C is the logstd of the normal distribution (mean is the actor's output)
        """
        edge_dest_score_mean = edge_dest_actor_output
        edge_dest_score_std = torch.exp(self.exploration_coeff.expand_as(edge_dest_score_mean))  # 1-dim tensor, numel=sum([g.E*g.N for graph g in batch])
        edge_dest_score_dist = Normal(edge_dest_score_mean, edge_dest_score_std)
        edge_dest_score = edge_dest_score_dist.sample()  # 1-dim tensor, numel=sum([g.E*g.N for graph g in batch])
        _, selected_edge_dest_idx = self._select_edges(edge_dest_score, edge_dest_idx)
        logprob = edge_dest_score_dist.log_prob(edge_dest_score)[selected_edge_dest_idx]

        return (edge_dest_score, selected_edge_dest_idx), logprob

    def _evaluate_action(self, edge_dest_actor_output, edge_dest_idx, action) -> (Tensor, Tensor):
        edge_dest_score_mean = edge_dest_actor_output
        action_edge_dest_score, selected_edge_dest_idx = action
        edge_dest_score_std = torch.exp(self.exploration_coeff.expand_as(edge_dest_score_mean))  # 1-dim tensor, numel=sum([g.E*g.N for graph g in batch])
        edge_dest_score_dist = Normal(edge_dest_score_mean, edge_dest_score_std)
        logprob = edge_dest_score_dist.log_prob(action_edge_dest_score)[selected_edge_dest_idx]

        # ENTROPY: We're still dealing with a categorical distribution over the possible edges per src_dest node pair,
        # but here the categories are indirectly parametrized by the independent gaussians over the scores,
        # from which we sample and take the maximum. Calculating the entropy for this exploration mode
        # involves getting the probability for each edge of the corresponding sampled score being higher than for all
        # competitor edges, which requires multidimensional integration over the gaussians -> practically infeasible.
        # Therefore, we approximate the probabilities via Monte Carlo sampling from the distribution and calculating
        # pseudo-probabilities per edge.

        # sample edges from the distribution
        n_samples = 20
        edge_dest_score_values = torch.stack([edge_dest_score_dist.sample() for _ in range(n_samples)],
                                             dim=-1)  # (sum([g.E*g.N for graph g in batch]), n_samples)
        _, edge_dest_sampled_edges = torch_scatter.scatter_max(edge_dest_score_values, edge_dest_idx,
                                                               dim=0)  # (sum([g.N**2 for graph g in batch]), n_samples)

        # calculate pseudo-probabilities per edge via selection counts
        counts = edge_dest_sampled_edges.flatten().bincount(
            minlength=edge_dest_score_values.shape[0])  # 1-dim tensor, numel=sum([g.E*g.N for graph g in batch])
        action_edge_dest_probs = counts.double() / n_samples

        # calculate entropy from categorical distribution over edges, with calculated pseudo-probabilities
        action_edge_p_log_p = action_edge_dest_probs * torch.log(action_edge_dest_probs + 1e-10)  # add epsilon to avoid log(0)
        entropy = -torch_scatter.scatter_add(action_edge_p_log_p,
                                             edge_dest_idx)  # 1-dim tensor, numel=sum([g.E*g.N for graph g in batch])
        return logprob, entropy
