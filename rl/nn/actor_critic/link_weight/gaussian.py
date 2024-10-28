import torch
from torch.distributions.normal import Normal

from rl.nn.actor_critic.link_weight._link_weight_actor_critic import LinkWeightActorCritic
from utils.types import *


class GaussianLinkWeightActorCritic(LinkWeightActorCritic):
    """
    The GaussianLinkWeightActorCritic obtains link weights via sampling from a diagonal gaussian distribution
    whose mean is obtained by the actor module (and the std is a learnable parameter)
    """
    def _get_link_weights_deterministic(self, input: Batch) -> Tensor:
        """
        Obtain link weights deterministically from the actor module
        """
        link_weights = self._actor(input)
        return link_weights

    def _get_link_weights_sampled(self, input: Batch) -> (Tensor, Tensor):
        """
        Obtain link weights by sampling from a gaussian distribution whose mean is obtained by the actor module
        """
        link_weights_mean = self._actor(input)
        link_weights_std = torch.exp(self.exploration_coeff.expand_as(link_weights_mean))
        link_weights_dist = Normal(link_weights_mean, link_weights_std)
        link_weights = link_weights_dist.sample()
        logprob = link_weights_dist.log_prob(link_weights)
        return link_weights, logprob

    def _evaluate_action(self, input: Union[Data, Batch], action) -> (Tensor, Tensor):
        """
        Returns the logprob and entropy of the given action after obtaining an action distribution from the input
        """
        link_weights_mean = self._actor(input)
        link_weights_std = torch.exp(self.exploration_coeff.expand_as(link_weights_mean))
        link_weights_dist = Normal(link_weights_mean, link_weights_std)
        action_link_weights, _ = action
        logprob = link_weights_dist.log_prob(action_link_weights)
        entropy = link_weights_dist.entropy()
        return logprob, entropy
