from ._actor_critic import ActorCritic
from .next_hop import (EpsGreedyNextHopActorCritic,
                       ScoreNextHopActorCritic,
                       SoftmaxNextHopActorCritic)
from .link_weight import GaussianLinkWeightActorCritic


actor_critics = {
    "next_hop_eps_greedy": EpsGreedyNextHopActorCritic,
    "next_hop_score": ScoreNextHopActorCritic,
    "next_hop_softmax": SoftmaxNextHopActorCritic,
    "link_weight_gaussian": GaussianLinkWeightActorCritic,
}

def get_actor_critic(ac_config, nn_config, feature_counts, value_scope, learning_rate, device) -> ActorCritic:
    """
    Returns an actor critic model that contains an actor_mode-dependent actor and a critic_mode-dependent
     critic. Depending on the actor_critic_mode, the chosen ActorCritic exhibits different exploration and
     actor output processing behavior.
    """
    actor_critic_mode = ac_config["actor_critic_mode"]
    actor_critic_cls = actor_critics.get(actor_critic_mode, None)
    if actor_critic_cls is None:
        raise ValueError(f"get_actor_critic(): {actor_critic_mode=} is unknown")
    else:
        return actor_critic_cls(ac_config, nn_config, feature_counts, value_scope, learning_rate, device)
