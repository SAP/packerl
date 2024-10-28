from rl.nn.actor._actor import Actor


class LinkWeightActor(Actor):
    """
    Base class for all link weight actors. Link weight actors are used to predict link weights in a graph
    which are then used to compute shortest routing paths.
    """
    pass
