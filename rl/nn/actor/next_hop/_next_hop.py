from rl.nn.actor._actor import Actor


class NextHopActor(Actor):
    """
    Base class for all next-hop actors. Next-hop actors are used to predict the next hop for packets
    at each routing node and for all possible destinations.
    """
    pass
