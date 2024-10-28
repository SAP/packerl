from ..event import Event


class LinkFailure(Event):
    """
    Event that represents a link failure. This includes the time of the failure and the two incident nodes.
    """

    def __init__(self, t, fst, snd):
        super().__init__(t)
        self.fst = fst
        self.snd = snd

    def __str__(self):
        """
        Return a string representation of the event.
        """
        return f"LinkFailure(t={self.t}, ({self.fst},{self.snd}))"

    def apply(self, scenario):
        """
        Apply the event to the scenario.
        """
        scenario.network.remove_edge(self.fst, self.snd)
