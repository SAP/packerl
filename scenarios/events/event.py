class Event:
    """
    Base class for events.
    """

    def __init__(self, t):
        self.t = t

    def __str__(self):
        return f"Event(t={self.t})"

    def apply(self, scenario):
        raise NotImplementedError
