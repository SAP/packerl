import signal


class SharedMemUnavailableException(Exception):
    """
    A Custom exception thrown when the shared memory object provided via a context manager is None.
    This indicates that the shared memory is unavailable, e.g. due to a segfault on the C++ side.
    """
    def __init__(self, message="Couldn't access shared memory from py side (unavailable) -> terminating..."):
        self.message = message
        super().__init__(self.message)


class TimeoutException(Exception):
    """
    A custom exception raised when a time-critical operation times out.
    """
    def __init__(self, message="Couldn't access shared memory from py side (timeout) -> terminating..."):
        self.message = message
        super().__init__(self.message)


def signal_handler(signum, frame):
    """
    custom signal handler that raises TimeoutExceptions.
    """
    raise TimeoutException()


class timed_fragile(object):
    """
    A light wrapper around context manager objects that allow for deliberate scope breaking
    and automated TimeoutException raises when too much time is spent in the scope.
    """
    class Break(Exception):
        """Break out of the with statement"""

    def __init__(self, value, max_seconds=600):
        """
        Creates the current context
        :param value: The actual context manager
        :param max_seconds: The amount of seconds this context is allowed to exist
        """
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(max_seconds)
        self.value = value

    def __enter__(self):
        """
        Enter the context manager passed at creation time
        :return:
        """
        return self.value.__enter__()

    def __exit__(self, etype, value, traceback):
        """
        Wrapper for exiting the context, taking cake of when the 'Break' exception has been raised.
        """
        signal.alarm(0)
        error = self.value.__exit__(etype, value, traceback)
        if etype == self.Break:
            return True
        return error
