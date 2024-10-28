from .graph_on_policy_buffer import GraphOnPolicyBuffer

def get_buffer(value_scope: str, buffer_size: int, gae_lambda: float, discount_factor: float, device= None, **kwargs):
    """
    Returns the appropriate buffer for the given value scope.
    """
    if value_scope == "edge":
        raise NotImplementedError("EdgeOnPolicyBuffer not implemented yet")
    elif value_scope == "node":
        raise NotImplementedError("NodeOnPolicyBuffer not implemented yet")
    elif value_scope == "graph":
        return GraphOnPolicyBuffer(buffer_size=buffer_size,
                                   gae_lambda=gae_lambda,
                                   discount_factor=discount_factor,
                                   device=device)
    else:
        raise ValueError(f"invalid value scope: {value_scope}")