import networkx as nx
import numpy as np

from reward.reward_functions.reward_utils import build_routing_graphs


def dummy_reward(**reward_input):
    """
    Dummy reward function that always returns 1.0.
    """
    return 1.0


def max_LU(**reward_input):
    """
    Returns the maximum link utilization in the network.
    """
    state = reward_input.get("global_monitoring", None)
    if state is None:
        raise ValueError("invoking reward without global_monitoring")
    return np.max([edge[2]["LU"] for edge in state.edges(data=True)])


def drop_ratio(**reward_input):
    """
    Returns the drop ratio of the network, which is the ratio of dropped bytes to dropped + received bytes.
    """
    state = reward_input.get("global_monitoring", None)
    if state is None:
        raise ValueError("invoking reward without global_monitoring")
    dropped = state.graph['droppedBytes']
    received = state.graph['receivedBytes']
    dropped_and_received = dropped + received
    if dropped_and_received == 0:
        return 0.0
    return dropped / dropped_and_received


def avg_delay(**reward_input):
    """
    Returns the average packet delay in the network.
    """
    state = reward_input.get("global_monitoring", None)
    if state is None:
        raise ValueError("invoking reward without global_monitoring")
    return state.graph["avgPacketDelay"]


def max_delay(**reward_input):
    """
    Returns the maximum packet delay in the network.
    """
    state = reward_input.get("global_monitoring", None)
    if state is None:
        raise ValueError("invoking reward without global_monitoring")
    return state.graph["maxPacketDelay"]


def weighted_delay(**reward_input):
    """
    Returns the weighted delay of the network, where dropped packets have a higher impact on the delay.
    """
    state = reward_input.get("global_monitoring", None)
    if state is None:
        raise ValueError("invoking reward without global_monitoring")
    dropped_b = state.graph['droppedBytes']
    received_b = state.graph['receivedBytes']
    dropped_and_received_b = dropped_b + received_b
    if dropped_and_received_b == 0:
        return 0.0
    max_delay = state.graph["maxPacketDelay"]
    weighted_delay = (2 * max_delay * dropped_b + received_b * state.graph['avgPacketDelay']) / dropped_and_received_b
    return weighted_delay


def avg_packet_jitter(**reward_input):
    """
    Returns the average packet jitter in the network.
    """
    state = reward_input.get("global_monitoring", None)
    if state is None:
        raise ValueError("invoking reward without global_monitoring")
    return state.graph["avgPacketJitter"]


def avg_routing_loops_per_node(**reward_input):
    """
    Returns the average amount of routing loops per node in the provided routing.
    """
    actions = reward_input.get("actions", None)
    if actions is None:
        raise ValueError("invoking routing_loops reward function with actions=None")
    routing_graphs_per_dest = build_routing_graphs(actions)
    total_amount_of_cycles = sum([len(list(nx.simple_cycles(rg))) for rg in routing_graphs_per_dest])
    return total_amount_of_cycles / actions.num_nodes


def sent_MB(**reward_input):
    """
    Returns the amount of sent MB in the network (~throughput).
    """
    state = reward_input.get("global_monitoring", None)
    if state is None:
        raise ValueError("invoking reward without global_monitoring")
    return state.graph['sentBytes'] / 1e6


def received_MB(**reward_input):
    """
    Returns the amount of received MB in the network (~goodput).
    """
    state = reward_input.get("global_monitoring", None)
    if state is None:
        raise ValueError("invoking reward without global_monitoring")
    return state.graph['receivedBytes'] / 1e6


def retransmitted_MB(**reward_input):
    """
    Returns the amount of retransmitted MB in the network.
    """
    state = reward_input.get("global_monitoring", None)
    if state is None:
        raise ValueError("invoking reward without global_monitoring")
    return state.graph['retransmittedBytes'] / 1e6
