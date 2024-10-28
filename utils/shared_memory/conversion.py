"""
This module provides functions to convert between python-side components and ctypes-compatible representations
as obtained/used by the C++ side of PackeRL.
"""
from typing import List, Union

import networkx as nx
import numpy as np

from scenarios.events import Event as PSEvent, LinkFailure as PSLinkFailure
from scenarios.events.traffic.traffic_demand import TrafficDemand as PSTrafficDemand

from utils.shared_memory.structs import (TopologyEdge,
                                         TrafficDemand as CTypeTrafficDemand,
                                         LinkFailure as CTypeLinkFailure,
                                         TrafficByteMatrixEntry,
                                         TrafficPacketMatrixEntry,
                                         MonitoringGlobalSnapshot,
                                         MonitoringDirEdgeSnapshot,
                                         RoutingActionComponent)


def topology_edge_to_ctype(fst, snd, attributes) -> TopologyEdge:
    """
    Converts a given topology edge into a ctype-compatible representation.
    """
    c_edge = TopologyEdge()
    c_edge.fst = fst
    c_edge.snd = snd
    c_edge.datarate = int(attributes["datarate"])
    c_edge.delay = int(attributes["delay"])
    return c_edge


def link_failure_to_ctype(link_failure: PSLinkFailure) -> CTypeLinkFailure:
    """
    Converts a given link failure event (from scenarios) into a ctype-compatible representation.
    """
    c_link_failure = CTypeLinkFailure()
    c_link_failure.t = link_failure.t
    c_link_failure.fst = link_failure.fst
    c_link_failure.snd = link_failure.snd
    return c_link_failure


def traffic_demand_to_ctype(demand: PSTrafficDemand) -> CTypeTrafficDemand:
    """
    Converts a given traffic demand event (from scenarios) into a ctype-compatible representation.
    """
    c_demand = CTypeTrafficDemand()
    c_demand.t = demand.t
    c_demand.src = demand.src
    c_demand.dst = demand.dst
    c_demand.amount = demand.amount
    c_demand.datarate = demand.datarate
    c_demand.isTCP = demand.is_tcp
    return c_demand


def event_to_ctype(event: PSEvent):
    """
    Converts a given event (from scenarios) into a ctype-compatible representation.
    """
    if isinstance(event, PSTrafficDemand):
        return 0, traffic_demand_to_ctype(event)
    elif isinstance(event, PSLinkFailure):
        return 1, link_failure_to_ctype(event)
    else:
        raise ValueError(f"Event type {type(event)} not supported!")


def action_component_to_ctype(edge_src, edge_dst, demand_dst, value) -> RoutingActionComponent:
    """
    Converts a given routing action component (i.e., a combination of routing edge, packet destination and value)
    into a ctype-compatible representation
    """
    c_action_component = RoutingActionComponent()
    c_action_component.edgeSrc = edge_src
    c_action_component.edgeDst = edge_dst
    c_action_component.demandDst = demand_dst
    c_action_component.value = value
    return c_action_component


DROP_REASONS = [
    "Ipv4L3Protocol::DROP_NO_ROUTE",
    "Ipv4L3Protocol::DROP_TTL_EXPIRE",
    "Ipv4L3Protocol::DROP_BAD_CHECKSUM",
    "Ipv4L3Protocol::DROP_QUEUE",
    "Ipv4L3Protocol::DROP_QUEUE_DISC",
    "Ipv4L3Protocol::DROP_INTERFACE_DOWN",
    "Ipv4L3Protocol::DROP_ROUTE_ERROR",
    "Ipv4L3Protocol::DROP_FRAGMENT_TIMEOUT",
    "Ipv4L3Protocol::DROP_INVALID_REASON",
    "PointToPointNetDevice::MacTxDrop",
    "PointToPointNetDevice::PhyTxDrop",
    "PointToPointNetDevice::PhyRxDrop",
    "QueueDisc::Drop",
]


def monitoring_from_ctype(monitoring_graph: MonitoringGlobalSnapshot,
                          monitoring_edges: List[MonitoringDirEdgeSnapshot],
                          matrices) -> (nx.DiGraph, dict, dict):
    """
    Converts a given ns-3 monitoring snapshot into a networkx representation.
    """
    global_feat = {k: getattr(monitoring_graph, k) for k, _ in monitoring_graph._fields_}
    num_nodes = global_feat.pop('numNodes')
    dropped_packets_per_reason_arr = global_feat.pop('droppedPacketsPerReason')
    dropped_bytes_per_reason_arr = global_feat.pop('droppedBytesPerReason')
    dropped_bytes_per_reason, dropped_packets_per_reason = dict(), dict()
    dropped_packets, dropped_bytes = 0, 0
    for i, reason in enumerate(DROP_REASONS):
        dropped_packets_per_reason[f"droppedPackets_{reason}"] = dropped_packets_per_reason_arr[i]
        dropped_bytes_per_reason[f"droppedBytes_{reason}"] = dropped_bytes_per_reason_arr[i]
        dropped_packets += dropped_packets_per_reason_arr[i]
        dropped_bytes += dropped_bytes_per_reason_arr[i]
    global_feat.update(droppedPackets=dropped_packets, droppedBytes=dropped_bytes)
    monitoring = nx.DiGraph(**global_feat)

    per_node_stats = {
        "sentPackets": np.sum(matrices['sent_packets'], axis=1),
        "receivedPackets": np.sum(matrices['received_packets'], axis=1),
        "sentBytes": np.sum(matrices['sent_bytes'], axis=1),
        "receivedBytes": np.sum(matrices['received_bytes'], axis=1),
        "retransmittedPackets": np.sum(matrices['retransmitted_packets'], axis=1),
        "retransmittedBytes": np.sum(matrices['retransmitted_bytes'], axis=1),
    }

    # add nodes with their features
    for node_id in range(num_nodes):
        monitoring.add_node(node_id, **{k: stats[node_id] for k, stats in per_node_stats.items()})

    # placeholders for aggregated global features
    global_sent_bits = 0
    global_sendable_bits = 0
    global_LUs = list()

    # add edges with their features (some features are also used for global features)
    for monitoring_edge in monitoring_edges:
        edge_attrs = {key: getattr(monitoring_edge, key) for (key, _) in monitoring_edge._fields_}

        # skip edge if it is not up (TODO: add this edge to monitoring, but just for vis (keep it disabled for policy)
        if not edge_attrs['isLinkUp']:
            continue

        src = edge_attrs.pop("src")
        dst = edge_attrs.pop("dst")

        sent_bits = edge_attrs['sentBytes'] * 8
        global_sent_bits += sent_bits

        sendable_bits = round(edge_attrs['channelDataRate'] * monitoring_graph.elapsedTime)
        global_sendable_bits += sendable_bits

        # cap LU to 1.0, because border cases (e.g. packets stuck in the pipe at timestep beginning)
        # can lead to values slightly above 1.0
        link_utilization = 0. if sendable_bits == 0 else min(sent_bits / sendable_bits, 1.0)
        global_LUs.append(link_utilization)
        edge_attrs.update(LU=link_utilization)

        # update aggregate edge features (note: these features could also be aggregated in nodes)
        edge_attrs.update(
            txQueueLastLoad=round(edge_attrs['txQueueLastLoad'] / edge_attrs['txQueueCapacity'], 3),
            txQueueMaxLoad=round(edge_attrs['txQueueMaxLoad'] / edge_attrs['txQueueCapacity'], 3),
            queueDiscLastLoad=round(edge_attrs['queueDiscLastLoad'] / edge_attrs['queueDiscCapacity'], 3),
            queueDiscMaxLoad=round(edge_attrs['queueDiscMaxLoad'] / edge_attrs['queueDiscCapacity'], 3),
        )
        monitoring.add_edge(src, dst, **edge_attrs)

    # add aggregated global graph features
    monitoring.graph["sendableBytes"] = global_sendable_bits // 8
    monitoring.graph['maxLU'] = 0. if monitoring_graph.elapsedTime == 0 else max(global_LUs)
    monitoring.graph['avgTDU'] = 0. if monitoring_graph.elapsedTime == 0 else global_sent_bits / global_sendable_bits

    return monitoring, dropped_packets_per_reason, dropped_bytes_per_reason


def traffic_matrix_from_ctype(num_nodes: int,
                              tm_entries: List[Union[TrafficPacketMatrixEntry, TrafficByteMatrixEntry]]) -> np.ndarray:
    """
    Converts a given traffic matrix from ctype-compatible representation into a numpy array.
    """
    traffic_matrix = np.zeros((num_nodes, num_nodes), dtype=np.uint64)
    for entry in tm_entries:
        traffic_matrix[entry.src, entry.dst] = entry.amount
    return traffic_matrix
