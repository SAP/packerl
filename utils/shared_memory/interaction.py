"""
This module provides functions to place data into and read data from shared memory.
"""
from typing import List, Any
from copy import deepcopy

from scenarios.events import Event
import numpy as np

from utils.shared_memory.structs import MAX_ELEMS
from utils.shared_memory.context_mgmt import timed_fragile, SharedMemUnavailableException
from utils.shared_memory.conversion import (topology_edge_to_ctype,
                                            event_to_ctype,
                                            action_component_to_ctype,
                                            monitoring_from_ctype,
                                            traffic_matrix_from_ctype,
                                            )


def check_shm(sim, logger):
    """
    Check whether ns3 is ready to interact via shared memory
    """
    logger.log_function("shm::check_shm()")
    sim_ready = False  # sim is ready if done flag is set to False
    while not sim_ready:
        with sim as data:

            # ns3 not ready or some error
            if not data:
                raise SharedMemUnavailableException()
            sim_ready = data.env.simReady

    logger.log_function("shm::check_shm() END")


# === PLACING =========================================================================================================


def place_done(sim, timeout, logger, done):
    """
    Sets the done flag in shared memory to signify the sim that the current episode is over
    """
    logger.log_function("shm::set_done()")
    with timed_fragile(sim, max_seconds=timeout) as data:
        data.env.done = done
    logger.log_function("shm::set_done() END")


def place_network_graph(sim, timeout, logger, network_graph):
    """
    Places the given network graph into the shared memory so that
    the simulation can read and install it.
    """
    logger.log_function("shm::place_network_graph()")
    remaining_edges = list(network_graph.edges(data=True))
    while len(remaining_edges) > 0:
        with timed_fragile(sim, max_seconds=timeout) as data:

            # ns3 not ready or some error
            if not data:
                raise SharedMemUnavailableException()
            if data.env.topologyAvailable:
                logger.log_logic("ns3 has not yet consumed past topology")
                raise timed_fragile.Break  # release shared memory

            # if the remaining edges are less than MAX_ELEMS, send all of them (and general info)
            if len(remaining_edges) <= MAX_ELEMS:
                for i, (fst, snd, edge_attrs) in enumerate(remaining_edges):
                    logger.log_logic(f"sending edge {fst}->{snd}")
                    data.env.topologyEdges[i] = topology_edge_to_ctype(fst, snd, edge_attrs)
                data.env.numTopologyEdgesAvailable = len(remaining_edges)
                data.env.allTopologyEdgesSent = True
                remaining_edges = []

            # if the remaining edges are more than MAX_ELEMS, send MAX_ELEMS of them
            else:
                edges_to_be_sent = remaining_edges[:MAX_ELEMS]
                for i, (fst, snd, edge_attrs) in enumerate(edges_to_be_sent):
                    logger.log_logic(f"sending edge {fst}->{snd}")
                    data.env.topologyEdges[i] = topology_edge_to_ctype(fst, snd, edge_attrs)
                data.env.numTopologyEdgesAvailable = len(edges_to_be_sent)
                data.env.allTopologyEdgesSent = False
                remaining_edges = remaining_edges[MAX_ELEMS:]

            data.env.numTopologyNodes = network_graph.number_of_nodes()
            data.env.topologyAvailable = True  # signals ns3 that the topology is ready to be consumed

    # wait until ns3 fully consumes the data, then yield shm control to ns3 (by simply accessing the shm)
    data_consumed = False
    while not data_consumed:
        with timed_fragile(sim, max_seconds=timeout) as data:
            data_consumed = not data.env.topologyAvailable
    logger.log_function("shm::place_network_graph() END")


def place_events(sim, timeout, logger, events) -> None:
    """
    Places the given events into the shared memory so that the simulation can read and execute them.
    """
    logger.log_function("shm::place_events()")
    remaining_events: List[Event] = deepcopy(events)
    all_upcoming_events_sent = False

    while not all_upcoming_events_sent:
        with timed_fragile(sim, max_seconds=timeout) as data:

            # ns3 not ready or some error
            if not data:
                raise SharedMemUnavailableException()
            if data.env.upcomingEventsAvailable:
                logger.log_logic("ns3 has not yet consumed previously sent upcoming events")
                raise timed_fragile.Break  # release shared memory

            # check whether we can send all events this time
            if len(remaining_events) <= MAX_ELEMS:
                all_events_sendable = True
                events_to_send = remaining_events
                new_remaining_events = []
            else:
                all_events_sendable = False
                events_to_send = remaining_events[:MAX_ELEMS]
                new_remaining_events = remaining_events[MAX_ELEMS:]

            # place the events
            for i, event in enumerate(events_to_send):
                logger.log_logic(f"sending event '{event}'")
                c_event_type_id, c_event = event_to_ctype(event)
                c_event_field = "trafficDemand" if c_event_type_id == 0 else "linkFailure"
                setattr(data.env.upcomingEvents[i], c_event_field, c_event)  # set the event depending on type
                data.env.upcomingEventTypes[i] = c_event_type_id

            # bookkeeping and signaling
            all_upcoming_events_sent = all_events_sendable
            data.env.numUpcomingEventsAvailable = len(events_to_send)
            data.env.allUpcomingEventsSent = all_upcoming_events_sent
            data.env.upcomingEventsAvailable = True  # signals ns3 that the events are ready to be consumed
            remaining_events = new_remaining_events
            logger.log_logic(f"Remaining events: {len(remaining_events)}")

    # wait until ns3 fully consumes the data, then yield shm control to ns3 (by simply accessing the shm)
    data_consumed = False
    while not data_consumed:
        with timed_fragile(sim, max_seconds=timeout) as data:
            data_consumed = not data.env.upcomingEventsAvailable
    logger.log_function("shm::place_events() END")


def place_actions(sim, timeout, logger, actions) -> None:
    """
    Places the given actions into the shared memory so that the simulation can read and execute them.
    """
    logger.log_function("shm::place_actions()")
    num_edges, num_nodes = actions.edge_attr.shape

    unfolded_values: list = actions.edge_attr.cpu().numpy().flatten().tolist()
    unfolded_edge_idx = actions.edge_index.cpu().clone().repeat_interleave(num_nodes, dim=1).t().numpy()
    unfolded_edge_idx = [tuple(row) for row in unfolded_edge_idx]
    unfolded_dest_nodes = np.tile(np.arange(num_nodes), num_edges).tolist()
    remaining_action_components = list(zip(unfolded_edge_idx, unfolded_dest_nodes, unfolded_values))

    while len(remaining_action_components) > 0:
        with timed_fragile(sim, max_seconds=timeout) as data:

            # ns3 not ready or some error
            if not data:
                raise SharedMemUnavailableException()
            if data.act.actionsAvailable:
                logger.log_logic("ns3 has not yet consumed previously sent actions")
                raise timed_fragile.Break  # release shared memory

            if len(remaining_action_components) <= MAX_ELEMS:
                for i, ((edge_src, edge_dst), demand_dst, value) in enumerate(remaining_action_components):
                    logger.log_logic(f"placing action: {edge_src}->{edge_dst} to {demand_dst} with value {value}")
                    data.act.routingActions[i] = action_component_to_ctype(edge_src, edge_dst, demand_dst, value)
                num_sent_actions = len(remaining_action_components)
                data.act.numRoutingActionsAvailable = num_sent_actions
                data.act.allActionsSent = True
                remaining_action_components = []

            else:
                action_components_to_be_sent = remaining_action_components[:MAX_ELEMS]
                for i, ((edge_src, edge_dst), demand_dst, value) in enumerate(action_components_to_be_sent):
                    logger.log_logic(f"placing action: {edge_src}->{edge_dst} to {demand_dst} with value {value}")
                    data.act.routingActions[i] = action_component_to_ctype(edge_src, edge_dst, demand_dst, value)
                num_sent_actions = len(action_components_to_be_sent)
                data.act.numRoutingActionsAvailable = num_sent_actions
                data.act.allActionsSent = False
                remaining_action_components = remaining_action_components[MAX_ELEMS:]

            data.act.actionsAvailable = True  # signals ns3 that the actions are ready to be consumed

    # wait until ns3 fully consumes the data, then yield shm control to ns3 (by simply accessing the shm)
    data_consumed = False
    while not data_consumed:
        with timed_fragile(sim, max_seconds=timeout) as data:
            data_consumed = not data.act.actionsAvailable
    logger.log_function("shm::place_actions() END")


def place_shm(sim, timeout, logger, mode: str, data: Any):
    """
    Places the given data into the shared memory according to the given mode.
    """
    if mode == "done":
        place_done(sim, timeout, logger, data)
    elif mode == "network_graph":
        place_network_graph(sim, timeout, logger, data)
    elif mode == "actions":
        place_actions(sim, timeout, logger, data)
    elif mode == "events":
        place_events(sim, timeout, logger, data)
    else:
        raise ValueError(f"Unknown shm placement mode: {mode}")


# === READING =========================================================================================================


def read_monitoring(sim, timeout, logger):
    """
    Reads the monitoring components (monitoring graph, traffic matrices etc.) from shared memory
    """
    logger.log_function("shm::read_monitoring()")
    monitoring_global = None
    monitoring_edges = []
    sent_packet_entries = []
    received_packet_entries = []
    sent_byte_entries = []
    received_byte_entries = []
    retransmitted_packet_entries = []
    retransmitted_byte_entries = []

    monitoring_global_received = False
    monitoring_edges_received = False
    sent_packet_entries_received = False
    received_packet_entries_received = False
    sent_byte_entries_received = False
    received_byte_entries_received = False
    retransmitted_packet_entries_received = False
    retransmitted_byte_entries_received = False
    monitoring_received = False  # only true if all monitoring components are received

    while not monitoring_received:
        with timed_fragile(sim, max_seconds=timeout) as data:

            # ns3 not ready or some error
            if not data:
                raise SharedMemUnavailableException()
            if not data.env.monitoringAvailable:
                logger.log_logic("ns3 has not yet sent monitoring")
                raise timed_fragile.Break  # release shared memory

            if not monitoring_global_received:
                monitoring_global = deepcopy(data.env.monitoringGlobal)
                logger.log_logic("read globalSnapshot")
                monitoring_global_received = True

            if not monitoring_edges_received:
                num_edges_sent = data.env.numAvailableMonitoringDirEdges
                for i in range(num_edges_sent):
                    new_monitoring_edge = deepcopy(data.env.monitoringDirEdges[i])
                    monitoring_edges.append(new_monitoring_edge)
                    logger.log_logic(f"read edgeSnapshot {new_monitoring_edge.src}->{new_monitoring_edge.dst}")
                monitoring_edges_received = data.env.allMonitoringDirEdgesSent

            if not sent_packet_entries_received:
                num_sent_packet_entries_sent = data.env.numAvailableSentPacketEntries
                for i in range(num_sent_packet_entries_sent):
                    new_entry = deepcopy(data.env.sentPacketEntries[i])
                    sent_packet_entries.append(new_entry)
                    logger.log_logic(f"read sent packet entry {new_entry.src}->{new_entry.dst}, "
                                     f"{new_entry.amount} packets")
                sent_packet_entries_received = data.env.allSentPacketEntriesSent

            if not received_packet_entries_received:
                num_received_packet_entries_sent = data.env.numAvailableReceivedPacketEntries
                for i in range(num_received_packet_entries_sent):
                    new_entry = deepcopy(data.env.receivedPacketEntries[i])
                    received_packet_entries.append(new_entry)
                    logger.log_logic(f"read received packet entry {new_entry.src}->{new_entry.dst}, "
                                     f"{new_entry.amount} packets")
                received_packet_entries_received = data.env.allReceivedPacketEntriesSent

            if not sent_byte_entries_received:
                num_sent_byte_entries_sent = data.env.numAvailableSentByteEntries
                for i in range(num_sent_byte_entries_sent):
                    new_entry = deepcopy(data.env.sentByteEntries[i])
                    sent_byte_entries.append(new_entry)
                    logger.log_logic(f"read sent byte entry {new_entry.src}->{new_entry.dst}, "
                                     f"{new_entry.amount} bytes")
                sent_byte_entries_received = data.env.allSentByteEntriesSent

            if not received_byte_entries_received:
                num_received_byte_entries_sent = data.env.numAvailableReceivedByteEntries
                for i in range(num_received_byte_entries_sent):
                    new_entry = deepcopy(data.env.receivedByteEntries[i])
                    received_byte_entries.append(new_entry)
                    logger.log_logic(f"read received byte entry {new_entry.src}->{new_entry.dst}, "
                                     f"{new_entry.amount} bytes")
                received_byte_entries_received = data.env.allReceivedByteEntriesSent

            if not retransmitted_packet_entries_received:
                num_retransmitted_packet_entries_sent = data.env.numAvailableRetransmittedPacketEntries
                for i in range(num_retransmitted_packet_entries_sent):
                    new_entry = deepcopy(data.env.retransmittedPacketEntries[i])
                    retransmitted_packet_entries.append(new_entry)
                    logger.log_logic(f"read retransmitted packet entry {new_entry.src}->{new_entry.dst}, "
                                     f"{new_entry.amount} packets")
                retransmitted_packet_entries_received = data.env.allRetransmittedPacketEntriesSent

            if not retransmitted_byte_entries_received:
                num_retransmitted_byte_entries_sent = data.env.numAvailableRetransmittedByteEntries
                for i in range(num_retransmitted_byte_entries_sent):
                    new_entry = deepcopy(data.env.retransmittedByteEntries[i])
                    retransmitted_byte_entries.append(new_entry)
                    logger.log_logic(f"read retransmitted byte entry {new_entry.src}->{new_entry.dst}, "
                                     f"{new_entry.amount} bytes")
                retransmitted_byte_entries_received = data.env.allRetransmittedByteEntriesSent

            # mark monitoring as consumed, next monitoring should only be read once the next monitoring is provided
            data.env.monitoringAvailable = False
            monitoring_received = (monitoring_global_received
                                   and monitoring_edges_received
                                   and sent_packet_entries_received
                                   and received_packet_entries_received
                                   and sent_byte_entries_received
                                   and received_byte_entries_received
                                   and retransmitted_packet_entries_received
                                   and retransmitted_byte_entries_received)

    # convert monitoring components to traffic matrices
    sent_packet_tm = traffic_matrix_from_ctype(int(monitoring_global.numNodes), sent_packet_entries)
    received_packet_tm = traffic_matrix_from_ctype(int(monitoring_global.numNodes), received_packet_entries)
    sent_byte_tm = traffic_matrix_from_ctype(int(monitoring_global.numNodes), sent_byte_entries)
    received_byte_tm = traffic_matrix_from_ctype(int(monitoring_global.numNodes), received_byte_entries)
    retransmitted_packet_tm = traffic_matrix_from_ctype(int(monitoring_global.numNodes), retransmitted_packet_entries)
    retransmitted_byte_tm = traffic_matrix_from_ctype(int(monitoring_global.numNodes), retransmitted_byte_entries)

    matrices = {
        "sent_packets": sent_packet_tm,
        "received_packets": received_packet_tm,
        "sent_bytes": sent_byte_tm,
        "received_bytes": received_byte_tm,
        "retransmitted_packets": retransmitted_packet_tm,
        "retransmitted_bytes": retransmitted_byte_tm
    }
    logger.log_logic(f"finished reading matrices from monitoring components")

    # convert monitoring to networkx graph for policy input and visualization
    # WARNING: we need to reverse the edge list, as the monitoring is sent in reverse order.
    monitoring, dropped_packets_per_reason, dropped_bytes_per_reason\
        = monitoring_from_ctype(monitoring_global, monitoring_edges[::-1], matrices)
    logger.log_logic(f"finished creating monitoring graph: {monitoring}")
    logger.log_function("shm::read_monitoring() END")
    return monitoring, matrices, dropped_packets_per_reason, dropped_bytes_per_reason


def read_shm(sim, timeout, logger, mode: str):
    """
    Reads the data from the shared memory according to the given mode.
    """
    if mode == "monitoring":
        return read_monitoring(sim, timeout, logger)
    else:
        raise ValueError(f"Unknown shm read mode: {mode}")
