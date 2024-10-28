/*
 * These structs define basic components of the entire simulation framework
 * such as network graphs and their parts, monitoring structures and
 * action spaces. As we're using ns3-ai and its shared memory paradigm, we have to
 * define all those data structures that are accessed both from the simulator
 * side (C++) as well as the learning/control side (python) as structs on both
 * sides and even take care of the member ordering to guarantee identical struct
 * sizes.
 */

#ifndef NS3_PACKERL_SHARED_STRUCTS_H
#define NS3_PACKERL_SHARED_STRUCTS_H

#include "ns3/ptr.h"
#include "../monitoring/monitoring-dir-edge.h"

#include <stdint.h>
#include <vector>
#include <tuple>

using std::vector, std::tuple;

/**
 * We can send arbitrary amounts of data between the C++ and Python side, but we have to
 * define a maximum number of elements for each data structure. This is due to the fact that
 * we have to allocate memory for these structures in the shared memory space, and we have to
 * know the maximum size of the memory block to allocate. If there is more data to send than
 * the maximum size, we have to split the data into multiple blocks and send them one after the
 * other.
 */
const uint32_t MAX_ELEMS = 1000;

/**
 * Models a network edge, which is a point-to-point communication channel characterized by
 * bandwidth, 2 nodes, as well as channel delay.
 */
typedef struct TopologyEdge
{
    /**
     * First node ID
     */
    uint32_t fst;

    /**
     * Second node ID
     */
    uint32_t snd;

    /**
     * Datarate in bps
     */
    uint64_t datarate;

    /**
     * Delay in ms
     */
    uint32_t delay;

} TopologyEdge;


/**
 * Models a traffic demand event, which is characterized by a source and destination node,
 * an amount of data to send, a datarate, and a flag indicating whether the traffic is TCP or not.
 * In the 'upcomingEventTypes' array, this event is represented by the value 0.
 */
typedef struct TrafficDemand
{
    /**
     * The time the demand is created.
     */
    double t;

    /**
     * The source node ID.
     */
    uint32_t src;

    /**
     * The destination node ID.
     */
    uint32_t dst;

    /**
     * The amount of bits of the demand.
     */
    uint64_t amount;

    /**
     * The datarate of the demand in bps. This is zero for TCP traffic since senders
     * will adapt their sending rate to the network conditions.
     */
    uint64_t datarate;

    /**
     * Flag indicating whether the demand is TCP or not.
     */
    bool isTCP;

} TrafficDemand;


/**
 * Models a link failure event, which is characterized by a timestamp and the two nodes that are
 * connected by the failed link. In the 'upcomingEventTypes' array, this event is represented by the value 1.
 */
typedef struct LinkFailure
{
    /**
     * The time the link failure occurs.
     */
    double t;

    /**
     * The first node ID.
     */
    uint32_t fst;

    /**
     * The second node ID.
     */
    uint32_t snd;

} LinkFailure;


/**
 * Models an event that can occur in the simulation. This is a union type that can hold
 * either a TrafficDemand or a LinkFailure event. The type of the event is determined by
 * the 'upcomingEventTypes' array.
 */
union Event {
    TrafficDemand trafficDemand;
    LinkFailure linkFailure;
};


/**
 * Models a snapshot of a monitoring graph, which is taken with respect
 * to a certain amount of passed simulation time. Such snapshots hold network performance
 * statistics within the given time frame.
 */
typedef struct MonitoringGlobalSnapshot
{
    uint32_t numNodes;
    uint64_t sentPackets;
    uint64_t sentBytes;
    uint64_t receivedPackets;
    uint64_t receivedBytes;
    uint64_t droppedPacketsPerReason[13];  // 13 reasons for dropping packets
    uint64_t droppedBytesPerReason[13];  // 13 reasons for dropping packets
    uint64_t retransmittedPackets;
    uint64_t retransmittedBytes;

    double avgPacketDelay;
    double maxPacketDelay;
    double avgPacketJitter;
    double elapsedTime;

} MonitoringGlobalSnapshot;


/**
 * Models a snapshot of a (directed) monitoring edge, which is taken with respect
 * to a certain amount of passed simulation time. Such snapshots hold network performance
 * statistics within the given time frame, in this case specific to an outgoing
 * net device sending data over its incident channel.
 */
typedef struct MonitoringDirEdgeSnapshot
{
    /**
     * Source node ID.
     */
    uint32_t src;

    /**
     * Destination node ID.
     */
    uint32_t dst;

    /**
     * Channel data rate in bps.
     */
    uint64_t channelDataRate;

    /**
     * Channel delay in ms.
     */
    uint32_t channelDelay;

    /**
     * The maximum amount of packets the outgoing NetDevice's TxQueue can hold.
     */
    uint32_t txQueueCapacity;

    /**
     * The maximum load of the outgoing NetDevice's TxQueue throughout the
     * time period monitored (in packets).
     */
    uint32_t txQueueMaxLoad;

    /**
     * The load of the outgoing NetDevice's TxQueue at the moment of snapshotting
     * (in packets).
     */
    uint32_t txQueueLastLoad;

    /**
     * The maximum amount of packets the outgoing queueDisc can hold.
     */
    uint32_t queueDiscCapacity;

    /**
     * The maximum load of the outgoing queueDisc throughout the
     * time period monitored (in packets).
     */
    uint32_t queueDiscMaxLoad;

    /**
     * The load of the outgoing queueDisc at the moment of snapshotting
     * (in packets).
     */
    uint32_t queueDiscLastLoad;

    /**
     * Amount of packets sent out from the outgoing NetDevice.
     */
    uint64_t sentPackets;

    /**
     * Amount of bytes sent out from the outgoing NetDevice.
     * uint64_t is enough for 213.5 days of continuous transmission with 1Tbps
     */
    uint64_t sentBytes;

    /**
     * Amount of packets received at the incoming NetDevice.
     */
    uint64_t receivedPackets;

    /**
     * Amount of bytes received at  the incoming NetDevice.
     * uint64_t is enough for 213.5 days of continuous transmission with 1Tbps
     */
    uint64_t receivedBytes;

    /**
     * Amount of packets dropped in one of the incident NetDevices,
     * or on the way between them.
     */
    uint64_t droppedPackets;

    /**
     * Amount of bytes dropped in one of the incident NetDevices,
     * or on the way between them.
     */
    uint64_t droppedBytes;

    /**
     * The amount of time the snapshot encompasses.
     */
    double elapsedTime;

    /**
     * Flag indicating whether the link is up or down.
     */
    bool isLinkUp;

} MonitoringDirEdgeSnapshot;

/**
 * Create a monitoring edge snapshot from given monitoring graph and simulation time.
 * Also resets the edge's statistics.
 * @param edge The monitoring edge to read from (and to reset)
 * @param simTime The amount of time the snapshot represents
 * @return The created monitoring edge snapshot
 */
const MonitoringDirEdgeSnapshot makeMonitoringDirEdgeSnapshot(ns3::Ptr<MonitoringDirEdge> edge, double simTime);

/**
 * Models a matrix entry for packet traffic, which is characterized by a source and destination node,
 * as well as the amount of packets sent between them.
 */
typedef struct TrafficPacketMatrixEntry
{
    /**
     * The source node ID.
     */
    uint32_t src;

    /**
     * The destination node ID.
     */
    uint32_t dst;

    /**
     * The amount of sent packets.
     */
    uint32_t amount;

} TrafficPacketMatrixEntry;


/**
 * Models a matrix entry for byte traffic, which is characterized by a source and destination node,
 * as well as the amount of bytes sent between them.
 */
typedef struct TrafficByteMatrixEntry
{
    /**
     * The source node ID.
     */
    uint32_t src;

    /**
     * The destination node ID.
     */
    uint32_t dst;

    /**
     * The amount of sent bytes.
     */
    uint64_t amount;

} TrafficByteMatrixEntry;


/**
 * Models the shared memory object for the 'environment'.
 * The environment consists of all simulation-related information such as the topology, the current
 * traffic demands, the current monitoring, and some control signals to steer the simulation/learning
 * interaction.
 */
typedef struct PackerlEnvStruct
{
    // ==== TOPOLOGY =============================================================================================

    /**
     * Amount of nodes in the graph
     */
    uint32_t numTopologyNodes;

    /**
     * Amount of edges in the graph
     */
    TopologyEdge topologyEdges[MAX_ELEMS];

    /**
     * Amount of edges that are currently communicated
     */
    uint32_t numTopologyEdgesAvailable;

    /**
     * true if the topology is available in the shared memory, false otherwise
     */
    bool topologyAvailable = false;

    /**
     * true if all topology edges have been sent, false otherwise
     */
    bool allTopologyEdgesSent = false;


    // ==== UPCOMING EVENTS (TRAFFIC, LINK FAILURE ETC.) ======================================================

    /**
     * Upcoming events in the simulation
     */
    Event upcomingEvents[MAX_ELEMS];

    /**
     * Types of upcoming events in the simulation
     */
    uint32_t upcomingEventTypes[MAX_ELEMS];

    /**
     * Amount of upcoming events that are currently communicated
     */
    uint32_t numUpcomingEventsAvailable;

    /**
     * true if upcoming events are available in the shared memory, false otherwise
     */
    bool upcomingEventsAvailable = false;

    /**
     * true if all upcoming events have been sent, false otherwise
     */
    bool allUpcomingEventsSent = false;

    // ==== MONITORING =======================================================================================

    MonitoringGlobalSnapshot monitoringGlobal;

    MonitoringDirEdgeSnapshot monitoringDirEdges[MAX_ELEMS];
    uint32_t numAvailableMonitoringDirEdges;

    TrafficPacketMatrixEntry sentPacketEntries[MAX_ELEMS];
    uint32_t numAvailableSentPacketEntries;

    TrafficPacketMatrixEntry receivedPacketEntries[MAX_ELEMS];
    uint32_t numAvailableReceivedPacketEntries;

    TrafficByteMatrixEntry sentByteEntries[MAX_ELEMS];
    uint32_t numAvailableSentByteEntries;

    TrafficByteMatrixEntry receivedByteEntries[MAX_ELEMS];
    uint32_t numAvailableReceivedByteEntries;

    TrafficPacketMatrixEntry retransmittedPacketEntries[MAX_ELEMS];
    uint32_t numAvailableRetransmittedPacketEntries;

    TrafficByteMatrixEntry retransmittedByteEntries[MAX_ELEMS];
    uint32_t numAvailableRetransmittedByteEntries;

    /**
     * true if monitoring data is available in the shared memory, false otherwise
     */
    bool monitoringAvailable = false;

    bool allMonitoringDirEdgesSent = false;
    bool allSentPacketEntriesSent = false;
    bool allReceivedPacketEntriesSent = false;
    bool allSentByteEntriesSent = false;
    bool allReceivedByteEntriesSent = false;
    bool allRetransmittedPacketEntriesSent = false;
    bool allRetransmittedByteEntriesSent = false;


    // ==== MISC =======================================================================================

    /**
     * Set to true by the sim when the simulation is ready to be used by the RL learning loop
     */
    bool simReady = false;

    /**
     * Set to false by the RL learning loop at the start. If set to true by the sim, the current env is terminated
     */
    bool done = false;

} PackerlEnvStruct;


/**
 * Models a routing action component, which is a tuple of 4 values:
 * - the source node of the edge
 * - the destination node of the edge
 * - the destination of the packet
 * - the value of the routing action
 */
typedef struct RoutingActionComponent
{
    uint32_t edgeSrc;
    uint32_t edgeDst;
    uint32_t demandDst;
    float value;

} RoutingActionComponent;


/**
 * Models the shared memory object for the 'action space'.
 * It consists of the currently selected action of the model as well as
 * a logical flow control variable. Each action consists of providing
 * routing descriptors for all nodes.
 */
typedef struct PackerlActStruct
{
    /**
     * The routing actions for all nodes in the network
     */
    RoutingActionComponent routingActions [MAX_ELEMS];

    /**
     * The amount of routing actions that are currently communicated in the shared memory
     */
    uint32_t numRoutingActionsAvailable;

    /**
     * true if routing actions are available in the shared memory, false otherwise
     */
    bool actionsAvailable = false;

    /**
     * true if all routing actions have been sent, false otherwise
     */
    bool allActionsSent = false;

} PackerlActStruct;

#endif /*NS3_PACKERL_SHARED_STRUCTS_H*/
