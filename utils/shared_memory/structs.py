"""
These structs define basic components of the entire simulation framework
such as network graphs and their parts, monitoring structures and
action spaces. As we're using ns3-ai and its shared memory paradigm, we have to
define all data structures that are accessed both from the simulator
side (C++) as well as the learning/control side (python) as structs on both
sides, and even take care of the member ordering to guarantee identical struct
sizes.

Consult shared-structs.h in the ns3-packerl module for a more detailed documentation
of all the structs' members.
"""
from ctypes import Structure, c_bool, c_double, c_uint32, \
    c_uint64, c_float, sizeof, Union as cUnion


"""
We can send arbitrary amounts of data between the C++ and Python side, but we have to
define a maximum number of elements for each data structure. This is due to the fact that
we have to allocate memory for these structures in the shared memory space, and we have to
know the maximum size of the memory block to allocate. If there is more data to send than
the maximum size, we have to split the data into multiple blocks and send them one after the
other.
"""
MAX_ELEMS = 1000


class BaseStructure(Structure):
    """
    Base class for all shared memory structures.
    It is implemented as a Ctypes.Structure with integrated default values.
    """

    def __init__(self, **kwargs):
        """
        Initialize the structure.
        :param kwargs: values different to defaults
        :type kwargs: dict
        """
        values = type(self)._defaults_.copy()
        values.update(kwargs)

        super().__init__(**values)


class TopologyEdge(BaseStructure):
    """
    Models a network edge, which is a point-to-point communication channel characterized by
    datarate, 2 nodes, as well as channel delay in ms.
    """
    _fields_ = [('fst', c_uint32),
                ('snd', c_uint32),
                ('datarate', c_uint64),
                ('delay', c_uint32)]
    _defaults_ = {}


class TrafficDemand(BaseStructure):
    """
    Models a traffic demand event, which is characterized by a source and destination node,
    an amount of data to send, a datarate, and a flag indicating whether the traffic is TCP or not.
    In the 'upcomingEventTypes' array, this event is represented by the value 0.
    """
    _fields_ = [('t', c_double),
                ('src', c_uint32),
                ('dst', c_uint32),
                ('amount', c_uint64),
                ('datarate', c_uint64),
                ('isTCP', c_bool)]
    _defaults_ = {}


class LinkFailure(BaseStructure):
    """
    Models a link failure event, which is characterized by a timestamp and the two nodes that are
    connected by the failed link. In the 'upcomingEventTypes' array, this event is represented by the value 1.
    """
    _fields_ = [('t', c_double),
                ('fst', c_uint32),
                ('snd', c_uint32)]
    _defaults_ = {}


class Event(cUnion):
    """
    A union type that can hold either a TrafficDemand or a LinkFailure event.
    """
    _fields_ = [('trafficDemand', TrafficDemand),
                ('linkFailure', LinkFailure)]
    _defaults_ = {}


class MonitoringGlobalSnapshot(BaseStructure):
    """
    Models a snapshot of a monitoring graph, which is taken with respect
    to a certain amount of passed simulation time. Such snapshots hold network performance
    statistics within the given time frame.
    """
    _fields_ = [('numNodes', c_uint32),
                ('sentPackets', c_uint64),
                ('sentBytes', c_uint64),
                ('receivedPackets', c_uint64),
                ('receivedBytes', c_uint64),
                ('droppedPacketsPerReason', c_uint64 * 13),  # we currently track 13 reasons for drops
                ('droppedBytesPerReason', c_uint64 * 13),  # we currently track 13 reasons for drops
                ('retransmittedPackets', c_uint64),
                ('retransmittedBytes', c_uint64),
                ('avgPacketDelay', c_double),
                ('maxPacketDelay', c_double),
                ('avgPacketJitter', c_double),
                ('elapsedTime', c_double)]
    _defaults_ = {}


class MonitoringDirEdgeSnapshot(BaseStructure):
    """
    Models a snapshot of a (directed) monitoring edge, which is taken with respect
    to a certain amount of passed simulation time. Such snapshots hold network performance
    statistics within the given time frame, in this case specific to an outgoing
    net device sending data over its incident channel.
    """
    _fields_ = [('src', c_uint32),
                ('dst', c_uint32),
                ('channelDataRate', c_uint64),
                ('channelDelay', c_uint32),
                ('txQueueCapacity', c_uint32),
                ('txQueueMaxLoad', c_uint32),
                ('txQueueLastLoad', c_uint32),
                ('queueDiscCapacity', c_uint32),
                ('queueDiscMaxLoad', c_uint32),
                ('queueDiscLastLoad', c_uint32),
                ('sentPackets', c_uint64),
                ('sentBytes', c_uint64),
                ('receivedPackets', c_uint64),
                ('receivedBytes', c_uint64),
                ('droppedPackets', c_uint64),
                ('droppedBytes', c_uint64),
                ('elapsedTime', c_double),
                ('isLinkUp', c_bool)]
    _defaults_ = {}

class TrafficPacketMatrixEntry(BaseStructure):
    """
    Models a matrix entry for packet traffic, which is characterized by a source and destination node,
    as well as the amount of packets sent between them.
    """
    _fields_ = [('src', c_uint32),
                ('dst', c_uint32),
                ('amount', c_uint32)]
    _defaults_ = {}


class TrafficByteMatrixEntry(BaseStructure):
    """
    Models a matrix entry for byte traffic, which is characterized by a source and destination node,
    as well as the amount of bytes sent between them.
    """
    _fields_ = [('src', c_uint32),
                ('dst', c_uint32),
                ('amount', c_uint64)]
    _defaults_ = {}


class PackerlEnvStruct(BaseStructure):
    """
    Models the shared memory object for the 'environment'.
    The environment consists of all simulation-related information such as the topology, the current
    traffic demands, the current monitoring, and some control signals to steer the simulation/learning
    interaction.
    """
    _fields_ = [('numTopologyNodes', c_uint32),
                ('topologyEdges', TopologyEdge * MAX_ELEMS),
                ('numTopologyEdgesAvailable', c_uint32),
                ('topologyAvailable', c_bool),
                ('allTopologyEdgesSent', c_bool),

                ('upcomingEvents', Event * MAX_ELEMS),
                ('upcomingEventTypes', c_uint32 * MAX_ELEMS),
                ('numUpcomingEventsAvailable', c_uint32),
                ('upcomingEventsAvailable', c_bool),
                ('allUpcomingEventsSent', c_bool),

                ('monitoringGlobal', MonitoringGlobalSnapshot),
                ('monitoringDirEdges', MonitoringDirEdgeSnapshot * MAX_ELEMS),
                ('numAvailableMonitoringDirEdges', c_uint32),
                ('sentPacketEntries', TrafficPacketMatrixEntry * MAX_ELEMS),
                ('numAvailableSentPacketEntries', c_uint32),
                ('receivedPacketEntries', TrafficPacketMatrixEntry * MAX_ELEMS),
                ('numAvailableReceivedPacketEntries', c_uint32),
                ('sentByteEntries', TrafficByteMatrixEntry * MAX_ELEMS),
                ('numAvailableSentByteEntries', c_uint32),
                ('receivedByteEntries', TrafficByteMatrixEntry * MAX_ELEMS),
                ('numAvailableReceivedByteEntries', c_uint32),
                ('retransmittedPacketEntries', TrafficPacketMatrixEntry * MAX_ELEMS),
                ('numAvailableRetransmittedPacketEntries', c_uint32),
                ('retransmittedByteEntries', TrafficByteMatrixEntry * MAX_ELEMS),
                ('numAvailableRetransmittedByteEntries', c_uint32),

                ('monitoringAvailable', c_bool),
                ('allMonitoringDirEdgesSent', c_bool),
                ('allSentPacketEntriesSent', c_bool),
                ('allReceivedPacketEntriesSent', c_bool),
                ('allSentByteEntriesSent', c_bool),
                ('allReceivedByteEntriesSent', c_bool),
                ('allRetransmittedPacketEntriesSent', c_bool),
                ('allRetransmittedByteEntriesSent', c_bool),
                ('simReady', c_bool),
                ('done', c_bool)]

    _defaults_ = {
        'topologyAvailable': False,
        'allTopologyEdgesSent': False,
        'upcomingEventsAvailable': False,
        'allUpcomingEventsSent': False,
        'monitoringAvailable': False,
        'allMonitoringDirEdgesSent': False,
        'allSentPacketEntriesSent': False,
        'allReceivedPacketEntriesSent': False,
        'allSentByteEntriesSent': False,
        'allReceivedByteEntriesSent': False,
        'allRetransmittedPacketEntriesSent': False,
        'allRetransmittedByteEntriesSent': False,
        'simReady': False,
        'done': False
    }


# ====================================================================


class RoutingActionComponent(BaseStructure):
    """
    Models a routing action component, which is a tuple of 4 values:
    - the source node of the edge
    - the destination node of the edge
    - the destination of the packet
    - the value of the routing action
    """
    _fields_ = [('edgeSrc', c_uint32),
                ('edgeDst', c_uint32),
                ('demandDst', c_uint32),
                ('value', c_float)]
    _defaults_ = {}


class PackerlActStruct(BaseStructure):
    """
    Models the shared memory object for the 'action space'.
    It consists of the currently selected action of the model as well as
    a logical flow control variable. Each action consists of providing
    routing descriptors for all nodes.
    """
    _fields_ = [('routingActions', RoutingActionComponent * MAX_ELEMS),
                ('numRoutingActionsAvailable', c_uint32),
                ('actionsAvailable', c_bool),
                ('allActionsSent', c_bool)]
    _defaults_ = {'actionsAvailable': False,
                  'allActionsSent': False}


# ====================================================================

# A dict containing all memory block sizes (for convenient access)
ALL_SHM_SIZES = {
    'ENV': sizeof(PackerlEnvStruct),
    'r.ac. component': sizeof(RoutingActionComponent),
    'ACT': sizeof(PackerlActStruct),
    'TOTAL': sizeof(PackerlEnvStruct) + sizeof(PackerlActStruct)
}
