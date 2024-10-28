#ifndef NS3_PACKERL_H
#define NS3_PACKERL_H

#include <tuple>

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-layout-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/traffic-control-module.h"
#include "ns3/ns3-ai-module.h"

#include "utils/shared-structs.h"
#include "utils/sim-parameters.h"
#include "utils/types.h"
#include "applications/disposable-application.h"
#include "monitoring/monitoring-dir-edge.h"
#include "odd-routing/odd-label-map.h"


using std::string, std::pair, std::unordered_map, std::map, std::tuple, std::vector;

typedef unordered_map<U32Pair, U32Pair, pair_hash> UPPMap;
typedef unordered_map<U32Pair, uint32_t, pair_hash> UP32Map;
typedef unordered_map<U32Pair, uint64_t, pair_hash> UP64Map;
typedef unordered_map<U32Triple, uint32_t, triple_hash> UT32Map;
typedef unordered_map<U32Triple, uint64_t, triple_hash> UT64Map;
typedef unordered_map<uint32_t, unordered_map<uint32_t, map<uint32_t, double>>> OifDistMap;
typedef unordered_map<uint32_t, unordered_map<uint32_t, uint32_t>> OifMap;

namespace ns3
{

/**
 * The PackerlEnv class defines the simulation 'environment' for a reinforcement learning experiment
 * on routing optimization with ns3. It contains several methods to set up an ns3 simulation
 * environment including pre-defined/customizable topologies and applications, run simulation steps
 * of pre-configurable length and monitor the results and effects on the network. It uses a memory
 * block that is shared with a learning loop running in python that provides actions and reads the
 * monitoring, aiming to learn optimal routing behavior within the simulation scenario described
 * here.
 */
class PackerlEnv : public Ns3AIRL<PackerlEnvStruct, PackerlActStruct>
{
  public:
    /**
     * Constructor that sets/initializes necessary values
     */
    PackerlEnv(SimParameters simParameters);

    /**
     * Creates a network node with given ID.
     * @param i
     * @return true if successful, false otherwise.
     */
    bool createNode(uint32_t i);

    /**
     * Creates a network edge between two nodes.
     * @param edge
     * @return true if successful, false otherwise.
     */
    bool createEdge(const TopologyEdge& edge);

    /**
     * Creates a network topology from shm graph =(edge list and number of nodes).
     * @return true if setup was successful, false otherwise.
     */
    bool readAndInstallNetworkTopology();

    /**
     * Creates an application sending data from the given demand information
     * @param demand
     * @return true if successful, false otherwise.
     */
    bool createApplicationFromDemand(const TrafficDemand& demand);

    /**
     * Globally registers a packet retransmission event
     */
    void registerRetransmit(Ptr<const Packet> p, const TcpHeader& header, const Address& localAddr,
                            const Address& peerAddr, Ptr<const TcpSocketBase>);

    /**
     * Globally registers a dropped packet event on the physical layer (sender side).
     */
    void registerPhyTxDrop(Ptr<const Packet> p);

    /**
     * Globally registers a dropped packet event on the physical layer (receiver side).
     */
    void registerPhyRxDrop(Ptr<const Packet> p);

    /**
     * Globally registers a dropped packet event on the MAC layer (sender side).
     */
    void registerMacTxDrop(Ptr<const Packet> p);

    /**
     * Globally registers a dropped packet event from the queue disc.
     */
    void registerQueueDiscDrop(Ptr<const Packet> p);

    /**
     * Removes an application from the simulation.
     */
    void removeApplication(string context, Ptr<DisposableApplication> app);

    /**
     * Deactivates the specified link in the network.
     * @return true if successful, false otherwise.
     */
    bool executeLinkFailure(const LinkFailure& linkFailure);

    /**
     * Consults the shared memory for the newest events (traffic demands and e.g. link failures) that shall be
     * transferred to the simulation (to the source applications).
     */
    bool readAndInstallEvents();

    /**
     * Consults the shared memory for the newest actions that shall be transferred to the simulation.
     */
    bool readAndInstallActions();

    /**
     * Measures/monitors the network for metric such as load, queuing times, availability etc.
     * Places this information in the simulation object's member variable.
     * @return true if successful, false otherwise.
     */
    bool monitorNetwork();

    /**
     * Consults the shared memory for whether the done signal has been sent by the python side
     * (indicating that the episode is over and simulation should be terminated).
     * @return true if successful, false otherwise.
     */
    bool getDone();

    /**
     * Runs a single simulation step of the episode.
     */
    bool runSimStep();

    /**
     * Runs the main simulation loop of an episode. First sets up the network before iteratively
     * installing the actions delivered by the python side, executing a simulation step and
     * providing the newest network measurements back to the python side. Includes extensive anomaly
     * handling that also notify the python side if needed.
     * @return true if the episode terminated without issue, false otherwise.
     */
    bool runSimLoop();

  private:

    /**
     * Simulation parameters such as timesteps, packet size etc.
     */
    SimParameters m_simParameters;

    PointToPointHelper m_p2pHelper;
    Ipv4AddressHelper m_ipv4Helper;

    /**
     * Current ipv4 block for the simulation (used for setting up address spaces between nodes)
     */
    int m_ipv4Block;

    /**
     * Internet stack helper used throughout the simulation
     */
    InternetStackHelper m_inetHelper;

    /**
     * Traffic control helper (if desired) adds active queue management (incl. queueDiscs)
     * to the internet stack of each node
     */
    TrafficControlHelper m_tcHelper;

    PacketSinkHelper m_tcpSinkHelper;

    /**
     * Default port for all TCP sink applications.
     */
    const uint16_t m_tcpSinkPort = 1;

    PacketSinkHelper m_udpSinkHelper;

    /**
     * Default port for all UDP sink applications.
     */
    const uint16_t m_udpSinkPort = 2;

    /**
     * Holds all currently initialized simulation nodes
     */
    NodeContainer m_allNodes;

    /**
     * A map from node IDs to the device IDs that span the actual channel between these two nodes.
     * keys: (srcId, dstId); values: (srcDeviceIdOnNode, dstDeviceIdOnNode)
     */
    UPPMap m_deviceIdsPerEdge;

    Ptr<FlowMonitor> m_flowMonitor;

    FlowMonitorHelper m_flowHelper;

    /**
     * A vector of all monitoring edges in the network.
     */
    vector<Ptr<MonitoringDirEdge>> m_monitoredEdges;

    /**
     * A map from node-device ID pair to monitoring edge indexable by
     * sender's ID information.
     */
    unordered_map<U32Pair, Ptr<MonitoringDirEdge>, pair_hash> m_monitoredEdgesBySender;

    /**
     * A map from node-device ID pair to monitoring edge indexable by
     * receiver's ID information.
     */
    unordered_map<U32Pair, Ptr<MonitoringDirEdge>, pair_hash> m_monitoredEdgesByReceiver;

    UP32Map m_retransmittedPacketsPerPair;
    UP64Map m_retransmittedBytesPerPair;

    /**
     * can't be modeled as a matrix since NetDevice drops don't have src/dst information
     */
    uint32_t m_droppedPacketsPerReason[13];

    /**
     * can't be modeled as a matrix since NetDevice drops don't have src/dst information
     */
    uint64_t m_droppedBytesPerReason[13];

    /**
     * A map from IP addresses to node IDs, used for routing.
     */
    Ptr<OddLabelMap> m_routingLabelMap;

    /**
     * Timestamp of the last monitoring in seconds
     */
    double m_lastMonitoring = 0;
};

} // namespace ns3

#endif  // NS3_PACKERL_H