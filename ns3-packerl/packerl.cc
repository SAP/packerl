#include <iostream>
#include <stdexcept>

#include "packerl.h"
#include "applications/disposable-bulk-send-application.h"
#include "applications/disposable-onoff-application.h"
#include "odd-routing/odd-node.h"
#include "odd-routing/odd-routing-helper.h"


using namespace std;

namespace ns3
{
NS_LOG_COMPONENT_DEFINE("PackerlEnv");

PackerlEnv::PackerlEnv(SimParameters simParameters)
    : Ns3AIRL<PackerlEnvStruct, PackerlActStruct>(simParameters.memblockKey),
      m_simParameters(simParameters),
      m_ipv4Block(10 * 255 * 255),
      m_tcpSinkHelper(PacketSinkHelper("ns3::TcpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), m_tcpSinkPort))),
      m_udpSinkHelper(PacketSinkHelper("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), m_udpSinkPort)))
{
    // Set global configuration
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue (m_simParameters.packetSize));
    Config::SetDefault("ns3::TcpSocketBase::Sack", BooleanValue(m_simParameters.useTcpSack));
    Config::SetDefault("ns3::DisposableBulkSendApplication::SendSize", UintegerValue(m_simParameters.packetSize));
    Config::SetDefault("ns3::DisposableOnOffApplication::PacketSize", UintegerValue (m_simParameters.packetSize));
    Config::SetDefault("ns3::DisposableOnOffApplication::OnTime", StringValue("ns3::ConstantRandomVariable[Constant=3600]"));  // TODO: make max_time
    Config::SetDefault("ns3::DisposableOnOffApplication::OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0.0]"));
    Config::SetDefault("ns3::OddRouting::RespondToInterfaceEvents", BooleanValue(false));
    Config::SetDefault("ns3::OddRouting::SampleOnDemand", BooleanValue(m_simParameters.probabilisticRouting));

    // Set up helpers and bookkeeping
    if (!m_simParameters.useFlowControl)
    {
        m_p2pHelper.DisableFlowControl();
    }
    m_inetHelper = InternetStackHelper();
    m_tcHelper = TrafficControlHelper();
    m_tcHelper.SetRootQueueDisc ("ns3::PfifoFastQueueDisc");
    OddRoutingHelper oddRoutingHelper;
    m_inetHelper.SetRoutingHelper(oddRoutingHelper);
    m_routingLabelMap = oddRoutingHelper.GetOddLabelMap();
    for (size_t i = 0; i < 13; ++i)
    {
        m_droppedPacketsPerReason[i] = 0;
        m_droppedBytesPerReason[i] = 0;
    }
}

bool
PackerlEnv::createNode(const uint32_t i)
{
    NS_LOG_FUNCTION(this << i);
    Ptr<OddNode> curNode = CreateObject<OddNode>();
    curNode->setNodeId(i);
    m_inetHelper.Install(curNode);  // also installs LoopbackNetDevice at device index 0

    // directly create sink applications (TCP, UDP) for that node. The app helpers ensure the app is added to the node
    Ptr<PacketSink> udpApp = DynamicCast<PacketSink>(m_udpSinkHelper.Install(curNode).Get(0));
    if (udpApp == nullptr)
    {
        NS_LOG_ERROR("could not cast UDP sink to PacketSink (shouldn't happen)");
        return false;
    }
    udpApp->SetStartTime (Seconds(0.0));
    udpApp->SetStopTime (Simulator::GetMaximumSimulationTime());

    Ptr<PacketSink> tcpApp = DynamicCast<PacketSink>(m_tcpSinkHelper.Install(curNode).Get(0));
    if (tcpApp == nullptr)
    {
        NS_LOG_ERROR("could not cast TCP sink to PacketSink (shouldn't happen)");
        return false;
    }
    tcpApp->SetStartTime (Seconds(0.0));
    tcpApp->SetStopTime (Simulator::GetMaximumSimulationTime());

    // register node
    m_allNodes.Add(curNode);
    m_flowMonitor = m_flowHelper.Install(curNode);
    return true;
}

bool
PackerlEnv::createEdge(const TopologyEdge& edge)
{
    NS_LOG_FUNCTION(this << edge.fst << edge.snd);
    // get/prepare current edge information
    stringstream ssdr;
    ssdr << edge.datarate << "bps";  // datarate values are given in bits per second (bps)
    string dataRateStr = ssdr.str();
    uint32_t maxPacketsInQueue = uint32_t((static_cast<uint64_t>(2 * edge.delay) * edge.datarate) / (1000 * m_simParameters.packetSize * 8));  // bandwidth-delay product (BDP)
    if (maxPacketsInQueue < 1)
    {
        maxPacketsInQueue = 1;
    }  // minimum buffer size
    stringstream ssmp;
    ssmp << maxPacketsInQueue << "p";
    string maxPacketsInTxQueueStr = ssmp.str();
    Ptr<Node> firstNode = m_allNodes.Get(edge.fst);
    Ptr<Node> secondNode = m_allNodes.Get(edge.snd);

    // install connection and thus get net devices.
    m_p2pHelper.SetDeviceAttribute("DataRate", StringValue(dataRateStr));
    m_p2pHelper.SetChannelAttribute("Delay", TimeValue(
            ns3::Time::FromInteger(edge.delay, ns3::Time::MS)));
    NetDeviceContainer curConnectionNetDevices = m_p2pHelper.Install(firstNode, secondNode);

    // create address space string and assign to new net devices
    // NOTE: there are address spaces for up to 245 * 255 * 255 connections.
    stringstream ss;
    ss << m_ipv4Block / (255 * 255)
       << "." << (m_ipv4Block / 255) % 255
       << "." << m_ipv4Block % 255
       << ".0";
    string ipv4AddressSpace = ss.str();
    m_ipv4Block++;

    m_ipv4Helper.SetBase(ipv4AddressSpace.c_str(), "255.255.255.0");
    Ipv4InterfaceContainer ipic = m_ipv4Helper.Assign(curConnectionNetDevices);

    // A lambda function to finish setup of created network devices
    auto finishNetDeviceSetup = [&](uint32_t nodeId, uint32_t devId, Ptr<MonitoringDirEdge> edgeWhereDevIsSender, Ptr<MonitoringDirEdge> edgeWhereDevIsReceiver)
    {
        // Preparation
        Ptr<PointToPointNetDevice> dev = DynamicCast<PointToPointNetDevice>(m_allNodes.Get(nodeId)->GetDevice(devId));
        if (dev == nullptr)
        {
            NS_LOG_ERROR("nullptr device in node/devId" << nodeId << ", " << devId << " (should not happen)");
        }

        // Set buffer limits for net device TxQueue according to bandwidth-delay product (BDP)
        dev->GetQueue()->SetMaxSize(QueueSize(QueueSizeUnit::PACKETS, maxPacketsInQueue));

        // Connect receive/send/enqueue/dequeue callbacks. Drop callbacks are doubly connected to register in matrix and edges
        dev->TraceConnectWithoutContext("MacRx", MakeCallback(&MonitoringDirEdge::registerRxComplete, &(*edgeWhereDevIsReceiver)));
        dev->TraceConnectWithoutContext("PhyTxEnd", MakeCallback(&MonitoringDirEdge::registerTxComplete, &(*edgeWhereDevIsSender)));
        dev->GetQueue()->TraceConnectWithoutContext("Enqueue", MakeCallback(&MonitoringDirEdge::registerTxEnqueue, &(*edgeWhereDevIsSender)));
        dev->GetQueue()->TraceConnectWithoutContext("Dequeue", MakeCallback(&MonitoringDirEdge::registerTxDequeue, &(*edgeWhereDevIsSender)));
        dev->TraceConnectWithoutContext("MacTxDrop", MakeCallback(&MonitoringDirEdge::registerDrop, &(*edgeWhereDevIsSender)));
        dev->TraceConnectWithoutContext("MacTxDrop", MakeCallback(&PackerlEnv::registerMacTxDrop, this));
        dev->TraceConnectWithoutContext("PhyTxDrop", MakeCallback(&MonitoringDirEdge::registerDrop, &(*edgeWhereDevIsSender)));
        dev->TraceConnectWithoutContext("PhyTxDrop", MakeCallback(&PackerlEnv::registerPhyTxDrop, this));
        dev->TraceConnectWithoutContext("PhyRxDrop", MakeCallback(&MonitoringDirEdge::registerDrop, &(*edgeWhereDevIsReceiver)));
        dev->TraceConnectWithoutContext("PhyRxDrop", MakeCallback(&PackerlEnv::registerPhyRxDrop, this));

        // if using flow control, adjust RootQueueDisc sizes per netDevice pair (also BDP) and connect traces
        if (m_simParameters.useFlowControl)
        {
            // TODO: SetRootQueueDisc() APPENDS a qdisc to its list, so we end up with multiple qdiscs
            //  on repeated calls -> how to SET params of our single qdisc without re-creating tcHelper every time?
            m_tcHelper.Uninstall(dev);
            m_tcHelper = TrafficControlHelper();
            m_tcHelper.SetRootQueueDisc("ns3::PfifoFastQueueDisc", "MaxSize", StringValue(maxPacketsInTxQueueStr));
            m_tcHelper.Install(dev);

            Ptr<QueueDisc> rootDisc = dev->GetNode()->GetObject<TrafficControlLayer>()->GetRootQueueDiscOnDevice(dev);
            rootDisc->TraceConnectWithoutContext("Enqueue", MakeCallback(&MonitoringDirEdge::registerQDEnqueue, &(*edgeWhereDevIsSender)));
            rootDisc->TraceConnectWithoutContext("Dequeue", MakeCallback(&MonitoringDirEdge::registerQDDequeue, &(*edgeWhereDevIsSender)));
            rootDisc->TraceConnectWithoutContext("Drop", MakeCallback(&MonitoringDirEdge::registerQDDrop, &(*edgeWhereDevIsSender)));
            rootDisc->TraceConnectWithoutContext("Drop", MakeCallback(&PackerlEnv::registerQueueDiscDrop, this));
        }
    };

    // newly created NetDevice sits at last idx position
    uint32_t fstDevId = curConnectionNetDevices.Get(0)->GetIfIndex();
    uint32_t sndDevId = curConnectionNetDevices.Get(1)->GetIfIndex();

    // Create a monitoredEdge from the new devices
    auto firstND = make_pair(edge.fst, fstDevId);
    auto secondND = make_pair(edge.snd, sndDevId);
    Ptr<MonitoringDirEdge> edge1 = Create<MonitoringDirEdge>(firstND, secondND, edge.datarate,
                                                             edge.delay, maxPacketsInQueue, maxPacketsInQueue);
    m_monitoredEdges.push_back(edge1);
    m_monitoredEdgesBySender[firstND] = edge1;
    m_monitoredEdgesByReceiver[secondND] = edge1;

    Ptr<MonitoringDirEdge> edge2 = Create<MonitoringDirEdge>(secondND, firstND, edge.datarate,
                                                             edge.delay, maxPacketsInQueue, maxPacketsInQueue);
    m_monitoredEdges.push_back(edge2);
    m_monitoredEdgesBySender[secondND] = edge2;
    m_monitoredEdgesByReceiver[firstND] = edge2;

    // Finish setup of network devices, incl. connecting traces for monitoring
    finishNetDeviceSetup(edge.fst, fstDevId, edge1, edge2);
    finishNetDeviceSetup(edge.snd, sndDevId, edge2, edge1);

    m_deviceIdsPerEdge.emplace(make_pair(edge.fst, edge.snd), make_pair(fstDevId, sndDevId));  // one direction
    m_deviceIdsPerEdge.emplace(make_pair(edge.snd, edge.fst), make_pair(sndDevId, fstDevId));  // other direction

    // We'll need to update the sink applications of the two incident nodes so that they are bound to the new devices
    // UDP sinks: first app of respective nodes
    Ptr<PacketSink> fstUdpSink = DynamicCast<PacketSink>(firstNode->GetApplication(0));
    if (fstUdpSink == nullptr)
    {
        NS_LOG_ERROR("could not cast UDP sink to PacketSink (shouldn't happen)");
        return false;
    }
    fstUdpSink->SetAttribute("Local", AddressValue(InetSocketAddress(Ipv4Address::GetAny(), m_udpSinkPort)));

    Ptr<PacketSink> sndUdpSink = DynamicCast<PacketSink>(secondNode->GetApplication(0));
    if (sndUdpSink == nullptr)
    {
        NS_LOG_ERROR("could not cast UDP sink to PacketSink (shouldn't happen)");
        return false;
    }
    sndUdpSink->SetAttribute("Local", AddressValue(InetSocketAddress(Ipv4Address::GetAny(), m_udpSinkPort)));

    // TCP sinks: second app of respective nodes
    Ptr<PacketSink> fstTcpSink = DynamicCast<PacketSink>(firstNode->GetApplication(1));
    if (fstTcpSink == nullptr)
    {
        NS_LOG_ERROR("could not cast TCP sink to PacketSink (shouldn't happen)");
        return false;
    }
    fstTcpSink->SetAttribute("Local", AddressValue(InetSocketAddress(Ipv4Address::GetAny(), m_tcpSinkPort)));

    Ptr<PacketSink> sndTcpSink = DynamicCast<PacketSink>(secondNode->GetApplication(1));
    if (sndTcpSink == nullptr)
    {
        NS_LOG_ERROR("could not cast TCP sink to PacketSink (shouldn't happen)");
        return false;
    }
    sndTcpSink->SetAttribute("Local", AddressValue(InetSocketAddress(Ipv4Address::GetAny(), m_tcpSinkPort)));

    // DONE
    NS_LOG_LOGIC("created edge: (dev: " << fstDevId << ", ip: " << ipic.GetAddress(0) << ") [" << edge.fst << " <-> "
                            << edge.snd << "] (devId=" << sndDevId << ", IP=" << ipic.GetAddress(1) << ") @ datarate(bps)="
                            << edge.datarate << ", delay(ms)=" << edge.delay << ", queuesize(p)=" << maxPacketsInQueue);
    return true;
}

bool
PackerlEnv::readAndInstallNetworkTopology()
{
    NS_LOG_FUNCTION(this);

    bool nodesReady = false;
    bool edgesReady = false;
    bool topologyReady = nodesReady && edgesReady;
    PackerlEnvStruct *envInfo;

    while (! topologyReady)
    {
        // get data
        envInfo = EnvGetterCond();
        bool topologyAvailableData = envInfo->topologyAvailable;
        bool allEdgesSentData = envInfo->allTopologyEdgesSent;
        uint32_t numNodesData = envInfo->numTopologyNodes;
        uint32_t numEdgesAvailableData = envInfo->numTopologyEdgesAvailable;  // needed because not all fields in edgeData might be initialized
        TopologyEdge *edgesData = envInfo->topologyEdges;

        // before yielding shm control, copy data to local array (avoid python side overwriting)
        bool topologyAvailable = topologyAvailableData;
        bool allEdgesSent = allEdgesSentData;
        uint32_t numNodes = numNodesData;
        uint32_t numEdgesAvailable = numEdgesAvailableData;
        std::vector<TopologyEdge> edges;
        edges.reserve(numEdgesAvailable);
        for (size_t i = 0; i < numEdgesAvailable; ++i)
        {
            edges[i] = edgesData[i];
        }
        GetCompleted();

        // set topologyAvailable to false, so that the python side can send more topology data if there is more
        envInfo = EnvSetterCond();
        envInfo->topologyAvailable = false;
        SetCompleted();

        // if obtained data is valid, use it to initialize the topology
        if (topologyAvailable)
        {
            if (! nodesReady)
            {
                // caution: if, later, there will be configuration for nodes, we can't set up all at once,
                // and we might have to first receive all data before creating edges
                m_allNodes = NodeContainer();
                for (uint32_t i = 0; i < numNodes; ++i)
                {
                    if (! createNode(i))
                    {
                        NS_LOG_ERROR("could not create node " << i);
                        return false;
                    }
                }
                NS_LOG_LOGIC("created nodes and installed internet stack on them");
                nodesReady = true;
            }

            if (! edgesReady)
            {
                for (size_t i = 0; i < numEdgesAvailable; ++i)
                {
                    if (! createEdge(edges[i]))
                    {
                        NS_LOG_ERROR("could not create edge " << i);
                        return false;
                    }
                }
                if (allEdgesSent)
                {
                    edgesReady = true;  // edges are ready once we've installed all new ones and no more will be sent
                    NS_LOG_LOGIC("created edges (connections between nodes)");
                }
            }
        }
        topologyReady = nodesReady && edgesReady;
    }

    NS_LOG_FUNCTION(this << "END");
    // print all entries of m_deviceIdsPerEdge (it's an unordered map from pair<uint32_t, uint32_t> to pair<uint32_t, uint32_t>)
    for (auto const& [edge, devIds] : m_deviceIdsPerEdge)
    {
        NS_LOG_LOGIC("device ids for edge " << edge.first << "->" << edge.second
                     << ": " << devIds.first << "->" << devIds.second);
    }

    // TODO: when nodes/edges are added/removed later, we will need to re-register them in the label map
    m_routingLabelMap->registerNodes(m_allNodes);
    return true;
}

bool
PackerlEnv::createApplicationFromDemand(const TrafficDemand& demand)
{
    // first, get nodes and source port
    Ptr<OddNode> srcNode = DynamicCast<OddNode>(m_allNodes.Get(demand.src));
    if (srcNode == nullptr)
    {
        NS_LOG_ERROR("could not cast source node to OddNode (shouldn't happen since they're all OddNodes)");
        return false;
    }
    Ptr<OddNode> dstNode = DynamicCast<OddNode>(m_allNodes.Get(demand.dst));
    if (dstNode == nullptr)
    {
        NS_LOG_ERROR("could not cast destination node to OddNode (shouldn't happen since they're all OddNodes)");
        return false;
    }

    uint16_t srcPort = srcNode->reservePort();
    NS_LOG_FUNCTION(this << demand.src << demand.dst << srcPort);
    if (srcPort == 0)  // for OddNodes, return value 0 signifies there are no free ports
    {
        NS_LOG_ERROR("could not reserve free port for source node " << demand.src);
        return false;
    }

    // create application according to demand type. Note: some app attributes are set in the class initializer list
    // We use custom variants of the applications that allow for disposal upon completion, for sending smaller
    // last packets in the case of UDP applications, and for registering retransmits in the case of TCP applications
    Ptr<DisposableApplication> app;
    uint16_t dstPort;
    if (demand.isTCP)  // TCP: DisposableBulkSendApplication
    {
        app = CreateObject<DisposableBulkSendApplication>();
        // TCP Retransmissions are not covered by FlowMonitor, so connect the trace manually for TCP applications
        app->TraceConnectWithoutContext("TcpRetransmission", MakeCallback(&PackerlEnv::registerRetransmit, this));
        // app->TraceConnect("ReadyToDispose", std::to_string(demand.src), MakeCallback(&PackerlEnv::removeApplication, this));
        dstPort = m_tcpSinkPort;
    }
    else  // UDP: DisposableOnOffApplication
    {
        app = CreateObject<DisposableOnOffApplication>();
        app->SetAttribute("DataRate", DataRateValue(DataRate(demand.datarate)));
        // app->TraceConnect("ReadyToDispose", std::to_string(demand.src), MakeCallback(&PackerlEnv::removeApplication, this));
        dstPort = m_udpSinkPort;
    }
    app->SetAttribute("MaxBytes", UintegerValue(demand.amount));

    // NOTE: getDefaultIpv4Address() is a function specific to OddNode returning the first non-loopback address
    AddressValue sourceAddress(InetSocketAddress(srcNode->getDefaultIpv4Address(), srcPort));
    app->SetAttribute("Local", sourceAddress);
    AddressValue remoteAddress(InetSocketAddress(dstNode->getDefaultIpv4Address(), dstPort));
    app->SetAttribute("Remote", remoteAddress);
    std::stringstream ss;
    ss << demand.src << "," << demand.dst;

    // Set start time relative to current simulation time
    double relativeTime = 0.001 * (demand.t - Simulator::Now().GetMilliSeconds());
    app->SetStartTime(Seconds(relativeTime));
    app->SetStopTime (Simulator::GetMaximumSimulationTime());

    // Finally, add the sender application to the source node
    srcNode->AddApplication(app);
    NS_LOG_LOGIC("created demand " << demand.src << " -> " << demand.dst << " at "
                              << demand.t << "ms (relative: " << relativeTime << "s): "
                              << demand.amount << " bytes, " << demand.datarate << " bps, "
                              << (demand.isTCP ? "TCP" : "UDP"));
    return true;
}

void
PackerlEnv::registerRetransmit(Ptr<const Packet> p, const TcpHeader& header, const Address& localAddr,
                               const Address& peerAddr, Ptr<const TcpSocketBase>)
{
    NS_LOG_FUNCTION(this);
    U32Pair nodePair = make_pair(m_routingLabelMap->getNodeIdForIpv4Address(Ipv4Address::ConvertFrom(localAddr)),
                                 m_routingLabelMap->getNodeIdForIpv4Address(Ipv4Address::ConvertFrom(peerAddr)));
    if (m_retransmittedPacketsPerPair.find(nodePair) == m_retransmittedPacketsPerPair.end())
    {
        m_retransmittedPacketsPerPair[nodePair] = 1;
    }
    else
    {
        m_retransmittedPacketsPerPair[nodePair]++;
    }
    if (m_retransmittedBytesPerPair.find(nodePair) == m_retransmittedBytesPerPair.end())
    {
        m_retransmittedBytesPerPair[nodePair] = p->GetSize();
    }
    else
    {
        m_retransmittedBytesPerPair[nodePair] += p->GetSize();
    }
}

void
PackerlEnv::registerMacTxDrop(Ptr<const Packet> p)
{
    NS_LOG_FUNCTION(this << p);
    m_droppedPacketsPerReason[9] += 1;
    m_droppedBytesPerReason[9] += p->GetSize();
}

void
PackerlEnv::registerPhyTxDrop(Ptr<const Packet> p)
{
    NS_LOG_FUNCTION(this << p);
    m_droppedPacketsPerReason[10] += 1;
    m_droppedBytesPerReason[10] += p->GetSize();
}

void
PackerlEnv::registerPhyRxDrop(Ptr<const Packet> p)
{
    NS_LOG_FUNCTION(this << p);
    m_droppedPacketsPerReason[11] += 1;
    m_droppedBytesPerReason[11] += p->GetSize();
}

void
PackerlEnv::registerQueueDiscDrop(Ptr<const Packet> p)
{
    NS_LOG_FUNCTION(this << p);
    m_droppedPacketsPerReason[12] += 1;
    m_droppedBytesPerReason[12] += p->GetSize();
}


void
PackerlEnv::removeApplication(string context, Ptr<DisposableApplication> app)
{
    NS_LOG_FUNCTION(this << context << app);
    NS_LOG_INFO("removeApplication");
    uint32_t nodeId = static_cast<uint32_t>(std::stoul(context));  // context is the node id
    Ptr<OddNode> curNode = DynamicCast<OddNode>(m_allNodes.Get(nodeId));
    if (curNode == nullptr)
    {
        NS_FATAL_ERROR("could not cast node to OddNode (shouldn't happen since they're all OddNodes)");
    }
    AddressValue appLocalAddress;
    app->GetAttribute("Local", appLocalAddress);
    uint16_t appLocalPort = InetSocketAddress::ConvertFrom(appLocalAddress.Get()).GetPort();
    if (! curNode->RemoveApplication(app, appLocalPort))
    {
        NS_FATAL_ERROR("could not remove application in node " << nodeId);
    }
    NS_LOG_FUNCTION(this << context << app << "END");
}

bool
PackerlEnv::executeLinkFailure(const LinkFailure& linkFailure)
{
    NS_LOG_FUNCTION(this << linkFailure.t << "s: " << linkFailure.fst << " -> " << linkFailure.snd);
    uint32_t fst = linkFailure.fst;
    uint32_t snd = linkFailure.snd;
    auto it = m_deviceIdsPerEdge.find(make_pair(fst, snd));
    if (it == m_deviceIdsPerEdge.end())
    {
        NS_LOG_ERROR("could not find device ids for edge " << fst << ", " << snd);
        return false;
    }
    auto [fstDevId, sndDevId] = it->second;
    Ptr<OddNode> fstNode = DynamicCast<OddNode>(m_allNodes.Get(fst));
    Ptr<OddNode> sndNode = DynamicCast<OddNode>(m_allNodes.Get(snd));
    if (fstNode == nullptr || sndNode == nullptr)
    {
        NS_LOG_ERROR("could not cast nodes to OddNode");
        return false;
    }
    Ptr<PointToPointNetDevice> fstDev = DynamicCast<PointToPointNetDevice>(fstNode->GetDevice(fstDevId));
    Ptr<PointToPointNetDevice> sndDev = DynamicCast<PointToPointNetDevice>(sndNode->GetDevice(sndDevId));
    if (fstDev == nullptr || sndDev == nullptr)
    {
        NS_LOG_ERROR("could not cast devices to PointToPointNetDevice");
        return false;
    }
    // we deactivate the entire link by setting the error model on incident network devices
    // to a rate error model with 100% error rate
    Ptr<RateErrorModel> em = CreateObject<RateErrorModel>();
    em->SetAttribute("ErrorRate", DoubleValue(1.0));
    fstDev->SetAttribute("ReceiveErrorModel", PointerValue(em));
    sndDev->SetAttribute("ReceiveErrorModel", PointerValue(em));

    // register link down in monitoring edges (since they're pointers, this will also update the other edge containers)
    m_monitoredEdgesBySender[make_pair(fst, fstDevId)]->registerLinkDown();
    m_monitoredEdgesBySender[make_pair(snd, sndDevId)]->registerLinkDown();
    return true;
}

bool
PackerlEnv::readAndInstallEvents() {
    NS_LOG_FUNCTION(this);

    bool allEventsRead = false;
    PackerlEnvStruct *envInfo;
    while (!allEventsRead) {
        // get data
        envInfo = EnvGetterCond();
        bool upcomingEventsAvailableData = envInfo->upcomingEventsAvailable;
        bool allUpcomingEventsSentData = envInfo->allUpcomingEventsSent;
        uint32_t numUpcomingEventsAvailableData = envInfo->numUpcomingEventsAvailable;
        Event *upcomingEventsData = envInfo->upcomingEvents;
        uint32_t *upcomingEventTypesData = envInfo->upcomingEventTypes;

        // before yielding shm control, copy data to local array (avoid python side overwriting)
        bool upcomingEventsAvailable = upcomingEventsAvailableData;
        bool allUpcomingEventsSent = allUpcomingEventsSentData;
        uint32_t numUpcomingEventsAvailable = numUpcomingEventsAvailableData;
        std::vector<Event> upcomingEvents;
        std::vector<uint32_t> upcomingEventTypes;
        upcomingEvents.reserve(numUpcomingEventsAvailable);
        upcomingEventTypes.reserve(numUpcomingEventsAvailable);
        for (size_t i = 0; i < numUpcomingEventsAvailable; ++i)
        {
            upcomingEvents[i] = upcomingEventsData[i];
            upcomingEventTypes[i] = upcomingEventTypesData[i];
        }
        GetCompleted();

        // set available to false, so that the python side can send more data if there is more
        envInfo = EnvSetterCond();
        envInfo->upcomingEventsAvailable = false;
        SetCompleted();

        // if obtained data is valid, use it to initialize the topology
        if (upcomingEventsAvailable) {
            for (size_t i = 0; i < numUpcomingEventsAvailable; ++i) {
                Event e = upcomingEvents[i];
                uint32_t eventType = upcomingEventTypes[i];
                if (eventType == 0)  // TrafficDemand
                {
                    TrafficDemand demand = e.trafficDemand;
                    if (! createApplicationFromDemand(demand))
                    {
                        NS_LOG_ERROR("could not create application from demand");
                        return false;
                    }
                }
                else if (eventType == 1)  // LinkFailure
                {
                    LinkFailure linkFailure = e.linkFailure;
                    double relativeTime = 0.001 * (linkFailure.t - Simulator::Now().GetMilliSeconds());
                    Simulator::Schedule(Seconds(relativeTime), &PackerlEnv::executeLinkFailure, this, linkFailure);
                }
                else
                {
                    NS_LOG_ERROR("Unsupported event type " << eventType);
                }
            }
            NS_LOG_LOGIC("Read and installed " << numUpcomingEventsAvailable << " events");
            if (allUpcomingEventsSent) {
                allEventsRead = true;  // events are ready once we've installed all new ones and no more will be sent
            }
        }
    }
    NS_LOG_FUNCTION(this << "END");
    return true;
}

bool
PackerlEnv::readAndInstallActions()
{
    NS_LOG_FUNCTION(this);

    OifDistMap oifDistPerDestinationPerNode;
    OifMap oifPerDestinationPerNode;
    PackerlActStruct *actInfo;

    bool allActionsReadAndInstalled = false;
    while (! allActionsReadAndInstalled) {

        // get data
        actInfo = ActionGetterCond();
        bool actionsAvailableData = actInfo->actionsAvailable;
        bool allActionsSentData = actInfo->allActionsSent;
        uint32_t numRoutingActionsAvailableData = actInfo->numRoutingActionsAvailable;  // needed because not all fields might be initialized
        RoutingActionComponent *routingActionsData = actInfo->routingActions;

        // before yielding shm control, copy data to local array (avoid python side overwriting)
        bool actionsAvailable = actionsAvailableData;
        bool allActionsSent = allActionsSentData;
        uint32_t numRoutingActionsAvailable = numRoutingActionsAvailableData;
        std::vector<RoutingActionComponent> routingActions;
        routingActions.reserve(numRoutingActionsAvailable);
        for (size_t i = 0; i < numRoutingActionsAvailable; ++i)
        {
            routingActions[i] = routingActionsData[i];
        }
        GetCompleted();

        // set available to false, so that the python side can send more data if there is more
        actInfo = ActionSetterCond();
        actInfo->actionsAvailable = false;
        SetCompleted();

        // if obtained data is valid, use it
        if (actionsAvailable) {
            for (size_t i = 0; i < numRoutingActionsAvailable; ++i) {
                RoutingActionComponent rac = routingActions[i];
                uint32_t edgeSrc = rac.edgeSrc;
                uint32_t edgeDst = rac.edgeDst;
                uint32_t demandDst = rac.demandDst;
                float value = rac.value;
                uint32_t nextHopDevId = m_deviceIdsPerEdge[make_pair(edgeSrc, edgeDst)].first;
                NS_LOG_LOGIC(this << " routing action: (" << edgeSrc << ", " << edgeDst
                                  << "), demandDst: " << demandDst
                                  << ", nextHopDevId: " << nextHopDevId
                                  <<  ", value: " << value);
                if (m_simParameters.probabilisticRouting)  // probabilistic routing -> assign probability
                {
                    oifDistPerDestinationPerNode[edgeSrc][demandDst][nextHopDevId] = value;
                }
                else if (value > 0.)  // deterministic routing and this edge is the maximum edge
                {
                    oifPerDestinationPerNode[edgeSrc][demandDst] = nextHopDevId;
                }
            }
            if (allActionsSent) {
                allActionsReadAndInstalled = true;  // ready once we've installed all new ones and no more will be sent
            }
        }
    }
    NS_LOG_LOGIC("read OddRoutingAction from shm");

    // install routing decisions in nodes
    if (m_simParameters.probabilisticRouting)
    {
        for (auto const& [v, dstMap] : oifDistPerDestinationPerNode)
        {
            for (auto const& [dst, nextHopDist] : dstMap)
            {
                for (auto const& [nextHopId, p] : nextHopDist)
                {
                    NS_LOG_LOGIC("routing @ " << v << " to " << dst << " via nextHopId " << nextHopId << ": p=" << p);
                }
            }
            Ptr<OddNode> curNode = DynamicCast<OddNode>(m_allNodes.Get(v));
            if (curNode == nullptr)
            {
                NS_LOG_ERROR("Node is not an OddNode: " << v);
                return false;
            }
            OifDistMap::const_iterator itOifDist = oifDistPerDestinationPerNode.find(v);
            if (itOifDist == oifDistPerDestinationPerNode.end())
            {
                NS_LOG_ERROR("could not find oifDist for OddNode " << v);
                return false;
            }
            curNode->setOifDistPerDestination(itOifDist->second);
        }
    }
    else
    {
        for (auto const& [v, dstMap] : oifPerDestinationPerNode)
        {
            for (auto const& [dst, nextHopId] : dstMap)
            {
                NS_LOG_LOGIC("routing @ " << v << " to " << dst << ": nextHopId = " << nextHopId);
            }
            Ptr<OddNode> curNode = DynamicCast<OddNode>(m_allNodes.Get(v));
            if (curNode == nullptr)
            {
                NS_LOG_ERROR("Node is not an OddNode: " << v);
                return false;
            }
            OifMap::const_iterator itOif = oifPerDestinationPerNode.find(v);
            if (itOif == oifPerDestinationPerNode.end())
            {
                NS_LOG_ERROR("could not find oif for OddNode " << v);
                return false;
            }
            curNode->setOifPerDestination(itOif->second);
        }
    }
    NS_LOG_FUNCTION(this << "END");
    return true;
}

bool
PackerlEnv::monitorNetwork()
{
    double monitoringTime = Simulator::Now().GetSeconds();
    NS_LOG_FUNCTION(this << monitoringTime);

    // obtain monitoring data from the flowMonitor, aggregating the stats per individual flow into a
    // traffic-matrix-like representation (does not yet reset the flowMonitor since we're using a const ref. to stats)
    UP32Map txPacketsPerNodePair, rxPacketsPerNodePair;
    UP64Map txBytesPerNodePair, rxBytesPerNodePair;
    double maxDelay = 0.;
    double delaySum = 0.;
    double avgJitterWeightedSum = 0.;
    uint64_t totalSentPackets = 0;
    uint64_t totalSentBytes = 0;
    uint64_t totalReceivedPackets = 0;
    uint64_t totalReceivedBytes = 0;
    uint32_t numFlows = 0;

    const FlowMonitor::FlowStatsContainer& flowMonitorStats = m_flowMonitor->GetFlowStats();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(m_flowHelper.GetClassifier());
    for (auto i = flowMonitorStats.begin(); i != flowMonitorStats.end(); ++i)
    {
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(i->first);
        uint32_t srcNode = m_routingLabelMap->getNodeIdForIpv4Address(t.sourceAddress);
        uint32_t dstNode = m_routingLabelMap->getNodeIdForIpv4Address(t.destinationAddress);
        U32Pair nodePair = make_pair(srcNode, dstNode);
        FlowMonitor::FlowStats stats = i->second;

        // sent packets
        if (txPacketsPerNodePair.find(nodePair) == txPacketsPerNodePair.end())
        {
            txPacketsPerNodePair[nodePair] = stats.txPackets;
        }
        else
        {
            txPacketsPerNodePair[nodePair] += stats.txPackets;
        }
        totalSentPackets += stats.txPackets;

        // sent bytes
        if (txBytesPerNodePair.find(nodePair) == txBytesPerNodePair.end())
        {
            txBytesPerNodePair[nodePair] = stats.txBytes;
        }
        else
        {
            txBytesPerNodePair[nodePair] += stats.txBytes;
        }
        totalSentBytes += stats.txBytes;

        // received packets (and delay/jitter values if there are any received packets)
        if (rxPacketsPerNodePair.find(nodePair) == rxPacketsPerNodePair.end())
        {
            rxPacketsPerNodePair[nodePair] = stats.rxPackets;
        }
        else
        {
            rxPacketsPerNodePair[nodePair] += stats.rxPackets;
        }
        if (stats.rxPackets > 0)
        {
            if (stats.maxDelay.GetSeconds() > maxDelay)
            {
                maxDelay = stats.maxDelay.GetSeconds();
            }
            delaySum += stats.delaySum.GetSeconds();
            double avgJitter = stats.rxPackets > 1 ? stats.jitterSum.GetSeconds() / (stats.rxPackets - 1) : 0.;
            avgJitterWeightedSum += avgJitter * stats.rxPackets;  // will be divided by totalReceivedPackets later
        }
        totalReceivedPackets += stats.rxPackets;

        // received bytes
        if (rxBytesPerNodePair.find(nodePair) == rxBytesPerNodePair.end())
        {
            rxBytesPerNodePair[nodePair] = stats.rxBytes;
        }
        else
        {
            rxBytesPerNodePair[nodePair] += stats.rxBytes;
        }
        totalReceivedBytes += stats.rxBytes;

        // dropped packets (monitoring dropReasons range from 0 to 8)
        NS_ASSERT_MSG(stats.packetsDropped.size() <= 9, "PackerlEnv currently expects up to 9 drop reasons)");
        for (uint32_t dropReason = 0; dropReason < stats.packetsDropped.size(); ++dropReason)
        {
            m_droppedPacketsPerReason[dropReason] += stats.packetsDropped[dropReason];
        }

        // dropped bytes (monitoring dropReasons range from 0 to 8)
        NS_ASSERT_MSG(stats.bytesDropped.size() <= 9, "PackerlEnv currently expects up to 9 drop reasons)");
        for (uint32_t dropReason = 0; dropReason < stats.bytesDropped.size(); ++dropReason)
        {
            // Log dropped bytes per reason (see ipv4-flow-probe.h for dropReasons)
            if (stats.bytesDropped[dropReason] > 0)
            {
                NS_LOG_LOGIC("Flow: " << srcNode << " -> " << dstNode << " dropReason: "
                             << dropReason << " bytes: " << stats.bytesDropped[dropReason]);
            }
            m_droppedBytesPerReason[dropReason] += stats.bytesDropped[dropReason];
        }

        numFlows++;
    }

    /*  THIS IS ONLY FOR DEBUG PURPOSES
    // (SEEING HOW MANY APPLICATIONS ARE RUNNING, TO CHECK WHETHER WE PROPERLY DELETE THEM)

    uint32_t numApplications = 0;
    uint32_t numConnectedApplications = 0;
    // get number of open sender applications from each node
    for (uint32_t i = 0; i < m_allNodes.GetN(); ++i)
    {
        Ptr<OddNode> node = DynamicCast<OddNode>(m_allNodes.Get(i));
        if (node == nullptr)
        {
            NS_LOG_ERROR("could not cast node to OddNode");
            return false;
        }
        numApplications += node->GetNApplications() - 2;  // subtract 2 since every node has two sink apps
    }
    NS_LOG_LOGIC(this << "(number of applications currently running: " << numApplications << ")");
    */

    // create snapshots from the obtained data
    MonitoringGlobalSnapshot monitoringGlobal;

    monitoringGlobal.numNodes = m_allNodes.GetN();
    monitoringGlobal.sentPackets = totalSentPackets;
    monitoringGlobal.sentBytes = totalSentBytes;
    monitoringGlobal.receivedPackets = totalReceivedPackets;
    monitoringGlobal.receivedBytes = totalReceivedBytes;
    for (size_t dropReason = 0; dropReason < 13; ++dropReason)
    {
        monitoringGlobal.droppedPacketsPerReason[dropReason] = m_droppedPacketsPerReason[dropReason];
        monitoringGlobal.droppedBytesPerReason[dropReason] = m_droppedBytesPerReason[dropReason];
    }

    monitoringGlobal.maxPacketDelay = maxDelay;
    monitoringGlobal.avgPacketDelay = totalReceivedPackets > 0 ? delaySum / totalReceivedPackets : 0.;
    monitoringGlobal.avgPacketJitter = totalReceivedPackets > 0 ? avgJitterWeightedSum / totalReceivedPackets : 0.;

    // sentPacketsEntries snapshot
    vector<TrafficPacketMatrixEntry> sentPacketsEntries;
    for (auto const& [nodePair, txPackets] : txPacketsPerNodePair)
    {
        TrafficPacketMatrixEntry tme;
        tme.src = nodePair.first;
        tme.dst = nodePair.second;
        tme.amount = txPackets;
        sentPacketsEntries.push_back(tme);
    }

    // sentBytesEntries snapshot
    vector<TrafficByteMatrixEntry> sentBytesEntries;
    for (auto const& [nodePair, txBytes] : txBytesPerNodePair)
    {
        TrafficByteMatrixEntry tme;
        tme.src = nodePair.first;
        tme.dst = nodePair.second;
        tme.amount = txBytes;
        sentBytesEntries.push_back(tme);
    }

    // receivedPacketsEntries snapshot
    vector<TrafficPacketMatrixEntry> receivedPacketsEntries;
    for (auto const& [nodePair, rxPackets] : rxPacketsPerNodePair)
    {
        TrafficPacketMatrixEntry tme;
        tme.src = nodePair.first;
        tme.dst = nodePair.second;
        tme.amount = rxPackets;
        receivedPacketsEntries.push_back(tme);
    }

    // receivedBytesEntries snapshot
    vector<TrafficByteMatrixEntry> receivedBytesEntries;
    for (auto const& [nodePair, rxBytes] : rxBytesPerNodePair)
    {
        TrafficByteMatrixEntry tme;
        tme.src = nodePair.first;
        tme.dst = nodePair.second;
        tme.amount = rxBytes;
        receivedBytesEntries.push_back(tme);
    }

    // retransmittedPacketsEntries snapshot
    vector<TrafficPacketMatrixEntry> retransmittedPacketEntries;
    uint64_t totalRetransmittedPackets = 0;
    for (auto const& [nodePair, retransmittedPackets] : m_retransmittedPacketsPerPair)
    {
        TrafficPacketMatrixEntry rpme;
        rpme.src = nodePair.first;
        rpme.dst = nodePair.second;
        rpme.amount = retransmittedPackets;
        retransmittedPacketEntries.push_back(rpme);
        totalRetransmittedPackets += retransmittedPackets;
    }
    monitoringGlobal.retransmittedPackets = totalRetransmittedPackets;

    // retransmittedBytesEntries snapshot
    vector<TrafficByteMatrixEntry> retransmittedByteEntries;
    uint64_t totalRetransmittedBytes = 0;
    for (auto const& [nodePair, retransmittedBytes] : m_retransmittedBytesPerPair)
    {
        TrafficByteMatrixEntry rbme;
        rbme.src = nodePair.first;
        rbme.dst = nodePair.second;
        rbme.amount = retransmittedBytes;
        retransmittedByteEntries.push_back(rbme);
        totalRetransmittedBytes += retransmittedBytes;
    }
    monitoringGlobal.retransmittedBytes = totalRetransmittedBytes;

    // monitored edges snapshot
    vector<MonitoringDirEdgeSnapshot> edgeSnapshots;
    for (Ptr<MonitoringDirEdge> edge : m_monitoredEdges)
    {
        edgeSnapshots.push_back(makeMonitoringDirEdgeSnapshot(edge, monitoringTime));
    }
    if (edgeSnapshots.size() == 0)
    {
        NS_LOG_ERROR("no edge snapshots available");
        return false;
    }
    monitoringGlobal.elapsedTime = edgeSnapshots[0].elapsedTime;

    // send the monitoring data to the python side
    bool monitoringGlobalSent = false;
    bool allSentPacketsEntriesSent = false;
    bool allSentBytesEntriesSent = false;
    bool allReceivedPacketsEntriesSent = false;
    bool allReceivedBytesEntriesSent = false;
    bool allRetransmittedPacketsEntriesSent = false;
    bool allRetransmittedBytesEntriesSent = false;
    bool allMonitoringEdgesSent = false;
    bool monitoringSent = monitoringGlobalSent
                          && allSentPacketsEntriesSent && allSentBytesEntriesSent
                          && allReceivedPacketsEntriesSent && allReceivedBytesEntriesSent
                          && allRetransmittedPacketsEntriesSent && allRetransmittedBytesEntriesSent
                          && allMonitoringEdgesSent;
    PackerlEnvStruct *envInfo;

    while (! monitoringSent)
    {
        envInfo = EnvSetterCond();
        if (! envInfo->monitoringAvailable)  // don't send if there is still sent data in the shared memory
        {
            // global monitoring data
            if (! monitoringGlobalSent)
            {
                NS_LOG_LOGIC("sending monitoringGlobal");
                envInfo->monitoringGlobal = monitoringGlobal;
                monitoringGlobalSent = true;
            }

            // sentPacketsEntries
            if (! allSentPacketsEntriesSent)
            {
                uint32_t numSentPacketsEntriesToBeSent;
                bool allSentPacketsEntriesSendable;
                if (sentPacketsEntries.size() <= MAX_ELEMS)
                {
                    numSentPacketsEntriesToBeSent = sentPacketsEntries.size();
                    allSentPacketsEntriesSendable = true;
                }
                else
                {
                    numSentPacketsEntriesToBeSent = MAX_ELEMS;
                    allSentPacketsEntriesSendable = false;
                }
                for (size_t i = 0; i < numSentPacketsEntriesToBeSent; ++i)
                {
                    TrafficPacketMatrixEntry sentPacketsEntry = sentPacketsEntries.back();
                    NS_LOG_LOGIC("sending sentPacketsEntry " << sentPacketsEntry.src << "->" << sentPacketsEntry.dst);
                    envInfo->sentPacketEntries[i] = sentPacketsEntry;
                    sentPacketsEntries.pop_back();
                }
                envInfo->numAvailableSentPacketEntries = numSentPacketsEntriesToBeSent;
                envInfo->allSentPacketEntriesSent = allSentPacketsEntriesSendable;
                if (allSentPacketsEntriesSendable)
                {
                    allSentPacketsEntriesSent = true;
                }
            }

            // sentBytesEntries
            if (! allSentBytesEntriesSent)
            {
                uint32_t numSentBytesEntriesToBeSent;
                bool allSentBytesEntriesSendable;
                if (sentBytesEntries.size() <= MAX_ELEMS)
                {
                    numSentBytesEntriesToBeSent = sentBytesEntries.size();
                    allSentBytesEntriesSendable = true;
                }
                else
                {
                    numSentBytesEntriesToBeSent = MAX_ELEMS;
                    allSentBytesEntriesSendable = false;
                }
                for (size_t i = 0; i < numSentBytesEntriesToBeSent; ++i)
                {
                    TrafficByteMatrixEntry sentBytesEntry = sentBytesEntries.back();
                    NS_LOG_LOGIC("sending sentBytesEntry " << sentBytesEntry.src << "->" << sentBytesEntry.dst);
                    envInfo->sentByteEntries[i] = sentBytesEntry;
                    sentBytesEntries.pop_back();
                }
                envInfo->numAvailableSentByteEntries = numSentBytesEntriesToBeSent;
                envInfo->allSentByteEntriesSent = allSentBytesEntriesSendable;
                if (allSentBytesEntriesSendable)
                {
                    allSentBytesEntriesSent = true;
                }
            }

            // receivedPacketsEntries
            if (! allReceivedPacketsEntriesSent)
            {
                uint32_t numReceivedPacketsEntriesToBeSent;
                bool allReceivedPacketsEntriesSendable;
                if (receivedPacketsEntries.size() <= MAX_ELEMS)
                {
                    numReceivedPacketsEntriesToBeSent = receivedPacketsEntries.size();
                    allReceivedPacketsEntriesSendable = true;
                }
                else
                {
                    numReceivedPacketsEntriesToBeSent = MAX_ELEMS;
                    allReceivedPacketsEntriesSendable = false;
                }
                for (size_t i = 0; i < numReceivedPacketsEntriesToBeSent; ++i)
                {
                    TrafficPacketMatrixEntry receivedPacketsEntry = receivedPacketsEntries.back();
                    NS_LOG_LOGIC("sending receivedPacketsEntry " << receivedPacketsEntry.src << "->" << receivedPacketsEntry.dst);
                    envInfo->receivedPacketEntries[i] = receivedPacketsEntry;
                    receivedPacketsEntries.pop_back();
                }
                envInfo->numAvailableReceivedPacketEntries = numReceivedPacketsEntriesToBeSent;
                envInfo->allReceivedPacketEntriesSent = allReceivedPacketsEntriesSendable;
                if (allReceivedPacketsEntriesSendable)
                {
                    allReceivedPacketsEntriesSent = true;
                }
            }

            // receivedBytesEntries
            if (! allReceivedBytesEntriesSent)
            {
                uint32_t numReceivedBytesEntriesToBeSent;
                bool allReceivedBytesEntriesSendable;
                if (receivedBytesEntries.size() <= MAX_ELEMS)
                {
                    numReceivedBytesEntriesToBeSent = receivedBytesEntries.size();
                    allReceivedBytesEntriesSendable = true;
                }
                else
                {
                    numReceivedBytesEntriesToBeSent = MAX_ELEMS;
                    allReceivedBytesEntriesSendable = false;
                }
                for (size_t i = 0; i < numReceivedBytesEntriesToBeSent; ++i)
                {
                    TrafficByteMatrixEntry receivedBytesEntry = receivedBytesEntries.back();
                    NS_LOG_LOGIC("sending receivedBytesEntry " << receivedBytesEntry.src << "->" << receivedBytesEntry.dst);
                    envInfo->receivedByteEntries[i] = receivedBytesEntry;
                    receivedBytesEntries.pop_back();
                }
                envInfo->numAvailableReceivedByteEntries = numReceivedBytesEntriesToBeSent;
                envInfo->allReceivedByteEntriesSent = allReceivedBytesEntriesSendable;
                if (allReceivedBytesEntriesSendable)
                {
                    allReceivedBytesEntriesSent = true;
                }
            }

            // retransmittedPacketsEntries
            if (! allRetransmittedPacketsEntriesSent)
            {
                uint32_t numRetransmittedPacketsEntriesToBeSent;
                bool allRetransmittedPacketsEntriesSendable;
                if (retransmittedPacketEntries.size() <= MAX_ELEMS)
                {
                    numRetransmittedPacketsEntriesToBeSent = retransmittedPacketEntries.size();
                    allRetransmittedPacketsEntriesSendable = true;
                }
                else
                {
                    numRetransmittedPacketsEntriesToBeSent = MAX_ELEMS;
                    allRetransmittedPacketsEntriesSendable = false;
                }
                for (size_t i = 0; i < numRetransmittedPacketsEntriesToBeSent; ++i)
                {
                    TrafficPacketMatrixEntry retransmittedPacketsEntry = retransmittedPacketEntries.back();
                    NS_LOG_LOGIC("sending retransmittedPacketsEntry " << retransmittedPacketsEntry.src << "->" << retransmittedPacketsEntry.dst);
                    envInfo->retransmittedPacketEntries[i] = retransmittedPacketsEntry;
                    retransmittedPacketEntries.pop_back();
                }
                envInfo->numAvailableRetransmittedPacketEntries = numRetransmittedPacketsEntriesToBeSent;
                envInfo->allRetransmittedPacketEntriesSent = allRetransmittedPacketsEntriesSendable;
                if (allRetransmittedPacketsEntriesSendable)
                {
                    allRetransmittedPacketsEntriesSent = true;
                }
            }

            // retransmittedBytesEntries
            if (! allRetransmittedBytesEntriesSent)
            {
                uint32_t numRetransmittedBytesEntriesToBeSent;
                bool allRetransmittedBytesEntriesSendable;
                if (retransmittedByteEntries.size() <= MAX_ELEMS)
                {
                    numRetransmittedBytesEntriesToBeSent = retransmittedByteEntries.size();
                    allRetransmittedBytesEntriesSendable = true;
                }
                else
                {
                    numRetransmittedBytesEntriesToBeSent = MAX_ELEMS;
                    allRetransmittedBytesEntriesSendable = false;
                }
                for (size_t i = 0; i < numRetransmittedBytesEntriesToBeSent; ++i)
                {
                    TrafficByteMatrixEntry retransmittedBytesEntry = retransmittedByteEntries.back();
                    NS_LOG_LOGIC("sending retransmittedBytesEntry " << retransmittedBytesEntry.src << "->" << retransmittedBytesEntry.dst);
                    envInfo->retransmittedByteEntries[i] = retransmittedBytesEntry;
                    retransmittedByteEntries.pop_back();
                }
                envInfo->numAvailableRetransmittedByteEntries = numRetransmittedBytesEntriesToBeSent;
                envInfo->allRetransmittedByteEntriesSent = allRetransmittedBytesEntriesSendable;
                if (allRetransmittedBytesEntriesSendable)
                {
                    allRetransmittedBytesEntriesSent = true;
                }
            }

            // monitoringDirEdges
            if (! allMonitoringEdgesSent)
            {
                uint32_t numEdgeSnapshotsToBeSent;
                bool allEdgeSnapshotsSendable;
                if (edgeSnapshots.size() <= MAX_ELEMS)
                {
                    numEdgeSnapshotsToBeSent = edgeSnapshots.size();
                    allEdgeSnapshotsSendable = true;
                }
                else
                {
                    numEdgeSnapshotsToBeSent = MAX_ELEMS;
                    allEdgeSnapshotsSendable = false;
                }
                for (size_t i = 0; i < numEdgeSnapshotsToBeSent; ++i)
                {
                    MonitoringDirEdgeSnapshot edgeSnapshot = edgeSnapshots.back();
                    NS_LOG_LOGIC("sending edgeSnapshot " << edgeSnapshot.src << "->" << edgeSnapshot.dst);
                    envInfo->monitoringDirEdges[i] = edgeSnapshot;
                    edgeSnapshots.pop_back();
                }
                envInfo->numAvailableMonitoringDirEdges = numEdgeSnapshotsToBeSent;
                envInfo->allMonitoringDirEdgesSent = allEdgeSnapshotsSendable;
                if (allEdgeSnapshotsSendable)
                {
                    allMonitoringEdgesSent = true;
                }
            }

            envInfo->monitoringAvailable = true;
        }
        SetCompleted();
        monitoringSent = monitoringGlobalSent
                         && allSentPacketsEntriesSent && allSentBytesEntriesSent
                         && allReceivedPacketsEntriesSent && allReceivedBytesEntriesSent
                         && allRetransmittedPacketsEntriesSent && allRetransmittedBytesEntriesSent
                         && allMonitoringEdgesSent;
    }

    // after sending, we wait for python side to signal that it has fully consumed the data, then yield shm control
    bool monitoringRead = false;
    while (! monitoringRead)
    {
        envInfo = EnvGetterCond();
        monitoringRead = ! envInfo->monitoringAvailable;
        GetCompleted();
    }

    // finally, reset the monitoring structures
    m_flowMonitor->ResetAllStats();
    for (auto& entry : m_retransmittedPacketsPerPair)
    {
        entry.second = 0;
    }
    for (auto& entry : m_retransmittedBytesPerPair)
    {
        entry.second = 0;
    }
    // set dropped array entries to 0
    for (size_t i = 0; i < 13; ++i)
    {
        m_droppedPacketsPerReason[i] = 0;
        m_droppedBytesPerReason[i] = 0;
    }
    for (Ptr<MonitoringDirEdge> edge : m_monitoredEdges)
    {
        edge->resetStats(monitoringTime);
    }

    NS_LOG_FUNCTION(this << monitoringTime << " END");
    return true;
}

bool
PackerlEnv::getDone()
{
    NS_LOG_FUNCTION(this);
    PackerlEnvStruct *envInfo = EnvGetterCond();
    bool done = envInfo->done;
    GetCompleted();
    NS_LOG_FUNCTION(this << "END");
    return done;
}

bool
PackerlEnv::runSimStep()
{
    NS_LOG_FUNCTION(this);
    try
    {
        NS_LOG_LOGIC("running sim step...");
        Simulator::Stop(MilliSeconds(m_simParameters.simStepDuration));
        Simulator::Run();
        NS_LOG_LOGIC("Simulation time is " << Simulator::Now().GetMilliSeconds() << "ms");
    }
    catch (const exception& e)
    {
        NS_LOG_ERROR(e.what());
        return false;
    }
    NS_LOG_FUNCTION(this << "END");
    return true;
}

bool
PackerlEnv::runSimLoop()
{
    NS_LOG_FUNCTION(this);

    // PREPARATION: network creation and initial monitoring
    if (! this->readAndInstallNetworkTopology()) { return false; }
    if (! this->monitorNetwork()) { return false; }

    // MAIN EPISODE LOOP. Termination due to episode length controlled by python side
    // with the 'done' variable; otherwise, only abnormal behaviour terminates this loop on the cpp side.
    bool done = false;
    while (!done)
    {
        if (! readAndInstallEvents()) { return false; }
        if (! readAndInstallActions()) { return false; }
        if (! runSimStep()) { return false; }
        if (! monitorNetwork()) { return false; }
        done = getDone();
        if (done) { SetFinish(); }
    }
    return true;
}
} // namespace ns3