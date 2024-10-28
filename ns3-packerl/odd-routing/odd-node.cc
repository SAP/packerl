#include <algorithm>
#include <cstdint>

#include "ns3/ipv4-routing-table-entry.h"
#include "ns3/log.h"
#include "ns3/point-to-point-net-device.h"
#include "ns3/queue.h"
#include "ns3/application.h"

#include "odd-node.h"


namespace ns3
{

NS_LOG_COMPONENT_DEFINE("OddNode");
NS_OBJECT_ENSURE_REGISTERED (OddNode);

TypeId
OddNode::GetTypeId()
{
    static TypeId tid = TypeId("ns3::OddNode")
                            .AddConstructor<OddNode>()
                            .SetParent<Node>()
                            .AddTraceSource("NewOif",
                                            "Trace source indicating new oifs have been set in the node",
                                            MakeTraceSourceAccessor(&OddNode::m_newOifTrace),
                                            "ns3::TracedValueCallback::Void");
    return tid;
}

OddNode::OddNode() : Node()
{
    // reserve and make available ports 3-65535 (1 and 2 are reserved for sink apps, 0 denotes an invalid port)
    // We need a uint32_t counter to avoid overflow
    m_freePorts.reserve(65533);
    for (uint32_t port = 3; port <= 65535; ++port)
    {
        m_freePorts.push_back(static_cast<uint16_t>(port));
    }
}

Ipv4Address
OddNode::getIpv4Address(const uint32_t netDeviceId) const
{
    Ptr<Ipv4> ipv4 = this->GetObject<Ipv4>();
    if (ipv4 == nullptr)
    {
        NS_LOG_ERROR("IP stack is not installed on node " << this->m_nodeId);
    }
    if (netDeviceId >= this->GetNDevices())
    {
        NS_LOG_ERROR("Given netDeviceId is invalid (node has " << this->GetNDevices() << " devices, provided index: "
                                                               << this->m_nodeId << ")");
    }
    return ipv4->GetAddress(netDeviceId, 0).GetLocal();
}

Ipv4Address
OddNode::getDefaultIpv4Address() const
{
    return getIpv4Address(1);
}

std::vector<Ipv4Address>
OddNode::getNonLoopbackIpv4Addresses() const
{
    std::vector<Ipv4Address> addresses;
    NS_ASSERT_MSG(this->GetNDevices() > 0,
                  "OddNode doesn't have any connected NetDevices, not even a loopback");
    Ptr<Ipv4> ipv4 = this->GetObject<Ipv4>();
    if (ipv4 == nullptr)
    {
        NS_LOG_ERROR("IP stack is not installed on node " << this->m_nodeId);
    }
    for (size_t i = 1; i < this->GetNDevices(); ++i)
    {
        addresses.push_back(ipv4->GetAddress(i, 0).GetLocal());
    }
    return addresses;
}

double
OddNode::getRelativeDeviceQueueLoad(uint32_t deviceId) const
{
    Ptr<PointToPointNetDevice> senderDevice
        = DynamicCast<PointToPointNetDevice> (this->GetDevice(deviceId));
    if (senderDevice == nullptr)
    {
        NS_LOG_ERROR(this << " in getRelativeDeviceQueueLoad() (nodeID: )" << this->m_nodeId
                          << ", deviceID: " << deviceId << ": device does not have a queue!");
    }
    uint32_t curQueueSize = senderDevice->GetQueue()->GetCurrentSize().GetValue();
    uint32_t maxQueueSize = senderDevice->GetQueue()->GetMaxSize().GetValue();
    return static_cast<double>(curQueueSize) / maxQueueSize;
}

void OddNode::setOifPerDestination(const OifPerDestination& oipd)
{
    this->m_oifPerDestination = oipd;
    m_newOifTrace();
}

void OddNode::setOifDistPerDestination(const OifDistPerDestination& oidpd)
{
    this->m_oifDistPerDestination = oidpd;
    m_newOifTrace();
}

uint16_t
OddNode::reservePort()
{
    if (m_freePorts.empty())
    {
        NS_LOG_WARN("No free ports available on node " << m_nodeId << ", returning 0");
        return 0;
    }
    uint16_t port = m_freePorts.back();
    m_freePorts.pop_back();
    return port;
}

bool
OddNode::RemoveApplication(Ptr<Application> app, uint16_t port)
{
    NS_LOG_FUNCTION(this << app << port);
    if (port == 0)
    {
        NS_LOG_ERROR("Port 0 cannot be freed");
        return false;
    }
    else if (port > 65535)
    {
        NS_LOG_ERROR("Port " << port << " is out of range");
        return false;
    }
    auto it = std::find(m_applications.begin(), m_applications.end(), app);
    if (it == m_applications.end())
    {
        NS_LOG_ERROR("Application " << app << " not found on node " << m_nodeId);
        return false;
    }
    m_applications.erase(it);
    app->Dispose();
    app = nullptr;
    m_freePorts.push_back(port);
    return true;
}

} // namespace ns3