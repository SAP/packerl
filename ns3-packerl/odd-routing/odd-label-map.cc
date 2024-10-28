#include "odd-label-map.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("OddLabelMap");
NS_OBJECT_ENSURE_REGISTERED(OddLabelMap);

TypeId
OddLabelMap::GetTypeId ()
{
    static TypeId tid = TypeId("ns3::OddLabelMap")
                            .SetParent<Object>()
                            .AddConstructor<OddLabelMap>();
    return tid;
}

TypeId
OddLabelMap::GetInstanceTypeId () const
{
    return OddLabelMap::GetTypeId ();
}

void
OddLabelMap::addNode(Ptr<OddNode> node)
{
    uint32_t nodeId = node->getNodeId();
    NS_LOG_FUNCTION(this << nodeId);
    m_registeredNodeIds.insert(nodeId);
    std::vector<Ipv4Address> ipv4Addresses = node->getNonLoopbackIpv4Addresses();
    for (Ipv4Address addr : ipv4Addresses)
    {
        NS_LOG_FUNCTION(this << nodeId << addr);
        m_nodeIdsPerAddress[addr] = nodeId;
    }
}

void
OddLabelMap::registerNodes(const NodeContainer &nodes)
{
    for (size_t i = 0; i < nodes.GetN(); ++i)
    {
        Ptr<OddNode> node = DynamicCast<OddNode>(nodes.Get(i));
        if (node)
        {
            addNode(node);
        }
    }
}

void
OddLabelMap::removeNode(Ptr<OddNode> node)
{
    uint32_t nodeId = node->getNodeId();
    if (m_registeredNodeIds.find(nodeId) == m_registeredNodeIds.end())
    {
        NS_LOG_INFO("node with id " << nodeId << " is not present in OddLabelMap");
        return;
    }
    unordered_map<Ipv4Address, uint32_t>::iterator iter = m_nodeIdsPerAddress.begin();
    unordered_map<Ipv4Address, uint32_t>::const_iterator endIter = m_nodeIdsPerAddress.end();

    for(; iter != endIter;)
    {
        if(iter->second == nodeId)
        {
            iter = m_nodeIdsPerAddress.erase(iter);
        }
        else
        {
            ++iter;
        }
    }
}

uint32_t
OddLabelMap::getNodeIdForIpv4Address(const Ipv4Address& addr) const
{
    return m_nodeIdsPerAddress.at(addr);
}

}