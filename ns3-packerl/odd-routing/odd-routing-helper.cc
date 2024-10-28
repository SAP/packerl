#include "ns3/ipv4-list-routing.h"
#include "ns3/names.h"
#include "ns3/node-list.h"
#include "ns3/ptr.h"

#include "odd-routing-helper.h"


namespace ns3
{

NS_LOG_COMPONENT_DEFINE("OddRoutingHelper");

OddRoutingHelper::OddRoutingHelper()
{
    NS_LOG_FUNCTION(this);
    TypeId oddRoutingTypeId = TypeId::LookupByName("ns3::OddRouting");
    m_agentFactory.SetTypeId(oddRoutingTypeId);
    m_labelMap = CreateObject<OddLabelMap>();
}

OddRoutingHelper::OddRoutingHelper(const OddRoutingHelper& o)
{
    this->m_agentFactory = o.m_agentFactory;
    this->m_labelMap = o.m_labelMap;
}

OddRoutingHelper*
OddRoutingHelper::Copy () const
{
  return new OddRoutingHelper(*this);
}

Ptr<Ipv4RoutingProtocol>
OddRoutingHelper::Create (Ptr<Node> node) const
{
  NS_LOG_FUNCTION(this << node->GetId());
  Ptr<OddNode> oddNode = DynamicCast<OddNode>(node);
  NS_ASSERT_MSG(oddNode, "given node is not an OddNode");

  Ptr<OddRouting> agent = m_agentFactory.Create<OddRouting> ();
  NS_ASSERT_MSG(agent->HasRand(), "no m_rand");
  NS_LOG_FUNCTION(this << "step 3");

  agent->SetLabelMap(m_labelMap);
  NS_LOG_FUNCTION(this << "step 4");

  node->AggregateObject (agent);
  return agent;
}

void
OddRoutingHelper::Set (std::string name, const AttributeValue &value)
{
  m_agentFactory.Set (name, value);
}

Ptr<OddRouting>
OddRoutingHelper::GetOddRouting(Ptr<Ipv4> ipv4) const
{
  NS_LOG_FUNCTION(this);
  Ptr<Ipv4RoutingProtocol> rp = ipv4->GetRoutingProtocol();
  NS_ASSERT_MSG(rp, "No routing protocol associated with Ipv4");
  if (DynamicCast<OddRouting>(rp))
  {
      NS_LOG_LOGIC("OddRouting found as the main IPv4 routing protocol.");
      return DynamicCast<OddRouting>(rp);
  }
  if (DynamicCast<Ipv4ListRouting>(rp))
  {
      Ptr<Ipv4ListRouting> lrp = DynamicCast<Ipv4ListRouting>(rp);
      int16_t priority;
      for (uint32_t i = 0; i < lrp->GetNRoutingProtocols(); i++)
      {
          NS_LOG_LOGIC("Searching for OddRouting in list");
          Ptr<Ipv4RoutingProtocol> temp = lrp->GetRoutingProtocol(i, priority);
          if (DynamicCast<OddRouting>(temp))
          {
              NS_LOG_LOGIC("Found OddRouting in list");
              return DynamicCast<OddRouting>(temp);
          }
      }
  }
  NS_LOG_LOGIC("OddRouting not found");
  return nullptr;
}

}
