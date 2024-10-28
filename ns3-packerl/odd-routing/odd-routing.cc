#include "odd-routing.h"

#include "ns3/boolean.h"
#include "ns3/ipv4-route.h"
#include "ns3/ipv4-routing-table-entry.h"
#include "ns3/log.h"
#include "ns3/names.h"
#include "ns3/net-device.h"
#include "ns3/node.h"
#include "ns3/object.h"
#include "ns3/packet.h"
#include "ns3/simulator.h"

#include <iomanip>
#include <random>
#include <vector>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("OddRouting");
NS_OBJECT_ENSURE_REGISTERED (OddRouting);

TypeId
OddRouting::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::OddRouting")
            .SetParent<Ipv4RoutingProtocol>()
            .AddConstructor<OddRouting>()
            .AddAttribute("RandomEcmpRouting",
                          "Set to true if packets are randomly routed among ECMP; set to false for "
                          "using only one route consistently",
                          BooleanValue(false),
                          MakeBooleanAccessor(&OddRouting::m_randomEcmpRouting),
                          MakeBooleanChecker())
            .AddAttribute("RespondToInterfaceEvents",
                          "Set to true if you want to dynamically recompute the global routes upon "
                          "Interface notification events (up/down, or add/remove address)",
                          BooleanValue(false),
                          MakeBooleanAccessor(&OddRouting::m_respondToInterfaceEvents),
                          MakeBooleanChecker())
            .AddAttribute("SampleOnDemand",
                          "If true, samples on demand from the node's action distributions. "
                          "If true and no distributions are provided, exceptions will be thrown.",
                          BooleanValue(false),
                          MakeBooleanAccessor(&OddRouting::m_sampleOnDemand),
                          MakeBooleanChecker());
    return tid;
}

OddRouting::OddRouting()
    : m_randomEcmpRouting(false),
      m_respondToInterfaceEvents(false),
      m_ipv4(nullptr),
      m_sampleOnDemand(false)
{
    NS_LOG_FUNCTION(this);
    m_rand = CreateObject<UniformRandomVariable>();
    NS_ASSERT_MSG(m_rand, "no m_rand");
}

OddRouting::~OddRouting()
{
}

bool
OddRouting::HasRand() const
{
    return m_rand ? true : false;
}

Ptr<Ipv4Route>
OddRouting::RouteOutput(Ptr<Packet> p,
                        const Ipv4Header& header,
                        Ptr<NetDevice> oif,
                        Socket::SocketErrno& sockerr)
{
  NS_LOG_FUNCTION(this << p << header << header.GetSource() << header.GetDestination() << oif << sockerr);

  Ipv4Address destination = header.GetDestination();
  NS_ASSERT_MSG(!destination.IsMulticast(),
                "OddRouting doesn't yet support multicast");

  Ptr<Ipv4Route> rtentry = lookupOdd(destination, oif);
  if (rtentry)
  {
      sockerr = Socket::ERROR_NOTERROR;
  }
  else
  {
      sockerr = Socket::ERROR_NOROUTETOHOST;
  }
  return rtentry;
}

bool
OddRouting::RouteInput (Ptr<const Packet> p, const Ipv4Header &header,
                        Ptr<const NetDevice> idev, const UnicastForwardCallback& ucb,
                        const MulticastForwardCallback& mcb, const LocalDeliverCallback& lcb,
                        const ErrorCallback& ecb)
{
  NS_LOG_FUNCTION(this << p << header << header.GetSource() << header.GetDestination() << idev
                       << &lcb << &ecb);
  // Check if input device supports IP
  NS_ASSERT(m_ipv4->GetInterfaceForDevice(idev) >= 0);
  uint32_t iif = m_ipv4->GetInterfaceForDevice(idev);

  if (m_ipv4->IsDestinationAddress(header.GetDestination(), iif))
  {
      if (!lcb.IsNull())
      {
          NS_LOG_LOGIC("Local delivery to " << header.GetDestination());
          lcb(p, header, iif);
          return true;
      }
      else
      {
          // The local delivery callback is null.  This may be a multicast
          // or broadcast packet, so return false so that another
          // multicast routing protocol can handle it.  It should be possible
          // to extend this to explicitly check whether it is a unicast
          // packet, and invoke the error callback if so
          return false;
      }
  }

  // Check if input device supports IP forwarding
  if (m_ipv4->IsForwarding(iif) == false)
  {
      NS_LOG_LOGIC("Forwarding disabled for this interface");
      ecb(p, header, Socket::ERROR_NOROUTETOHOST);
      return true;
  }
  // Next, try to find a route
  NS_LOG_LOGIC("Unicast destination- looking up global route");
  Ptr<Ipv4Route> rtentry = lookupOdd(header.GetDestination());
  if (rtentry)
  {
      NS_LOG_LOGIC("Found unicast destination- calling unicast callback");
      ucb(rtentry, p, header);
      return true;
  }
  else
  {
      NS_LOG_LOGIC("Did not find unicast destination- returning false");
      return false; // Let other routing protocols try to handle this
                    // route request.
  }
}

Ptr<Ipv4Route>
OddRouting::lookupOdd(Ipv4Address dest, Ptr<NetDevice> oif)
{
  NS_LOG_FUNCTION(this << dest << oif);
  // NS_LOG_LOGIC("Looking for route for destination " << dest);
  Ptr<Ipv4Route> rtentry = nullptr;
  // store all available routes that bring packets to their destination
  std::vector<Ipv4RoutingTableEntry*> allRoutes;

  // NS_LOG_LOGIC("Number of m_routes = " << m_routes.size());
  for (RoutesCI i = m_routes.begin(); i != m_routes.end(); i++)
  {
      NS_ASSERT((*i)->IsHost());
      if ((*i)->GetDest() == dest)
      {
          if (oif)
          {
              if (oif != m_ipv4->GetNetDevice((*i)->GetInterface()))
              {
                  NS_LOG_LOGIC("Not on requested interface, skipping");
                  continue;
              }
          }
          allRoutes.push_back(*i);
          NS_LOG_LOGIC(allRoutes.size() << "Found global host route" << *i);
      }
  }
  Ipv4RoutingTableEntry* route;
  if (allRoutes.size() > 0) // if route(s) found
  {
      // pick up one of the routes uniformly at random if random
      // ECMP routing is enabled, else select the first route
      uint32_t selectIndex =
          m_randomEcmpRouting ? m_rand->GetInteger(0, allRoutes.size() - 1)
                              : 0;
      route = allRoutes.at(selectIndex);
  }
  else
  {
      // create a route using on-demand lookup from the OddNode.
      route = CreateRTEntryOnDemand(dest);
  }

  if (route)
  {
      // create a Ipv4Route object from the selected routing table entry
      rtentry = Create<Ipv4Route>();
      rtentry->SetDestination(route->GetDest());
      rtentry->SetSource(m_ipv4->GetAddress(route->GetInterface(), 0).GetLocal());
      rtentry->SetGateway(route->GetGateway());
      uint32_t interfaceIdx = route->GetInterface();
      rtentry->SetOutputDevice(m_ipv4->GetNetDevice(interfaceIdx));
      return rtentry;
  }
  else
  {
      return nullptr;
  }
}

Ipv4RoutingTableEntry*
OddRouting::CreateRTEntryOnDemand(Ipv4Address dest)
{
  // NS_LOG_FUNCTION(this << dest);
  NS_ASSERT_MSG(m_routingLabelMap, "OddRouting does not hold OddLabelMap");
  uint32_t destNodeId = m_routingLabelMap->getNodeIdForIpv4Address(dest);
  uint32_t selectedInterface;
  // NS_LOG_FUNCTION(this << dest << destNodeId);
  if (m_sampleOnDemand)  // select correct entry from given action distributions and sample
  {
      OifDistPerDestination oisds = m_node->getOifDistPerDestination();
      NS_ASSERT_MSG(!oisds.empty(), "Distribution Table for OddNode is empty");
      Ptr<EmpiricalRandomVariable> erv = CreateObject<EmpiricalRandomVariable>();
      erv->SetInterpolate(false);
      erv->SetAntithetic(false);

      map<uint32_t, double> curDist = oisds.at(destNodeId);  // we need an ordered map here since we're using CDF
      double cumulativeProb = 0.0;
      for (auto const& [idx, prob] : curDist)
      {
          cumulativeProb += prob;
          if (idx == curDist.rbegin()->first)  // for last entry, set cumulativeProb to 1.0
          {
              cumulativeProb = 1.0;
          }
          erv->CDF((double) idx, cumulativeProb);
      }
      selectedInterface = erv->GetInteger();
  }
  else  // use the given discrete recommendations
  {
      selectedInterface = m_node->getOifPerDestination().at(destNodeId);
  }

  // assemble route and return it
  Ipv4RoutingTableEntry* route = new Ipv4RoutingTableEntry();
  *route = Ipv4RoutingTableEntry::CreateHostRouteTo(dest, selectedInterface);
  m_routes.push_back(route);
  return route;
}

void
OddRouting::NotifyInterfaceUp (uint32_t i)
{
  NS_LOG_FUNCTION(this << i);
  NS_ASSERT_MSG(!m_respondToInterfaceEvents,
                "OddRouting doesn't yet work with RespondToInterfaceEvents");
}

void
OddRouting::NotifyInterfaceDown (uint32_t i)
{
  NS_LOG_FUNCTION(this << i);
  NS_ASSERT_MSG(!m_respondToInterfaceEvents,
                "OddRouting doesn't yet work with RespondToInterfaceEvents");
}

void
OddRouting::NotifyAddAddress (uint32_t i, Ipv4InterfaceAddress address)
{
  NS_LOG_FUNCTION(this << i);
  NS_ASSERT_MSG(!m_respondToInterfaceEvents,
                "OddRouting doesn't yet work with RespondToInterfaceEvents");
}

void
OddRouting::NotifyRemoveAddress (uint32_t i, Ipv4InterfaceAddress address)
{
  NS_LOG_FUNCTION(this << i);
  NS_ASSERT_MSG(!m_respondToInterfaceEvents,
                "OddRouting doesn't yet work with RespondToInterfaceEvents");
}

void
OddRouting::SetIpv4(Ptr<Ipv4> ipv4)
{
  NS_LOG_FUNCTION(this << ipv4);
  NS_ASSERT(!m_ipv4 && ipv4);
  m_ipv4 = ipv4;

  Ptr<OddNode> node = DynamicCast<OddNode>(ipv4->GetObject<Node>());
  NS_ASSERT(!m_node && node);
  m_node = node;
  m_node->TraceConnectWithoutContext("NewOif",
                                     MakeCallback(&OddRouting::PurgeRoutingTable, this));
}

uint32_t
OddRouting::GetNRoutes() const
{
  NS_LOG_FUNCTION(this);
  uint32_t n = 0;
  n += m_routes.size();
  return n;
}

Ipv4RoutingTableEntry*
OddRouting::GetRoute(uint32_t index) const
{
  NS_LOG_FUNCTION(this << index);
  NS_ASSERT_MSG(index >= 0 && index < m_routes.size(), "Invalid index for GetRoute()");
  uint32_t tmp = 0;
  for (RoutesCI i = m_routes.begin(); i != m_routes.end(); i++)
  {
      if (tmp == index)
      {
          return *i;
      }
      tmp++;
  }
  NS_ASSERT(false);
  // quiet compiler.
  return nullptr;
}

// Formatted like output of "route -n" command
void
OddRouting::PrintRoutingTable(Ptr<OutputStreamWrapper> stream, Time::Unit unit) const
{
  NS_LOG_FUNCTION(this << stream);
  std::ostream* os = stream->GetStream();
  // Copy the current ostream state
  std::ios oldState(nullptr);
  oldState.copyfmt(*os);

  *os << std::resetiosflags(std::ios::adjustfield) << std::setiosflags(std::ios::left);

  *os << "Node: " << m_node->GetId() << ", Time: " << Now().As(unit)
      << ", Local time: " << m_node->GetLocalTime().As(unit)
      << ", OddRouting table" << std::endl;

  if (GetNRoutes() > 0)
  {
      *os << "Destination           Gateway         Genmask         Flags Metric Ref    Use Iface"
          << std::endl;
      for (uint32_t j = 0; j < GetNRoutes(); j++)
      {
          std::ostringstream dest;
          std::ostringstream gw;
          std::ostringstream mask;
          std::ostringstream flags;
          Ipv4RoutingTableEntry route = GetRoute(j);
          Ipv4Address destAddr = route.GetDest();
          dest << destAddr << " (node " << m_routingLabelMap->getNodeIdForIpv4Address(destAddr) << ")";
          *os << std::setw(26) << dest.str();
          gw << route.GetGateway();
          *os << std::setw(16) << gw.str();
          mask << route.GetDestNetworkMask();
          *os << std::setw(16) << mask.str();
          flags << "U";
          if (route.IsHost())
          {
              flags << "H";
          }
          else if (route.IsGateway())
          {
              flags << "G";
          }
          *os << std::setw(6) << flags.str();
          // Metric not implemented
          *os << "-"
              << "      ";
          // Ref ct not implemented
          *os << "-"
              << "      ";
          // Use not implemented
          *os << "-"
              << "   ";
          if (Names::FindName(m_ipv4->GetNetDevice(route.GetInterface())) != "")
          {
              *os << Names::FindName(m_ipv4->GetNetDevice(route.GetInterface()));
          }
          else
          {
              *os << route.GetInterface();
          }
          *os << std::endl;
      }
  }
  *os << std::endl;
  // Restore the previous ostream state
  (*os).copyfmt(oldState);
}

void
OddRouting::PurgeRoutingTable()
{
  NS_LOG_FUNCTION(this);
  for (RoutesI j = m_routes.begin(); j != m_routes.end();
       j = m_routes.erase(j))
  {
      delete (*j);
  }
}

void
OddRouting::getDestinationsPerOif(std::unordered_map<uint32_t, std::set<uint32_t>>& destinationsPerOif) const
{
  for (size_t i = 0; i < GetNRoutes(); ++i)
  {
      Ipv4RoutingTableEntry route = GetRoute(i);
      uint32_t oif = route.GetInterface();
      uint32_t destNodeId = m_routingLabelMap->getNodeIdForIpv4Address(route.GetDest());
      destinationsPerOif[oif].insert(destNodeId);
  }
}

void
OddRouting::DoDispose()
{
  NS_LOG_FUNCTION(this);
  PurgeRoutingTable();
  m_node = nullptr;
  m_ipv4 = nullptr;
  Ipv4RoutingProtocol::DoDispose();
}

}  // namespace ns3
