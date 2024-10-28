#ifndef NS3_ODD_ROUTING_H
#define NS3_ODD_ROUTING_H

#include "ns3/ipv4-address.h"
#include "ns3/ipv4-header.h"
#include "ns3/ipv4-interface.h"
#include "ns3/ipv4-l3-protocol.h"
#include "ns3/ipv4-routing-protocol.h"
#include "ns3/ipv4.h"
#include "ns3/node.h"
#include "ns3/output-stream-wrapper.h"
#include "ns3/ptr.h"
#include "ns3/random-variable-stream.h"

#include "odd-node.h"
#include "odd-label-map.h"

#include <list>
#include <stdint.h>

namespace ns3
{

class Packet;
class NetDevice;
class Ipv4Interface;
class Ipv4Address;
class Ipv4Header;
class Ipv4RoutingTableEntry;

using std::unordered_map, std::map;

/**
 * \brief Subclass of the ns3 Ipv4RoutingProtocol class that uses per-destination map entries
 *  (direct preferences or next-hop probability distributions) to create routing table entries on-demand.
 */
class OddRouting : public Ipv4RoutingProtocol
{
  public:
    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    static TypeId GetTypeId();

    OddRouting();
    ~OddRouting() override;

    /**
     * \return a bool indicating whether a random variable for ECMP routing has been set
     */
    bool HasRand() const;

    /**
     * (inherited from Ipv4RoutingProtocol)
     */
    Ptr<Ipv4Route> RouteOutput(Ptr<Packet> p,
                               const Ipv4Header& header,
                               Ptr<NetDevice> oif,
                               Socket::SocketErrno& sockerr) override;

    /**
     * (inherited from Ipv4RoutingProtocol, but invoked routing table entry creation on-demand
     * if no suitable routing rule is found)
     */
    bool RouteInput(Ptr<const Packet> p,
                    const Ipv4Header& header,
                    Ptr<const NetDevice> idev,
                    const UnicastForwardCallback& ucb,
                    const MulticastForwardCallback& mcb,
                    const LocalDeliverCallback& lcb,
                    const ErrorCallback& ecb) override;

    /**
     * \brief Lookup in the forwarding table for destination.
     * \param dest destination address
     * \param oif output interface if any (nullptr otherwise)
     * \return Ipv4Route to route the packet to reach dest address
     */
    Ptr<Ipv4Route> lookupOdd(Ipv4Address dest, Ptr<NetDevice> oif = nullptr);

    /**
     * \brief Create a routing table entry from the stored preferences on-demand
     * \param dest destination IP address
     * \return Ipv4RoutingTableEntry for dest
     */
    Ipv4RoutingTableEntry* CreateRTEntryOnDemand(Ipv4Address dest);

    /**
     * (inherited from Ipv4RoutingProtocol)
     */
    void NotifyInterfaceUp(uint32_t interface) override;

    /**
     * (inherited from Ipv4RoutingProtocol)
     */
    void NotifyInterfaceDown(uint32_t interface) override;

    /**
     * (inherited from Ipv4RoutingProtocol)
     */
    void NotifyAddAddress(uint32_t interface, Ipv4InterfaceAddress address) override;

    /**
     * (inherited from Ipv4RoutingProtocol)
     */
    void NotifyRemoveAddress(uint32_t interface, Ipv4InterfaceAddress address) override;

    /**
     * \brief in addition to setting the Ipv4 object, also sets corresponding OddNode.
     */
    void SetIpv4(Ptr<Ipv4> ipv4) override;

    /**
     * \brief Get the number of individual unicast routes that have been added
     * to the routing table.
     *
     * \warning The default route counts as one of the routes.
     * \returns the number of routes
     */
    uint32_t GetNRoutes() const;

    /**
     * (inherited from Ipv4RoutingProtocol)
     * \brief Get a route from the global unicast routing table.
     *
     * Externally, the unicast global routing table appears simply as a table with
     * n entries.  The one subtlety of note is that if a default route has been set
     * it will appear as the zeroth entry in the table.  This means that if you
     * add only a default route, the table will have one entry that can be accessed
     * either by explicitly calling GetDefaultRoute () or by calling GetRoute (0).
     *
     * Similarly, if the default route has been set, calling RemoveRoute (0) will
     * remove the default route.
     *
     * \param i The index (into the routing table) of the route to retrieve.  If
     * the default route has been set, it will occupy index zero.
     * \return If route is set, a pointer to that Ipv4RoutingTableEntry is returned, otherwise
     * a zero pointer is returned.
     *
     * \see Ipv4RoutingTableEntry
     * \see Ipv4GlobalRouting::RemoveRoute
     */
    Ipv4RoutingTableEntry* GetRoute(uint32_t i) const;

    /**
     * (inherited from Ipv4RoutingProtocol)
     */
    void PrintRoutingTable(Ptr<OutputStreamWrapper> stream,
                           Time::Unit unit = Time::S) const override;

    /**
     * set label map
     * \param labelMap the label map to be used for routing
     */
    void SetLabelMap(Ptr<OddLabelMap> labelMap) {m_routingLabelMap = labelMap;};

    /**
     * fill the provided destination map with the destinations per outgoing interface
     */
    void getDestinationsPerOif(unordered_map<uint32_t, std::set<uint32_t>> &destinationsPerOif) const;

    /**
     * \brief empties the routing table. Used on destruction or after the node's recommendations got updated.
     */
    void PurgeRoutingTable();

  protected:

    void DoDispose() override;

  private:

    /**
     * Set to true if packets are randomly routed among ECMP; set to false for using only one route consistently
     */
    bool m_randomEcmpRouting;

    /**
     * Set to true if this interface should respond to interface events by globally recomputing routes
     */
    bool m_respondToInterfaceEvents;

    /* A uniform random number generator for randomly routing packets among ECMP */
    Ptr<UniformRandomVariable> m_rand;

    /* IP protocol */
    Ptr<Ipv4> m_ipv4;

    /* Pointer to node that corresponds to m_ipv4 */
    Ptr<OddNode> m_node;

    /* The routing label map */
    Ptr<OddLabelMap> m_routingLabelMap;

    /**
     * If true, samples on demand from the node's action distributions.
     * If true and no distributions are provided, exceptions will be thrown.
     */
    bool m_sampleOnDemand;

    /* Container for the network routes */
    typedef std::list<Ipv4RoutingTableEntry*> Routes;

    /* Const Iterator for container for the network routes */
    typedef std::list<Ipv4RoutingTableEntry*>::const_iterator RoutesCI;

    /* Iterator for container for the network routes */
    typedef std::list<Ipv4RoutingTableEntry*>::iterator RoutesI;

    /* per-node network routes */
    Routes m_routes;
};

} // namespace ns3

#endif /* NS3_ODD_ROUTING_H */
