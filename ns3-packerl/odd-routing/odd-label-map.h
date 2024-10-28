#ifndef NS3_ODD_LABEL_MAP_H
#define NS3_ODD_LABEL_MAP_H

#include <set>

#include "ns3/tag.h"
#include "ns3/object.h"
#include "ns3/simulation-singleton.h"
#include "ns3/ipv4-address.h"
#include "ns3/node-container.h"

#include "odd-node.h"


using std::unordered_map;

namespace ns3
{

/**
 * \brief The odd label map contains the mapping between assigned IPv4 addresses and node IDs.
 */
class OddLabelMap : public Object
{
  public:

    OddLabelMap() {};
    virtual ~OddLabelMap() override {};

    /**
     * \return the object TypeId
     */
    static TypeId GetTypeId();

    /**
     * \return the instance TypeId
     */
    virtual TypeId GetInstanceTypeId() const override;

    /**
     * Add a node to the label map by adding all its assigned IPv4 addresses as keys (the node ID is the value).
     * \param node the node to be added
     */
    void addNode(Ptr<OddNode> node);

    /**
     * Add all nodes contained in the given NodeContainer to the label map.
     * \param nodes the NodeContainer containing the nodes to be added
     */
    void registerNodes(const NodeContainer &nodes);

    /**
     * Remove a node from the label map by removing all its assigned IPv4 addresses as keys.
     * \param node the node to be removed
     */
    void removeNode(Ptr<OddNode> node);

    /**
     * \return the node ID for the given IPv4 address
     */
    uint32_t getNodeIdForIpv4Address(const Ipv4Address& addr) const;

    /**
     * \return the node IDs of all registered nodes
     */
    std::set<uint32_t> getRegisteredNodeIds() const
    {
        return m_registeredNodeIds;
    };

    /**
     * \return the internal mapping between IPv4 addresses and node IDs
     */
    unordered_map<Ipv4Address, uint32_t, Ipv4AddressHash> getNodeIdsPerAddress() const
    {
        return m_nodeIdsPerAddress;
    };

  private:
    /**
     * A set containing the node IDs for all registered nodes
     */
    std::set<uint32_t> m_registeredNodeIds;

    /**
     * A mapping between IPv4 addresses and node IDs
     */
    unordered_map<Ipv4Address, uint32_t, Ipv4AddressHash> m_nodeIdsPerAddress;
};

}

#endif  // NS3_ODD_LABEL_MAP_H