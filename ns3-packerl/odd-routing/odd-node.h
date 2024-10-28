#ifndef NS3_ODD_NODE_H
#define NS3_ODD_NODE_H

#include "ns3/ipv4-address.h"
#include "ns3/ipv4.h"
#include "ns3/node.h"
#include "ns3/random-variable-stream.h"
#include "ns3/traced-callback.h"

using std::unordered_map, std::map, std::vector;

typedef unordered_map<uint32_t, uint32_t> OifPerDestination;
typedef unordered_map<uint32_t, map<uint32_t, double>> OifDistPerDestination;


namespace ns3
{

/**
 * \brief Subclass of the ns3 Node class that contains next-hop routing recommendations
 * to be used by the OddRouting protocol.
 */
class OddNode : public Node
{
  public:

    OddNode();
    virtual ~OddNode() override {};

    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    static TypeId GetTypeId();

    /*
     * \return the node's ID
     */
    unsigned int getNodeId() const {return m_nodeId;};

    /*
     * sets the node's ID
     * \param nodeId the node's ID
     */
    void setNodeId(unsigned int nodeId) {m_nodeId = nodeId;};

    /**
     * \return the IPv4 address of the net device with the given ID
     */
    Ipv4Address getIpv4Address(const uint32_t netDeviceId) const;

    /**
     * \return the default IPv4 address of the node, which is the address of the node's first net device
     */
    Ipv4Address getDefaultIpv4Address() const;

    /**
     * \return a vector containing all non-loopback IPv4 addresses of the node
     */
    std::vector<Ipv4Address> getNonLoopbackIpv4Addresses() const;

    /**
     * \return a double from 0.0 to 1.0 indicating the current load of the NetDevice with the given ID
     * \param deviceId the ID of the NetDevice
     */
    double getRelativeDeviceQueueLoad(uint32_t deviceId) const;

    /**
     * \return the map containing the current OifPerDestination routing recommendations
     */
    OifPerDestination getOifPerDestination() const {return this->m_oifPerDestination;};

    /**
     * set the current OifPerDestination routing recommendations
     * \param oipd An OifPerDestination map containing routing recommendations
     */
    void setOifPerDestination(const OifPerDestination& oipd);

    /**
     * \return the map containing the current OifDistPerDestination routing recommendations
     */
    OifDistPerDestination getOifDistPerDestination() const {return this->m_oifDistPerDestination;};

    /**
     * set the current OifDistPerDestination routing recommendations
     * \param oidpd An OifDistPerDestination map containing routing recommendations
     */
    void setOifDistPerDestination(const OifDistPerDestination& oidpd);

    /**
     * Reserve a free port number that can be used for socket connections.
     * \return the reserved port number
     */
    uint16_t reservePort();

    /**
     * Remove the given application from the node and free the port number.
     */
    bool RemoveApplication(Ptr<Application> app, uint16_t port);

  private:

    /**
     * The node ID. Since we assign these IDs ourselves (because we generate graphs), we can't use Node::m_id.
     */
    unsigned int m_nodeId;

    /**
     * A map containing, for each potential destination node (id),
     * the index of the outbound interface to be used for reaching that destination.
     */
    OifPerDestination m_oifPerDestination;

    /**
     * A map containing, for each potential destination node (id),
     * a distribution on all available outbound interface indices.
     */
    OifDistPerDestination m_oifDistPerDestination;

    /**
     * A trace, called if node has recently received new action suggestions,
     * and the aggregated routing protocol therefore needs to re-adjust its routes.
     */
    TracedCallback<> m_newOifTrace;

    /**
     * A vector containing all free port numbers that can be used for socket connections.
     */
    vector<uint16_t> m_freePorts;
};

} // namespace ns3

#endif // NS3_ODD_NODE_H
