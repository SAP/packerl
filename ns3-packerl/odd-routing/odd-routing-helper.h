#ifndef NS3_ODD_ROUTING_HELPER_H
#define NS3_ODD_ROUTING_HELPER_H

#include "ns3/ipv4-routing-helper.h"
#include "ns3/node-container.h"
#include "ns3/node.h"
#include "ns3/object-factory.h"

#include "odd-label-map.h"
#include "odd-routing.h"


namespace ns3
{

/**
 * \brief Helper class that adds OddRouting to nodes.
 */
class OddRoutingHelper : public Ipv4RoutingHelper
{
  public:
    OddRoutingHelper();

    /**
     * \internal
     * \returns pointer to clone of this OddRoutingHelper
     *
     * This method is mainly for internal use by the other helpers;
     * clients are expected to free the dynamic memory allocated by this method
     */
    OddRoutingHelper* Copy() const override;

    /**
     * \brief Construct an OddRoutingHelper from another previously
     * initialized instance (Copy Constructor).
     * \param o object to be copied
     */
    OddRoutingHelper(const OddRoutingHelper& o);

    /**
     * \param node the node on which the routing protocol will run
     * \returns a newly-created routing protocol
     *
     * This method will be called by ns3::InternetStackHelper::Install
     */
    Ptr<Ipv4RoutingProtocol> Create(Ptr<Node> node) const override;
    /**
     * \param name the name of the attribute to set
     * \param value the value of the attribute to set.
     *
     * This method controls the attributes of the OddRouting
     */
    void Set(std::string name, const AttributeValue& value);

    /**
     * Try and find the OddRouting as either the main routing
     * protocol or in the list of routing protocols associated with the
     * Ipv4 provided.
     *
     * \param ipv4 the Ptr<Ipv4> to search for the OddRouting
     * \returns OddRouting pointer or 0 if not found
     */
    Ptr<OddRouting> GetOddRouting(Ptr<Ipv4> ipv4) const;

    /**
     * @return the label map
     */
    Ptr<OddLabelMap> GetOddLabelMap() const {return m_labelMap;};


  private:
    /**
     * The factory to create an OddRouting object
     */
    ObjectFactory m_agentFactory;

    /**
     * @brief The map from IP address to node ID, see odd-label-map.h
     */
    Ptr<OddLabelMap> m_labelMap;
};

} // namespace ns3

#endif /* NS3_ODD_ROUTING_HELPER_H */
