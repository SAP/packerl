#ifndef NS3_ODD_TAG_H
#define NS3_ODD_TAG_H

#include "ns3/tag.h"

namespace ns3
{

/**
 * \brief A custom packet tag that provides the nodeId of the destination IP stored in the packet IP header.
 */
class OddTag : public Tag
{
  public:

    OddTag() : m_destinationNodeId(-1) {};
    virtual ~OddTag() override {};

    static TypeId GetTypeId(void);
    virtual TypeId GetInstanceTypeId(void) const override;

    /**
     * @return The serialized size of this packet tag
     */
    virtual uint32_t GetSerializedSize(void) const override;

    /**
     * Serialize the packet tag and its contents
     * @param i the buffer to write on to
     */
    virtual void Serialize (TagBuffer i) const override;

    /**
     * Deserialize a packet tag and its contents
     * @param i The buffer to read from
     */
    virtual void Deserialize (TagBuffer i) override;

    /**
     * Print a string representation of this packet tag
     * @param os out stram for printing
     */
    virtual void Print (std::ostream & os) const override;

    /**
     * @return the nodeId of the destination node
     */
    uint32_t getDestinationNodeId() const;

    /**
     * Set the node ID of the destination node
     * @param nodeId the node ID of the destination node
     */
    void setDestinationNodeId(uint32_t nodeId);

  private:
    /**
     * the nodeId of the destination node
     */
     uint32_t m_destinationNodeId;
};

}


#endif  // NS3_ODD_TAG_H