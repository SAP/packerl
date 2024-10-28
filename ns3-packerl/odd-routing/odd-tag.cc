#include "odd-tag.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("OddTag");
NS_OBJECT_ENSURE_REGISTERED(OddTag);

TypeId
OddTag::GetTypeId (void)
{
    static TypeId tid = TypeId ("ns3::OddTag")
                            .SetParent<Tag>()
                            .AddConstructor<OddTag>();
    return tid;
}

TypeId
OddTag::GetInstanceTypeId (void) const
{
    return OddTag::GetTypeId ();
}

uint32_t
OddTag::GetSerializedSize (void) const
{
    return sizeof (uint32_t);
}

/**
 * The order of how you do Serialize() should match the order of Deserialize()
 */
void
OddTag::Serialize (TagBuffer i) const
{
    i.WriteU32(m_destinationNodeId);
}

void
OddTag::Deserialize (TagBuffer i)
{
    m_destinationNodeId = i.ReadU32();
}

/**
 * This function can be used with ASCII traces if enabled.
 */
void
OddTag::Print (std::ostream &os) const
{
    os << "OddTag \t(destination node id: " << m_destinationNodeId  << ")";
}

uint32_t
OddTag::getDestinationNodeId() const
{
    return m_destinationNodeId;
}

void
OddTag::setDestinationNodeId(uint32_t nodeId)
{
    m_destinationNodeId = nodeId;
}

}