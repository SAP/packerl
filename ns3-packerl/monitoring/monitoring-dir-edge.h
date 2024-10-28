#ifndef NS3_MONITORING_DIR_EDGE_H
#define NS3_MONITORING_DIR_EDGE_H

#include <stdint.h>
#include <vector>

#include "ns3/core-module.h"
#include "ns3/internet-module.h"

#include "../utils/types.h"


using namespace ns3;
using std::vector;

namespace ns3
{

/**
* A directed edge that monitors the network performance along a channel
 * spanned between two given nodes. Implements trace sinks that register sent,
 * received and dropped packets at the lower levels. The edge is directed because
 * we model only one sending direction; thus, for a channel with bidirectional traffic,
 * we need two such MonitoringDirEdges in both orientations.
 */
class MonitoringDirEdge : public SimpleRefCount<MonitoringDirEdge>
{
  public:
    /**
     * Constructor taking basic channel/edge configuration
     * @param src source IDs (node and device)
     * @param dst destination IDs (node and device)
     * @param channelDataRate channel data rate
     * @param channelDelay channel delay
     * @param txQueueCapacity capacity of transmission NetDevice queue in packets
     * @param queueDiscCapacity capacity of transmission queueDisc in packets
     *  (if applicable, i.e. usingTrafficControlLayer)
     */
    MonitoringDirEdge(U32Pair src,
                      U32Pair dst,
                      uint32_t channelDataRate,
                      uint32_t channelDelay,
                      uint32_t txQueueCapacity,
                      uint32_t queueDiscCapacity);

    /**
     * Resets the monitored statistics to default values,
     * except the 'last monitored' time which is updated.
     * @param simTime 'last monitored' time is set to this value
     */
    void resetStats(double simTime);

    /**
     * Trace sink for link down events
     */
    void registerLinkDown();

    /**
     * Trace sink for successful packet transmissions from source NetDevice.
     * @param p sent packet
     */
    void registerTxComplete(Ptr<const Packet> p);

    /**
     * Trace sink for successful packet receive at destination NetDevice
     * @param p received packet
     */
    void registerRxComplete(Ptr<const Packet> p);

    /**
     * Trace sink for dropped packet somewhere between source and dest. NetDevices (inclusive)
     * @param p dropped packets
     */
    void registerDrop(Ptr<const Packet> p);

    /**
     * Trace sink for dropped packet in source NetDevice's queue disc
     */
    void registerQDDrop(Ptr<const QueueDiscItem> item);

    /**
     * Trace sink for packet enqueued in source NetDevice's Tx queue
     */
    void registerTxEnqueue(Ptr<const Packet>);

    /**
     * Trace sink for packet dequeued from source NetDevice's Tx queue
     */
    void registerTxDequeue(Ptr<const Packet>);

    /**
     * Trace sink for packet enqueued in source node's queue discipline
     */
    void registerQDEnqueue(Ptr<const QueueDiscItem> item);

    /**
     * Trace sink for packet dequeued from source node's queue discipline
     */
    void registerQDDequeue(Ptr<const QueueDiscItem> item);

    /**
     * @return source nodeId
     */
    uint32_t srcNodeId() const
    {
        return m_src.first;
    };

    /**
     * @return destination nodeId
     */
    uint32_t dstNodeId() const
    {
        return m_dst.first;
    };

    /**
     * @return channel data rate
     */
    uint32_t channelDataRate() const
    {
        return m_channelDataRate;
    };

    /**
     * @return channel delay
     */
    uint32_t channelDelay() const
    {
        return m_channelDelay;
    }

    /**
     * @return source NetDevice's transmission queue capacity
     */
    uint32_t txQueueCapacity() const
    {
        return m_txQueueCapacity;
    };

    /**
     * @return source NetDevice's transmission queue current load
     */
    uint32_t txQueueCurLoad() const
    {
        return m_txQueueCurLoad;
    };

    /**
     * @return source NetDevice's transmission queue max. load
     */
    uint32_t txQueueMaxLoad() const
    {
        return m_txQueueMaxLoad;
    };

    /**
     * @return source node's queue disc. capacity
     */
    uint32_t queueDiscCapacity() const
    {
        return m_queueDiscCapacity;
    };

    /**
     * @return source node's queue disc. current load
     */
    uint32_t queueDiscCurLoad() const
    {
        return m_queueDiscCurLoad;
    };

    /**
     * @return source node's queue disc. max load
     */
    uint32_t queueDiscMaxLoad() const
    {
        return m_queueDiscMaxLoad;
    };

    /**
     * @return sent packet count
     */
    uint64_t sentPackets() const
    {
        return m_sentPackets;
    };

    /**
     * @return sent byte count
     */
    uint64_t sentBytes() const
    {
        return m_sentBytes;
    };

    /**
     * @return received packet count
     */
    uint64_t receivedPackets() const
    {
        return m_receivedPackets;
    };

    /**
     * @return received bytes count
     */
    uint64_t receivedBytes() const
    {
        return m_receivedBytes;
    };

    /**
     * @return dropped packets count
     */
    uint64_t droppedPackets() const
    {
        return m_droppedPackets;
    };

    /**
     * @return dropped bytes count
     */
    uint64_t droppedBytes() const
    {
        return m_droppedBytes;
    };

    /**
     * @return a double describing the last time this object has been monitored
     */
    double lastMonitoringTime() const
    {
        return m_lastMonitoringTime;
    };

    /**
     * @return true if the link is up, false otherwise
     */
    bool isLinkUp() const
    {
        return m_isLinkUp;
    }

  private:

    /**
     * source node and device ID
     */
    U32Pair m_src;

    /**
     * destination node and device ID
     */
    U32Pair m_dst;

    /**
     * channel data rate in bps
     */
    uint32_t m_channelDataRate;

    /**
     * channel delay in ms
     */
    uint32_t m_channelDelay;

    /**
     * source NetDevice's transmission queue capacity in packets
     */
    uint32_t m_txQueueCapacity;

    /**
     * source NetDevice's transmission queue max. load
     */
    uint32_t m_txQueueMaxLoad;

    /**
     * source NetDevice's transmission queue current load
     */
    uint32_t m_txQueueCurLoad;

    /**
     * source node's queue disc capacity in packets
     */
    uint32_t m_queueDiscCapacity;

    /**
     * source node's queue disc max load
     */
    uint32_t m_queueDiscMaxLoad;

    /**
     * source node's queue disc current load
     */
    uint32_t m_queueDiscCurLoad;

    /**
     * sent packets
     */
    uint64_t m_sentPackets;

    /**
     * sent bytes
     * (uint64 is enough for 213.5 days of continuous transmission with 1Tbps)
     */
    uint64_t m_sentBytes;

    /**
     * received packets
     */
    uint64_t m_receivedPackets;

    /**
     * received bytes
     * (uint64 is enough for 213.5 days of continuous transmission with 1Tbps)
     */
    uint64_t m_receivedBytes;

    /**
     * dropped packets
     */
    uint64_t m_droppedPackets;

    /**
     * dropped bytes
     * (uint64 is enough for 213.5 days of continuous transmission with 1Tbps
     */
    uint64_t m_droppedBytes;

    /**
     * a double describing the last time this object has been monitored
     */
    double m_lastMonitoringTime;

    /**
     * source NetDevice
     */
    bool m_isLinkUp;

};

}

#endif // NS3_MONITORING_DIR_EDGE_H