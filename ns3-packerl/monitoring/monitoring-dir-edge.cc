#include "monitoring-dir-edge.h"


namespace ns3
{
NS_LOG_COMPONENT_DEFINE("MonitoringDirEdge");

MonitoringDirEdge::MonitoringDirEdge(U32Pair src,
                                     U32Pair dst,
                                     uint32_t channelDataRate,
                                     uint32_t channelDelay,
                                     uint32_t txQueueCapacity,
                                     uint32_t queueDiscCapacity)
    : m_src(src),
      m_dst(dst),
      m_channelDataRate(channelDataRate),
      m_channelDelay(channelDelay),
      m_txQueueCapacity(txQueueCapacity),
      m_txQueueCurLoad(0),
      m_queueDiscCapacity(queueDiscCapacity),
      m_queueDiscCurLoad(0),
      m_isLinkUp(true)
{
    this->resetStats(-1.);
}

void
MonitoringDirEdge::resetStats(double simTime)
{
    NS_LOG_FUNCTION(this);
    this->m_txQueueMaxLoad = this->m_txQueueCurLoad;
    this->m_queueDiscMaxLoad = this->m_queueDiscCurLoad;
    this->m_sentPackets = 0;
    this->m_sentBytes = 0;
    this->m_receivedPackets = 0;
    this->m_receivedBytes = 0;
    this->m_droppedPackets = 0;
    this->m_droppedBytes = 0;
    this->m_lastMonitoringTime = simTime;
}

void
MonitoringDirEdge::registerLinkDown()
{
    NS_LOG_FUNCTION(this);
    this->m_isLinkUp = false;
}

void
MonitoringDirEdge::registerTxComplete(Ptr<const Packet> p)
{
    NS_LOG_FUNCTION(this << this->m_sentPackets);
    this->m_sentPackets++;
    this->m_sentBytes += p->GetSize();
}

void
MonitoringDirEdge::registerRxComplete(Ptr<const Packet> p)
{
    NS_LOG_FUNCTION(this << this->m_receivedPackets);
    this->m_receivedPackets++;
    this->m_receivedBytes += p->GetSize();
}

void
MonitoringDirEdge::registerDrop(Ptr<const Packet> p)
{
    NS_LOG_FUNCTION(this << this->m_droppedPackets);
    this->m_droppedPackets++;
    this->m_droppedBytes += p->GetSize();
}


void
MonitoringDirEdge::registerQDDrop(Ptr<const QueueDiscItem> item)
{
    NS_LOG_FUNCTION(this);
    registerDrop(item->GetPacket());
}

void
MonitoringDirEdge::registerTxEnqueue(Ptr<const Packet>)
{
    NS_LOG_FUNCTION(this << this->m_txQueueCurLoad << this->m_txQueueMaxLoad);
    this->m_txQueueCurLoad++;
    if (this->m_txQueueCurLoad > m_txQueueMaxLoad)
    {
        this->m_txQueueMaxLoad = m_txQueueCurLoad;
    }
}

void
MonitoringDirEdge::registerTxDequeue(Ptr<const Packet>)
{
    NS_LOG_FUNCTION(this << this->m_txQueueCurLoad << this->m_txQueueMaxLoad);
    NS_ASSERT_MSG(m_txQueueCurLoad > 0, "Queue is empty");
    this->m_txQueueCurLoad--;
}

void
MonitoringDirEdge::registerQDEnqueue(Ptr<const QueueDiscItem>)
{
    NS_LOG_FUNCTION(this << this->m_queueDiscCurLoad << this->m_queueDiscMaxLoad);
    this->m_queueDiscCurLoad++;
    if (this->m_queueDiscCurLoad > m_queueDiscMaxLoad)
    {
        this->m_queueDiscMaxLoad = m_queueDiscCurLoad;
    }
}

void
MonitoringDirEdge::registerQDDequeue(Ptr<const QueueDiscItem>)
{
    NS_LOG_FUNCTION(this << this->m_queueDiscCurLoad << this->m_queueDiscMaxLoad);
    NS_ASSERT_MSG(m_queueDiscCurLoad > 0, "Queue is empty");
    this->m_queueDiscCurLoad--;
}

}