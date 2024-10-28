#include "shared-structs.h"


const MonitoringDirEdgeSnapshot
makeMonitoringDirEdgeSnapshot(ns3::Ptr<MonitoringDirEdge> edge, double simTime)
{
    MonitoringDirEdgeSnapshot snapshot;
    snapshot.src = edge->srcNodeId();
    snapshot.dst = edge->dstNodeId();
    snapshot.isLinkUp = edge->isLinkUp();
    snapshot.channelDataRate = edge->channelDataRate();
    snapshot.channelDelay = edge->channelDelay();
    snapshot.txQueueCapacity = edge->txQueueCapacity();
    snapshot.txQueueMaxLoad = edge->txQueueMaxLoad();
    snapshot.txQueueLastLoad = edge->txQueueCurLoad();
    snapshot.queueDiscCapacity = edge->queueDiscCapacity();
    snapshot.queueDiscMaxLoad = edge->queueDiscMaxLoad();
    snapshot.queueDiscLastLoad = edge->queueDiscCurLoad();
    snapshot.sentPackets = edge->sentPackets();
    snapshot.sentBytes = edge->sentBytes();
    snapshot.receivedPackets = edge->receivedPackets();
    snapshot.receivedBytes = edge->receivedBytes();
    snapshot.droppedPackets = edge->droppedPackets();
    snapshot.droppedBytes = edge->droppedBytes();
    snapshot.elapsedTime = edge->lastMonitoringTime() >= 0. ? simTime - edge->lastMonitoringTime() : 0.;

    return snapshot;
}