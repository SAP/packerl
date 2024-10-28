"""
Features are what we measure from the ns-3 simulator. 
They come in different value ranges:
 "unit" values range from 0 to 1,
 "small" values are not bounded but typically small (abs < 1),
 "large" values are not bounded and typically large (abs > 1).

The second entry of the tuple is whether the feature is a feature that indicates routing performance.
"""
ALL_GLOBAL_FEATURES = {
    # load
    "maxLU": ("unit", True),
    "avgTDU": ("unit", True),

    # sent/received/dropped/retransmitted
    "sentPackets":                                ("large", True),
    "receivedPackets":                            ("large", True),
    "droppedPackets":                             ("large", True),
    "retransmittedPackets":                       ("large", True),
    "sentBytes":                                  ("large", True),
    "receivedBytes":                              ("large", True),
    "droppedBytes":                               ("large", True),
    "retransmittedBytes":                         ("large", True),

    # dropped by reason
    "droppedBytes_Ipv4L3Protocol::DROP_NO_ROUTE":           ("large", True),
    "droppedBytes_Ipv4L3Protocol::DROP_TTL_EXPIRE":         ("large", True),
    "droppedBytes_Ipv4L3Protocol::DROP_BAD_CHECKSUM":       ("large", True),
    "droppedBytes_Ipv4L3Protocol::DROP_QUEUE":              ("large", True),
    "droppedBytes_Ipv4L3Protocol::DROP_QUEUE_DISC":         ("large", True),
    "droppedBytes_Ipv4L3Protocol::DROP_INTERFACE_DOWN":     ("large", True),
    "droppedBytes_Ipv4L3Protocol::DROP_ROUTE_ERROR":        ("large", True),
    "droppedBytes_Ipv4L3Protocol::DROP_FRAGMENT_TIMEOUT":   ("large", True),
    "droppedBytes_Ipv4L3Protocol::DROP_INVALID_REASON":     ("large", True),
    "droppedBytes_PointToPointNetDevice::MacTxDrop":        ("large", True),
    "droppedBytes_PointToPointNetDevice::PhyTxDrop":        ("large", True),
    "droppedBytes_PointToPointNetDevice::PhyRxDrop":        ("large", True),
    "droppedBytes_QueueDisc::Drop":                         ("large", True),
    "droppedPackets_Ipv4L3Protocol::DROP_NO_ROUTE":         ("large", True),
    "droppedPackets_Ipv4L3Protocol::DROP_TTL_EXPIRE":       ("large", True),
    "droppedPackets_Ipv4L3Protocol::DROP_BAD_CHECKSUM":     ("large", True),
    "droppedPackets_Ipv4L3Protocol::DROP_QUEUE":            ("large", True),
    "droppedPackets_Ipv4L3Protocol::DROP_QUEUE_DISC":       ("large", True),
    "droppedPackets_Ipv4L3Protocol::DROP_INTERFACE_DOWN":   ("large", True),
    "droppedPackets_Ipv4L3Protocol::DROP_ROUTE_ERROR":      ("large", True),
    "droppedPackets_Ipv4L3Protocol::DROP_FRAGMENT_TIMEOUT": ("large", True),
    "droppedPackets_Ipv4L3Protocol::DROP_INVALID_REASON":   ("large", True),
    "droppedPackets_PointToPointNetDevice::MacTxDrop":      ("large", True),
    "droppedPackets_PointToPointNetDevice::PhyTxDrop":      ("large", True),
    "droppedPackets_PointToPointNetDevice::PhyRxDrop":      ("large", True),
    "droppedPackets_QueueDisc::Drop":                       ("large", True),

    # delay/jitter
    "avgPacketDelay":                             ("small", True),
    "maxPacketDelay":                             ("small", True),
    "avgPacketJitter":                            ("small", True),

    # misc
    "elapsedTime":                                ("small", False),
    "sendableBytes":                              ("large", False),
}

ALL_NODE_FEATURES = {
    # sent/received/dropped/retransmitted
    "receivedBytes":                                ("large", True),
    "receivedPackets":                              ("large", True),
    "sentBytes":                                    ("large", True),
    "sentPackets":                                  ("large", True),
    "retransmittedBytes":                           ("large", True),
    "retransmittedPackets":                         ("large", True),
}

ALL_EDGE_FEATURES = {
    # config
    "txQueueCapacity":                              ("large", False),
    "queueDiscCapacity":                            ("large", False),
    "channelDelay":                                 ("large", False),
    "channelDataRate":                              ("large", False),

    # load
    "LU":                                           ( "unit", True),
    "txQueueLastLoad":                              ( "unit", True),
    "txQueueMaxLoad":                               ( "unit", True),
    "queueDiscLastLoad":                            ( "unit", True),
    "queueDiscMaxLoad":                             ( "unit", True),

    # sent/received/dropped (includes only what is measured in the incident NetDevices and channel!)
    "sentPackets":                                  ("large", True),
    "receivedPackets":                              ("large", True),
    "sentBytes":                                    ("large", True),
    "receivedBytes":                                ("large", True),
    "droppedBytes":                                 ("large", True),
    "droppedPackets":                               ("large", True),
}
