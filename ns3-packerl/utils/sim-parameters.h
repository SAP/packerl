#ifndef NS3_PACKERL_SIM_PARAMETERS_H
#define NS3_PACKERL_SIM_PARAMETERS_H


/**
 * A custom struct holding ns3 simulation parameters,
 * used within the PackeRL framework for repeatable RL experiments.
 */
typedef struct SimParameters {
    /**
     * Memory block key, shared with the python side of the framework
     */
    uint32_t memblockKey;

    /**
     * Simulation time per step in ms
     */
    double simStepDuration;

    /**
     * Reference datarate for OSPF weights
     */
    uint32_t ospfwRefValue;

    /**
     * Effective packet size excl. IP/ICMP headers
     */
    uint32_t packetSize;

    /**
     * Whether to use flow control or not
     * (incl. queueDiscs as "softwareized" packet queues)
     */
    bool useFlowControl;

    /**
     * Whether to enable TCP SACKs or not
     */
    bool useTcpSack;

    /**
     * The probability of a flow being TCP (instead of UDP).
     */
    double probTcp;

    /**
     * Whether to use probabilistic routing or not
     */
    bool probabilisticRouting;
} SimParameters;

#endif // NS3_PACKERL_SIM_PARAMETERS_H
