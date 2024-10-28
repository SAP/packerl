#ifndef NS3_DISPOSABLE_ONOFF_APPLICATION_H
#define NS3_DISPOSABLE_ONOFF_APPLICATION_H

#include "ns3/address.h"
#include "ns3/data-rate.h"
#include "ns3/event-id.h"
#include "ns3/ptr.h"
#include "ns3/seq-ts-size-header.h"
#include "ns3/traced-callback.h"

#include "disposable-application.h"


namespace ns3
{

class Address;
class RandomVariableStream;
class Socket;

/**
 * Like an OnOffApplication,
 * but with the ability to send smaller packets when there are less than m_pktSize bytes left.
 */
class DisposableOnOffApplication : public DisposableApplication
{
  public:
    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    static TypeId GetTypeId();

    DisposableOnOffApplication();

    ~DisposableOnOffApplication() override;

    /**
     * \brief Set the total number of bytes to send.
     *
     * Once these bytes are sent, no packet is sent again, even in on state.
     * The value zero means that there is no limit.
     *
     * \param maxBytes the total number of bytes to send
     */
    void SetMaxBytes(uint64_t maxBytes);

    /**
     * \brief Assign a fixed random variable stream number to the random variables
     * used by this model.
     *
     * \param stream first stream index to use
     * \return the number of stream indices assigned by this model
     */
    int64_t AssignStreams(int64_t stream);

  protected:
    void DoDispose() override;

  private:
    // inherited from Application base class.
    void StartApplication() override; // Called at time specified by Start
    void StopApplication() override;  // Called at time specified by Stop

    // helpers
    /**
     * \brief Cancel all pending events.
     */
    void CancelEvents();

    // Event handlers
    /**
     * \brief Start an On period
     */
    void StartSending();
    /**
     * \brief Start an Off period
     */
    void StopSending();
    /**
     * \brief Send a packet
     */
    void SendPacket();

    bool m_connected;                    //!< True if connected
    Ptr<RandomVariableStream> m_onTime;  //!< rng for On Time
    Ptr<RandomVariableStream> m_offTime; //!< rng for Off Time
    DataRate m_cbrRate;                  //!< Rate that data is generated
    DataRate m_cbrRateFailSafe;          //!< Rate that data is generated (check copy)
    uint32_t m_pktSize;                  //!< Size of packets
    uint32_t m_residualBits;             //!< Number of generated, but not sent, bits
    Time m_lastStartTime;                //!< Time last packet sent
    uint64_t m_maxBytes;                 //!< Limit total number of bytes sent
    uint64_t m_totBytes;                 //!< Total bytes sent so far
    EventId m_startStopEvent;            //!< Event id for next start or stop event
    EventId m_sendEvent;                 //!< Event id of pending "send packet" event
    TypeId m_tid;                        //!< Type of the socket used
    uint32_t m_seq{0};                   //!< Sequence
    Ptr<Packet> m_unsentPacket;          //!< Unsent packet cached for future attempt
    bool m_enableSeqTsSizeHeader{false}; //!< Enable or disable the use of SeqTsSizeHeader

    /// Traced Callback: transmitted packets.
    TracedCallback<Ptr<const Packet>> m_txTrace;

    /// Callbacks for tracing the packet Tx events, includes source and destination addresses
    TracedCallback<Ptr<const Packet>, const Address&, const Address&> m_txTraceWithAddresses;

    /// Callback for tracing the packet Tx events, includes source, destination, the packet sent,
    /// and header
    TracedCallback<Ptr<const Packet>, const Address&, const Address&, const SeqTsSizeHeader&>
        m_txTraceWithSeqTsSize;

  private:
    /**
     * \brief Schedule the next packet transmission
     */
    void ScheduleNextTx();
    /**
     * \brief Schedule the next On period start
     */
    void ScheduleStartEvent();
    /**
     * \brief Schedule the next Off period start
     */
    void ScheduleStopEvent();
    /**
     * \brief Handle a Connection Succeed event
     * \param socket the connected socket
     */
    void ConnectionSucceeded(Ptr<Socket> socket);
    /**
     * \brief Handle a Connection Failed event
     * \param socket the not connected socket
     */
    void ConnectionFailed(Ptr<Socket> socket);
};

} // namespace ns3

#endif /* NS3_DISPOSABLE_ONOFF_APPLICATION_H */
