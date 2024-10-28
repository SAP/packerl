#ifndef NS3_DISPOSABLE_BULK_SEND_APPLICATION_H
#define NS3_DISPOSABLE_BULK_SEND_APPLICATION_H

#include "ns3/address.h"
#include "ns3/event-id.h"
#include "ns3/ptr.h"
#include "ns3/traced-callback.h"
#include "ns3/seq-ts-size-header.h"

#include "disposable-application.h"


namespace ns3
{

class Address;
class Socket;
class TcpHeader;
class TcpSocketBase;

/**
 * Like a BulkSendApplication, but with the ability to trace retransmissions and dispose when all data is sent.
 */
class DisposableBulkSendApplication : public DisposableApplication
{
  public:
    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    static TypeId GetTypeId();

    DisposableBulkSendApplication();

    ~DisposableBulkSendApplication() override;

    /**
     * \brief Set the upper bound for the total number of bytes to send.
     *
     * Once this bound is reached, no more application bytes are sent. If the
     * application is stopped during the simulation and restarted, the
     * total number of bytes sent is not reset; however, the maxBytes
     * bound is still effective and the application will continue sending
     * up to maxBytes. The value zero for maxBytes means that
     * there is no upper bound; i.e. data is sent until the application
     * or simulation is stopped.
     *
     * \param maxBytes the upper bound of bytes to send
     */
    void SetMaxBytes(uint64_t maxBytes);


  protected:
    void DoDispose() override;

  private:
    void StartApplication() override;
    void StopApplication() override;

    /**
     * \brief Send data until the L4 transmission buffer is full.
     * \param from From address
     * \param to To address
     */
    void SendData(const Address& from, const Address& to);

    bool m_connected;                    //!< True if connected
    uint8_t m_tos;                       //!< The packets Type of Service
    uint32_t m_sendSize;                 //!< Size of data to send each time
    uint64_t m_maxBytes;                 //!< Limit total number of bytes sent
    uint64_t m_totBytes;                 //!< Total bytes sent so far
    TypeId m_tid;                        //!< The type of protocol to use.
    uint32_t m_seq{0};                   //!< Sequence
    Ptr<Packet> m_unsentPacket;          //!< Variable to cache unsent packet
    bool m_enableSeqTsSizeHeader{false}; //!< Enable or disable the SeqTsSizeHeader

    /// Traced Callback: sent packets
    TracedCallback<Ptr<const Packet>> m_txTrace;

    /// Traced Callback: retransmitted packets
    TracedCallback<Ptr<const Packet>, const TcpHeader&, const Address&, const Address&, Ptr<const TcpSocketBase>>
        m_retransmissionTrace;

    /// Callback for tracing the packet Tx events, includes source, destination,  the packet sent,
    /// and header
    TracedCallback<Ptr<const Packet>, const Address&, const Address&, const SeqTsSizeHeader&>
        m_txTraceWithSeqTsSize;

  private:
    /**
     * \brief Connection Succeeded (called by Socket through a callback)
     * \param socket the connected socket
     */
    void ConnectionSucceeded(Ptr<Socket> socket);
    /**
     * \brief Connection Failed (called by Socket through a callback)
     * \param socket the connected socket
     */
    void ConnectionFailed(Ptr<Socket> socket);
    /**
     * \brief Send more data as soon as some has been transmitted.
     *
     * Used in socket's SetSendCallback - params are forced by it.
     *
     * \param socket socket to use
     * \param unused actually unused
     */
    void DataSend(Ptr<Socket> socket, uint32_t unused);

    /**
     *  \brief Packet Retransmitted (called by Socket through a callback if it's a TCPSocketBase)
     *  \param p the retransmitted packet
     *  \param header the TCP header
     *  \param socket the socket that retransmitted the packet
     */
    void PacketRetransmitted(Ptr<const Packet> p, const TcpHeader& header,
                             const Address& localAddr, const Address& peerAddr, Ptr<const TcpSocketBase> socket);
};

} // namespace ns3

#endif /* NS3_DISPOSABLE_BULK_SEND_APPLICATION_H */
