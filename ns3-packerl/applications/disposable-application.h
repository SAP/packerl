#ifndef NS3_PACKERL_DISPOSABLE_APPLICATION_H
#define NS3_PACKERL_DISPOSABLE_APPLICATION_H

#include "ns3/application.h"
#include "ns3/traced-callback.h"


namespace ns3
{

class Socket;

/**
 * Like an Application, but with an extra trace source for deletion.
 */
class DisposableApplication : public Application
{
  public:
    typedef void (*ReadyToDisposeCallback)(Ptr<Application> app);

    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    static TypeId GetTypeId();

    DisposableApplication();

    ~DisposableApplication() override;

    /**
     * \brief Get the socket this application is attached to.
     * \return pointer to associated socket
     */
    Ptr<Socket> GetSocket() const;

  protected:
    /// The callback to be called when the application is ready to be disposed
    TracedCallback<Ptr<DisposableApplication>> m_readyToDisposeTrace;

    Ptr<Socket> m_socket;                //!< Associated socket
    Address m_peer;                      //!< Peer address
    Address m_local;                     //!< Local address to bind to

};

} // namespace ns3

#endif /* NS3_PACKERL_DISPOSABLE_APPLICATION_H */
