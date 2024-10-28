#include "ns3/socket.h"

#include "disposable-application.h"


namespace ns3
{

NS_LOG_COMPONENT_DEFINE("DisposableApplication");

NS_OBJECT_ENSURE_REGISTERED(DisposableApplication);

TypeId
DisposableApplication::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::DisposableApplication")
            .SetParent<Application>()
            .SetGroupName("Applications")
            .AddConstructor<DisposableApplication>()
            .AddAttribute("Remote",
                          "The address of the destination",
                          AddressValue(),
                          MakeAddressAccessor(&DisposableApplication::m_peer),
                          MakeAddressChecker())
            .AddAttribute("Local",
                          "The Address on which to bind the socket. If not set, it is generated "
                          "automatically.",
                          AddressValue(),
                          MakeAddressAccessor(&DisposableApplication::m_local),
                          MakeAddressChecker())
            .AddTraceSource("ReadyToDispose",
                            "This application is ready to be disposed",
                            MakeTraceSourceAccessor(&DisposableApplication::m_readyToDisposeTrace),
                            "ns3::DisposableApplication::ReadyToDisposeCallback");
    return tid;
}

DisposableApplication::DisposableApplication()
    : m_socket(nullptr)
{
    NS_LOG_FUNCTION(this);
}

DisposableApplication::~DisposableApplication()
{
    NS_LOG_FUNCTION(this);
}

Ptr<Socket>
DisposableApplication::GetSocket() const
{
    NS_LOG_FUNCTION(this);
    return m_socket;
}

} // Namespace ns3