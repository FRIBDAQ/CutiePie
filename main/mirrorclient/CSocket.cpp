/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2008

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Author:
             Ron Fox
	     NSCL
	     Michigan State University
	     East Lansing, MI 48824-1321
*/


static const char* Copyright= "(C) Copyright Michigan State University 2002, All rights reserved";
/*!
           Encapsulates a generalized TCP/IP SOCK_STREAM
           socket. 
             Note that TCP/IP Sockets can come in two flavors:
           Clients and Servers.  Clients must perform a connect,
           while servers perform a bind, listen and then serveral
           accepts to create 'server instances'.
           The state of a socket is maintained in the m_State
           variable and is from the enumerator:
           CSocket::State
           
           Disconnected  - The socket is not connected to anything.
           Bound              - The socket is a server socket which is
                                      not connected, but has been bound to
                                      a service port.
           Listening           - The socket is a server port which is
                                       listening and can therefore accept
                                       connections
           Connected        - The socket is either a client or a 
                                       server instance and is connected
                                       to it's counterpart.
           
*/

// Synchronization assumptions:
//    the (2) socket calls are recursive (threadsafe).
//    the (3) network database calls are not recursive (notthreadsafe).
//  as such, any calls to (3) network database calls are bracketed with
//  calls to globally synchronize the thread.
/*! \bug Perhaps global syncronization of (3) calls should be conditionalized
         so that this module can be used in a non spectrodaq environment too.
    */

////////////////////////// FILE_NAME.cpp /////////////////////////////////////////////////////



#include "CSocket.h"    

// Exception classes required:

#include "CTCPNoSuchHost.h"
#include "CTCPNoSuchService.h"
#include "CTCPBadSocketState.h"
#include "CTCPConnectionFailed.h"
#include "ErrnoException.h"
#include "CTCPConnectionLost.h"
#include "io.h"
#include <iostream>

// Standard and network library headers.

#ifndef _WIN64
#include <unistd.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/poll.h>
#include <sys/socket.h>
#include <sys/time.h>
typedef  void *OPTPTR;
typedef int SOCKET;
#define CONST const
#else
#include <winsock2.h>
typedef int socklen_t;
static const int SHUT_RDWR = SD_BOTH;
typedef char* OPTPTR;
#define poll WSAPoll
#pragma comment(lib, "Ws2_32.lib")    // Tells the linker we need winsock2.
#endif

#include <stdio.h>
#include <sys/types.h>


#include <string.h>


#include <io.h>
#include <string>


// STL includes.

#include <vector>

// #define DEBUGGING 1
#ifdef DEBUGGING
#define DEBUG(msg) std::cout << msg << std::endl; std::cout.flush()
#else
#define DEBUG(msg)
#endif

using namespace std;

// Manifest constannts:

#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif

#ifndef HAVE_GETPROTOBYNAME_R
#include <mutex>
class Monitor {
private:
  std::mutex m_lock;
public:
  Monitor() {m_lock.lock();}
  ~Monitor() {m_lock.unlock();}
};


int getprotobyname_r(
    const char* name,
    struct protoent* result_buf,
    char* buf,
    size_t buflen,
    struct protoent **result
) {
  Monitor mon;

  auto proto = getprotobyname(name);
  if (proto) {
    memcpy(result_buf, proto, sizeof(struct protoent));
    *result = result_buf;
    return 0;
  } else {
    return -1;
  }
}
#endif

// Static members:

map<CSocket::State, string> CSocket::m_StateNames;
#ifdef _WIN64
bool CSocket::m_initialized(false);
#endif

/*!
  The default constructor  initializes the fd to an illegal value
  and sets the state to Disconnected.  
  The body of the constructor attempts to set the m_Fd member via a call to 
  socket(2) to create an INET domain SOCK_STREAM, protocl tcp.

  It is up to the user to call Connect
  or alternatively set up the socket as a server by binding and listening.

  Exceptions:

  - CErrnoException if getprotoent for tcp fails or if the socket call itself
    fails.

  */
CSocket::CSocket ()
   : m_Fd(-1),
     m_State(Disconnected)
 
{
  // if windows may need to init the socket system
#ifdef _WIN64
    if (!m_initialized) {
        WSADATA data;
        m_initialized = true;
        WSAStartup(0, &data);
    }
#endif


  OpenSocket();
} 

/*! 
  Destructor action depends on the state:
  sockets which are Connected are shutdown and then closed.
  Sockets which are in other states are just closed if the fd > 0.
  */
CSocket::~CSocket ( )  //Destructor - Delete dynamic objects
{
  if(m_State == Connected) {
    try {
      Shutdown();
    }
    catch (...) {		//  In case the state lies.
    }
  }
  if(m_Fd > 0) {
    shutdown(m_Fd, SHUT_RDWR);
  }
}

/*!
  This parameterized constructor is intended to allow a program which already
  has created a socket in some arbitrary state to wrap it inside a CSocket
  object.  It must be used with care.. in the sense that the am_State
  parameter must match the actual socket state.

  \param am_Fd     The file descriptor already open on a socket.
  \param am_State  The current state of the socket am_Fd

  */
CSocket::CSocket (    int am_Fd,  CSocket::State am_State  ) :
  m_Fd(am_Fd),
  m_State(am_State)

{  
  StockStateMap();		// Ensure the statename map is stocked.
} 

// Functions for class CSocket

/*!
  Operation Type:
    Connection Control
 

Indicates that the socket will be used
as a client socket and attempts to connect it
to a server. 
The address of the server can be
passed in either in IP address or
IP name textual format.  Similarly,
the port can be passed in as a textual
port name (in /etc/services) or a port number.

The action of this function is to convert (if possible) the Host and service 
into numeric equivalents and call the overloaded
Connect(int IpAddress, int service)

Exceptions which can be thrown:

- CTCPBadSocketState -- m_State was not Disconnected
- CTCPNoSuchHost       - Host not in DNS or nonexistent.
- CTCPNoSuchService  - Named service does not translate.
- CTCPConnectionFailed- Connection refused by remote host (from
                        Connect(int IpAddress, int service).

\param host Specifies the system to connect to.  This can be either a DNS
            textual name or a string in dotted IP address format.
\param service Specifies a service offered by the host.  This can be either
               a service name or the textual equivalent of a service number.

On Success the socket state is set to Connected.

*/
void 
CSocket::Connect(const string& host, const string& service)  
{ 
  // We must be disconnected to do any of this...
  
  throwIfIncorrectState(Disconnected, "CSocket::Connect");
  
  // If the host is a dotted ip number just translate it.. otherwise
  // try to resolve the name:

  unsigned long int ipaddr;
  struct hostent* pEntry;

    {				                   // (3) calls not threadsafe.
      ipaddr = inet_addr(host.c_str());
      if(ipaddr == INADDR_NONE) {	// Need to translate the name...
	pEntry = gethostbyname(host.c_str());
	if(pEntry) memcpy(&ipaddr, pEntry->h_addr, 4);
      }                                              // <-- End Critical region
      if(!pEntry) throw CTCPNoSuchHost(host, 
				       "Null return from gethostbyname(3)");
    }
  // Next resolve the service name... this can be either a number or a 
  // service name.  Note that the port number is in the service database
  // in network byte order.

  unsigned short port = Service(service);


  // Now we have everything we need to do call 'the other' Connect.

  Connect(ntohl(ipaddr), ntohs(port));
}  

/*!
  Connects a socket as a client given numerical host, and port numbers in 
  host byte order.   For the host that means that an ip address of form:

  aa.bb.cc.dd  is stored Hi->Low as aabbccdd in a longword.

  See Connect(const string&host, const string& service) for more information.

  \param IpAddress The numerical ip address of the server in host byte order.
  \param service   The port number of the service.
  */

void 
CSocket::Connect(unsigned long int IpAddress, unsigned short service)

{
  // Must be disconnected:

  throwIfIncorrectState(Disconnected, "CSocket::Connect");
  
  // Both the ip address and service must be in network byte order to be 
  // acceptable to connect(2).

  IpAddress = htonl(IpAddress);
  service   = htons(service);

  // Build the socket peer address structure.

  sockaddr_in Peer;
  Peer.sin_port = service;
  memcpy(&(Peer.sin_addr), &IpAddress, sizeof(unsigned long int));
  Peer.sin_family = AF_INET;

  // Try the connect.

  if(connect(m_Fd, (const sockaddr*)&Peer, sizeof(sockaddr_in)) < 0) {
    int sErrno = errno;
    string Ip, Svc;
    {
      char port[50];
      Ip = inet_ntoa(Peer.sin_addr);
      sprintf(port,"%d", ntohs(service));
      Svc  = port;

    }
    errno = sErrno;		// sprintf e.g. destroys errno.
    throw CTCPConnectionFailed(Ip, Svc,
			       "CSocket::Connect client connect(2) failed");
  }

  m_State = Connected;
  

}

/*!


Indicates that the socket will be used
as a server listener socket, and binds it
to a service port.  The service can be
provided either as a numerical string
or as a string translated via getservbyname().

The following exceptions can be thrown:

- CTCPBadSocketState m_State != Disconnected
- CTCPNoSuchService  Service could not be determined.
- CErrnoException   bind(2) failed.

On success, m_State = Bound

*/
void 
CSocket::Bind(const string& service)  
{
  throwIfIncorrectState(Disconnected, "CSocket::Bind");
  
  // Determine the service in network byte order.

  unsigned short port = Service(service);


  // Now bind the socket.

  sockaddr_in Peer;
  Peer.sin_port   = port;
  Peer.sin_family = AF_INET;
  Peer.sin_addr.s_addr = htonl(INADDR_ANY);

  if(bind(m_Fd, (sockaddr*)&Peer, sizeof(sockaddr_in)) < 0) {
    throw CErrnoException("Error from bind(2) in CSocket::Bind()");
  }
  m_State = Bound;

}  

/*!

Indicates that the specified server 
listener socket is ready to listen
for connections. 

The Following exceptions can be 
thrown:

- CTCPBadSocketState - m_State != Bound
- CErrnoException    - listen(2) failed.

On success, m_State = Listening

\param nBacklog The limit on the queue size for incomming connections. 
                This value of this parameter defaults to 5. If a
                a connection is requested when the listen queue is full, it
                is refused or allowed to retry depending on the protocol
		(according to linux man connect(2)).  On some systems this
		parameter may be ignored or have other meaning.
*/
void 
CSocket::Listen(unsigned int nBacklog)  
{ 
  // Throw CTCPBadSocketState if not bound:

  throwIfIncorrectState(Bound, "CSocket::Listen");
  
  if(listen(m_Fd, nBacklog) < 0) 
    throw CErrnoException("CSocket::Listen listen(2) failed");

  m_State = Listening;
}  

/*!

This member function can be called on 
a server socket.  The calling thread is blocked
until a connection request is received.  At that
time, the connection is accepted (granted),
and a new CSocket is created in the Connected
state.  The new CSocket represents a Server
Instance socket, communication along that
socket can take place immediately and will
represent communication with the client.

Exceptions which can be thrown:

- CTCPBadSocketState m_State != Listening
- CErrnoException    accept(2) call failed.

Side effects:

The client parameter is 
written with a string representing the
hostname of the client or the IP address
if the hostname can not be determined.

The socket created is created via new, therefore 
it is the responsibility of the caller to delete it.

\param client Recieves the IP information of the connecting host.
       Where possible, this is the hostname.  Where not, a dotted IP
       address.

*/
CSocket*  
CSocket::Accept(string& client)  
{ 
  // Throw CTCPBadSocket if not listening.

  throwIfIncorrectState(Listening, "CSocket::Accept");
  
  // Attempt the accept.  If successfule, the return
  // value will be a socket fd.

  struct sockaddr_in peerInfo;
  socklen_t          addrlen(sizeof(sockaddr_in));
  SOCKET peerFd = accept(m_Fd, (sockaddr*)&peerInfo, &addrlen);
  if(peerFd < 0) {
    throw CErrnoException("CSocket::Accept accept(2) failed");
  }

  // At this time we have received a good connection. There are two things
  // we need to do:
  //   1. Construct a new CSocket in the connected state.
  //   2. Figure out and fill in the client:

  CSocket* pConnection = new CSocket(peerFd, Connected);
  client = AddressToHostString(peerInfo.sin_addr);

  return pConnection;
}  

/*!
 
Purpose: 	

  Shuts down a connection to a remote system.
Unlike shutdown(2) this function does not support
selectively shutting down reads or writes.  Both are
unconditionally shutdown.
  Note that the destructor will automatically call Shutdown
if necessary.

Exceptions:
- CTCPBadSocketState m_State != Connected.
- CErrnoException    shutdown(2) failed.

*/
void 
CSocket::Shutdown()  
{ 
  // Require that the socket be connected.

  throwIfIncorrectState(Connected, "CSocket::Shutdown");
  
 
  if(shutdown(m_Fd, SHUT_RDWR) < 0) {
    throw CErrnoException("CSocket::Shutdown failed in call to shutdown(2)");
  }
  dropConnection();
}  

/*!
 
Purpose: 	

Performs a read on the socket.  The read will transfer
all of the bytes currently waiting in the socket buffers
or block until data is avaialble.  The return valiue will
be the number of bytes transferred.  If the connection
is lost, CTCPConnectionLost will be thrown.  Multiple reads will not be
performed so that any known messaging structure can be maintained.

Throws:
- CErrnoException - the read(2) system service returned an error.
- CTCPConnectionLost - the read(2) system service returned 0 indicating
                       an end of file condition.
- CTCPBadSocketState - m_State != Connected.

*/
int 
CSocket::Read(void* pBuffer, size_t nBytes)  
{ 
  // Require the socket be connected:

  DEBUG("CSocket::Read " << nBytes << " to " << std::hex << pBuffer << std::dec );
  throwIfIncorrectState(Connected, "CSocket::Read");
  DEBUG("State good");
  // Attempt the read:

  int nB;  
  try {
    DEBUG("Trying read:");
    nB = io::readData(m_Fd, pBuffer, nBytes);
    DEBUG("Got " << nB << " bytes");
    // Check for EOF:
  
    if(nB == 0) {
      dropConnection();
      throw CTCPConnectionLost(this, "CSocket::Read: from read(2)");
    }
  } catch (int e) {
    DEBUG("Some exception was caught" << e);
    m_State = Disconnected;
    throw CErrnoException("CSocket::Read failed read(2)");
  }
  
  return nB;
}  


/*!
 
Purpose: 	

Writes data to the socket.  Note that
this member will block as needed until
all data has been queued to the socket buffers.
This may require multiple write(2) function calls
if the amount of data to be written is larger than
the socket's blocking factor.  Note that if the connection
is lost during the write, CTCPConnectionLost will be thrown.

Exceptions:
- CTCPBadSocketState   m_State != Connected
- CErrnoException      write(2) returned an error condition.
- CTCPConnectionLost   write(2) indicated an EPIPE condition which says
                       the peer closed the socket.
   \bug There's not a good way to handle failures on the second
           or later call to write(2). Since we'd like to indicate that
           part of the write completed before an error occured.
           The current assumption is that a followup write will produce
           the same error.  Perhaps the best long term thing to do is to
	   define a CTCPSocketIOError which will include as data the
	   number of bytes written along with a CErrnoException which 
	   describes why the write actually failed??
   

*/
int 
CSocket::Write(const void* pBuffer, size_t nBytes)  
{
  // Require that the socket is connected:

    
  throwIfIncorrectState(Connected, "CSocket::Write");
  // Write the data mapping the exceptions as appropriate.
  

  try {

    io::writeData(m_Fd, pBuffer, nBytes);
    return nBytes;
  }
  catch (int err) {
      
    if (err == EPIPE) {
      
      dropConnection();
      throw CTCPConnectionLost(this,"CSocket::Write - first write of loop");
   
    } else {
      std::string msg = "CSocket::Write failed: ";
      msg += strerror(err);
      m_State = Disconnected;
      throw CErrnoException(msg.c_str());
    }
  }

 
}  

/*!
 
Purpose: 	

Returns information about who a socket
is connected to.  If the socket is not
connected, CTCPBadSocketState is thrown.
If possible, the peername parameter is returned
as a string containing the DNS name of the
peer. If the DNS lookup fails, the IP address
is converted into dotted form.

\param port The number of the port to which the socket is connected.
\param peer The peer as either a DNS hostname or a dotted IP address.

Throws:
- CTCPBadSocketState if m_State != Connected
- CErrnoException if getpeeername(2) failed.

*/
void 
CSocket::getPeer(unsigned short&  port, string& peer)  
{
  // Enforce connection requirement:

  throwIfIncorrectState(Connected, "CSocket::getPeer");
  
  // Try the call to getpeername(2):

  sockaddr_in PeerInfo;
  socklen_t   addrsize(sizeof(sockaddr_in));
  if(getpeername(m_Fd, (sockaddr*)&PeerInfo, &addrsize) < 0) {
    throw CErrnoException("CSocket::getPeer failed call to getpeername(2)");
  }
  // Pull out the port in local host byte order, and translate the
  // ip address:

  port = ntohs(PeerInfo.sin_port);
  peer = AddressToHostString(PeerInfo.sin_addr);

}  

/*!
 
Purpose: 	

Allows Out Of Band (OOB) data to be inserted in line with buffered data.
OOB data is data with a higher delivery priority than 'normal data'. If this
flag is not set, then by default OOB data must be read through normal
socket interface functions by specifying it in the recv flags parameter.
If this flag is set, oob data is queue at the front of the data to be read
with the Read member.

This is allowed in any socket state.
Exceptions:
- CErrnoException if the setsockopt(2) function failed.

\param State  This can be (Defaults to TRUE):
- TRUE Enables out of band inline data.
- FALSE disables out of band inline data.

*/
void 
CSocket::OOBInline(bool State)  
{

  int state((int)State);

  if(setsockopt(m_Fd, SOL_SOCKET, 
		SO_OOBINLINE, (const OPTPTR)&state, sizeof(int)) < 0) {
    throw CErrnoException("CSocket::OOBInline setsockopt(2) failed");
  }
}  

/*!
 
Purpose: 	

Returns TRUE if OOBinline is set
FALSE otherwise.  Note that the underlying
socket state is inquired, not some saved internal
state.

This function is valid in any socket state.

Throws:
- CErrnoException if getsockopt(2) fails
.
*/
bool 
CSocket::isOOBInline()  
{ 
  unsigned int state;
  socklen_t    size(sizeof(int));
  if(getsockopt(m_Fd, SOL_SOCKET, 
		SO_OOBINLINE, (OPTPTR)&state, &size) < 0) {
    throw CErrnoException("CSocket::OOBInline getsockopt(2) failed");
  }
  return state ? TRUE : FALSE;
}  

/*!

 
Purpose: 	

Sets the Receive low water mark for the socket.
This is the number of bytes received by the
protocol before any is made available to the
user.  Note that some systems do not allow 
this to be changed.  It is not an error at this
level to attempt to do so, however you will need to 
call getRcvlowWaterMark to be sure the change was
actually made.


Throws:
- CErrnoException if an error from setsockopt is detected other than that the
                  system doesn't support resetting the low water mark.

*/
void 
CSocket::setRcvLowWaterMark(size_t nBytes)  
{ 
  if(setsockopt(m_Fd, SOL_SOCKET, SO_RCVLOWAT, (OPTPTR)&nBytes, sizeof(size_t)) < 0) {
    if (errno == ENOPROTOOPT) {
      return; // Just not suppported is ok..
    }
    throw 
      CErrnoException("CSocket::setRcvLowWaterMark() setsockopt(2) failed");
  }
}


/*!

Returns the size of the current receive low water mark.
See setRcvLowWaterMark for information about what this
parameter does.  Note that the value returned is inquired from
the socket rather than stored in internal state.

Throws:
- CErrnoException if an error from getsockopt is detected.

*/
size_t 
CSocket::getRcvLowWaterMark()  
{ 
  int nBytes;
  socklen_t addrlen(sizeof(int));
  if(getsockopt(m_Fd, SOL_SOCKET, SO_RCVLOWAT, (OPTPTR)&nBytes, &addrlen) < 0) {
    throw 
      CErrnoException("CSocket::getRcvLowWaterMark() getsockopt(2) failed");
  }
  return (size_t)nBytes;
}  

/*!

Sets the new value of the Send Low water mark.
This controls the number of bytes which must be written
before transferring data to the protocol layers for transmission.
Note that some systems don't allow this value to be changed.
It is not an error to attempt to change this value on those systems,
however you should use getSndLowWaterMark to determine the
actual value negotiated by the system.

\param nBytes Number of bytes in the low water mark.

Throws:
- CErrnoException if an error from setsockopt is detected other than that the
                  system doesn't support resetting the low water mark.

*/
void 
CSocket::setSndLowWaterMark(size_t nBytes)  
{ 
  if(setsockopt(m_Fd, SOL_SOCKET, SO_SNDLOWAT, (const OPTPTR) & nBytes, sizeof(size_t)) < 0) {
    if (errno != ENOPROTOOPT) {
      CErrnoException("CSocket::setSndLowWaterMark() setsockopt(2) failed");
    }
  }  
}
/*!

Return the value of the current Send Low Water Mark
Set setSndLowWaterMarrk for more information.

Throws:
- CErrnoException if an error from getsockopt is detected.

*/
size_t 
CSocket::getSndLowWaterMark()  
{
  int nBytes;
  socklen_t adrlen(sizeof(int));
  if(getsockopt(m_Fd, SOL_SOCKET, SO_SNDLOWAT, (OPTPTR) & nBytes, &adrlen) < 0) {
    throw 
      CErrnoException("CSocket::getRcvLowWaterMark() getsockopt(2) failed");
  }
  return (size_t)nBytes; 
}  

/*!

Set the protocol receive timeouts.  Note that in some systems,
these are not settable.  However it is not an error to attempt to do so.

\param nMs number of milliseconds of timeout to set.

Throws:
- CErrnoException if a setsockopt(3) returns an error other than that this is
                  unsupported.
*/
void 
CSocket::setRcvTimeout(unsigned int nMs)  
{ 
  //
  // First convert the nMs parameter into a timeval.
  //
  timeval timeout;
  timeout.tv_sec  = nMs / 1000;	         // Number of seconds...
  timeout.tv_usec = (nMs % 1000) * 1000; // Left over milliseconds in uSec.

  // Now make the request:

  if(setsockopt(m_Fd, SOL_SOCKET, SO_RCVTIMEO, 
		(CONST OPTPTR) & timeout, sizeof(timeval)) < 0) {
    if(errno != ENOPROTOOPT) {
      throw CErrnoException("CSocket::setRcvTimeout setsockopt(2) failed");
    }
  }
}  

/*!
 
Purpose: 	

Retrieve the protocol receive timeout. The time out is returned as an integer
number of milliseconds.

Throws:
- CErrnoException if getsockopt(3) returns an error.

*/
unsigned int 
CSocket::getRcvTimeout()  
{ 
  timeval timeout;
  socklen_t adrlen(sizeof(timeval));
  
  // Get the timeout if possible: 

  if(getsockopt(m_Fd, SOL_SOCKET, SO_RCVTIMEO, 
		(OPTPTR) & timeout, &adrlen) < 0) {
    throw CErrnoException("CSocket::getRcvTimeout getsockopt(2) failed");
  }
  // Convert to ms and return.

  unsigned int nMs = timeout.tv_sec*1000; + timeout.tv_usec/1000000;
  return nMs;
}  

/*!
 
Purpose: 	

Set the number of milliseconds in the send timeout.
Some systems may not allow this to be set, however it is not
an error to try.

Throws:
- CErrnoException if setsockop(2) returned an error other than this option is
   not supported.

\param nMs Number of milliseconds in the desired timeout.

*/
void 
CSocket::setSndTimeout(unsigned int nMs)  
{ 
  // Convert timeout to a timeval structure:

  timeval timeout;
  timeout.tv_sec  = nMs / 1000;	         // Number of seconds...
  timeout.tv_usec = (nMs % 1000) * 1000; // Left over milliseconds in uSec.

  // Now make the request:

  if(setsockopt(m_Fd, SOL_SOCKET, SO_SNDTIMEO, 
		(const OPTPTR) & timeout, sizeof(timeval)) < 0) {
    if(errno != ENOPROTOOPT) {
      throw CErrnoException("CSocket::setRcvTimeout setsockopt(2) failed");
    }
  }
}  

/*!
Purpose:

Returns the current send timeout in ms.

Throws:
- CErrnoException if there was an error in getsockopt(2).

*/
unsigned int 
CSocket::getSndTimeout()  
{ 
  timeval timeout;
  socklen_t adrlen(sizeof(timeval));
  
  // Get the timeout if possible: 

  if(getsockopt(m_Fd, SOL_SOCKET, SO_SNDTIMEO, 
		(OPTPTR) & timeout, &adrlen) < 0) {
    throw CErrnoException("CSocket::getRcvTimeout getsockopt(2) failed");
  }
  // Convert to ms and return.

  unsigned int nMs = timeout.tv_sec*1000; + timeout.tv_usec/1000000;
  return nMs;
}  

/*!

Purpose: 	

Attempts to turn on Socket debugging.
To support this. The user must have effective
UID = 0.  .

Throws:
- CErrnoException if an error was returned from setsockopt(2)  At present
                  we don't know of systems which don't implement this so
                  all errors will throw.

\param fState [TRUE] Desired state of debugging.  TRUE will cause
                     debugging to be turned on.  FALSE turned off.
*/
void 
CSocket::Debug(bool fState)  
{
  int state((int)fState);
  if(setsockopt(m_Fd, SOL_SOCKET, SO_DEBUG, (OPTPTR) & state, sizeof(int)) < 0) {
    throw CErrnoException("CSocket::Debug setsockopt(2) failed");
  }

}  

/*!
 
Purpose: 	

Returns TRUE if socket debugging is
turned on and False otherwise. 

Exceptions:
- CErrnoException  if an error is returned from the getsockopt(2) call.

*/
bool 
CSocket::isDebug()  
{ 
  int fstate;
  socklen_t adrlen(sizeof(int));
  if(getsockopt(m_Fd, SOL_SOCKET, SO_DEBUG, (OPTPTR) & fstate, &adrlen) < 0) {
    throw CErrnoException("CSocket:Debug getsockopt(2) failed");
  }
  return fstate ? TRUE : FALSE;
}  

/*!
 
Purpose: 	

Allows the caller to control the routability of messages
sent on the socket.  If set, messages will not
be sent through a gateway.
Note:
 The socket need not be connected. Presumably,
if this flag is set prior to Connect on a client socket
the client will be unable to connect outside the local
subnet, and if set prior to Bind for a server, the server will
be unable to accept connections from outside the subnet.

Exceptions:
-CErrnoException if the setsockopt(2) call failed.

\param fRoutable [TRUE] :
- TRUE to turn OFF  Routing
- FALSE to turn ON  Routing.

*/
void 
CSocket::SetNotRoutable(bool fRoutable)  
{ 
  int state((int)fRoutable);
  if(setsockopt(m_Fd, SOL_SOCKET, SO_DONTROUTE, (const OPTPTR) & state, sizeof(int)) < 0) {
    throw CErrnoException("CSocket::SetNotRoutable setsockopt(2) failed");
  }

}  

/*!

 
Purpose: 	

Returns the state of the routability flag.

Throws:
- CErrnoException If getsockopt(2) failed.

Return values:
- TRUE Routing is turned off.
- FALSE Routing is tured on.

*/
bool 
CSocket::isNotRoutable()  
{ 
  int fState;
  socklen_t adrlen(sizeof(int));
  if(getsockopt(m_Fd, SOL_SOCKET, SO_DONTROUTE, (OPTPTR) & fState, &adrlen) < 0) {
    throw CErrnoException("CSocket::isNotRoutable getsockopt(2) failed");
  }
  return fState ? TRUE : FALSE;
}  

/*!
  \fn void CSocket::setSndBufSize(size_t nBufferSize) 
 Operation Type:
    Configuration
 
Purpose: 	

Sets the socket send buffer size.  This 
has to do with how many bytes can be 
sent in a single write(2) service call. Messages
larger than that must be segmented into multiple
write(2) calls.  Note howerver that 
CTCPSocket::Write automatically handles
any necessary segmentation.


Throws:
- CErrnoException if setsockopt(2) fails.

\param nBufferSize Number of bytes which can be sent in one write(2) call

*/
void 
CSocket::setSndBufSize(size_t nBufferSize)  
{ 
  unsigned int size((unsigned int)nBufferSize);
  if(setsockopt(m_Fd, SOL_SOCKET, SO_SNDBUF, (const OPTPTR) & size, sizeof(int)) < 0) {
    throw CErrnoException("CSocket::setSndBufSize setsockopt(2) failed");
  }
}  

/*!

 
Purpose: 	

Returns the number of bytes that can be
written in a single write(2) call.

Throws:
- CErrnoException if getsockopt(2) fails.

*/
size_t 
CSocket::getSndBufSize()  
{ 
  unsigned int size;
  socklen_t    adrlen(sizeof(int));
  if(getsockopt(m_Fd, SOL_SOCKET, SO_SNDBUF, (OPTPTR) & size, &adrlen) < 0) {
    throw CErrnoException("CSocket::getSndBufSize getsockopt(2) failed");
  }
  return (size_t)size;
}  

/*!

 

Sets the maximum number of bytes which can be received in a single
read(2) operation.  Note that CTCPSocket::Read does >NOT< 
automatically segment or else you may block when you'd like
to believe that a message has been received.

\throw CErrnoException if setsockopt(2) fails.

\param nBytes Number of bytes which can be sent in one write(2) call

*/
void 
CSocket::setRcvBufSize(size_t nBytes)  
{
  unsigned int size = (unsigned int)nBytes;
 
  if(setsockopt(m_Fd, SOL_SOCKET, SO_RCVBUF, (const OPTPTR) & size, sizeof(int)) < 0) {
    throw CErrnoException("CSocket::setRcvBufSize setsockopt(2) failed");
  }
}  

/*!

Purpose: 	

Returns the maximum number of bytes which can be 
recieved in a single read(2) call.

Throws:
- CErrnoException if getsockopt(2) fails.


*/
size_t 
CSocket::getRcvBufSize()  
{ 
  unsigned int size;
  socklen_t    adrlen(sizeof(int));
  if(getsockopt(m_Fd, SOL_SOCKET, SO_RCVBUF, (OPTPTR) & size, &adrlen) < 0) {
    throw CErrnoException("CSocket::getSndBufSize getsockopt(2) failed");
  }
  return (size_t)size;
}  

/*!
Purpose: 	

Sets the socket linger parameters.  Linger properties
govern the way a shutdown, operates.  Note that
object destruction which requires a shutdown implicitly
turns off linger.  
  If linger is enabled, then the close will block until
all pending data has been successfully sent or until the
linger timeout is exceeded.  

Throws:
- CErrnoException if the setsockopt(2) call failed.

\param lOn Determines if linger is on or off:
- TRUE Linger is on.
- FALSE Linger is off.
\param nLingerSeconds (only used if lOn is TRUE).  Indicates how many seconds
                  to linger on the close until a timeout is declared and the
                  close unblocks.
*/
void 
CSocket::setLinger(bool lOn, int nLingerSeconds)  
{ 
  // The linger structure is filled in the way it is so that the user could
  // conceivably omit a timeout interval if lOn == FALSE since in that case,
  // nLingerSeconds won't be referenced.

  linger lInfo;
  lInfo.l_onoff = lOn;
  if(lOn) lInfo.l_linger = nLingerSeconds;

  // Set the linger parameters:

  if(setsockopt(m_Fd, SOL_SOCKET, SO_LINGER, (const OPTPTR) & lInfo, sizeof(linger)) < 0) {
    throw CErrnoException("CSocket::setLinger setsockopt(2) failed");
  }
}  

/*!
Purpose: 	

Retrieve the linger parameters.

Throws:
- CErrnoException if the setsockopt(2) call failed.

\param isLingering [out] Receives the value of the linger state:
- TRUE linger is enabled.
- FALSE linger is not enabled.
\param nLingerSeconds [out] Receives the value of the linger timeout.  This
                 only has meaning in the event that isLingering == TRUE.


*/
void 
CSocket::getLinger(bool& isLingering, int& nLingerSeconds)  
{ 
  linger linfo;
  socklen_t addrlen(sizeof(linger));
    if(getsockopt(m_Fd, SOL_SOCKET, SO_LINGER, (OPTPTR) & linfo, &addrlen) < 0) {
      throw CErrnoException("CSocket::getLinger getsockopt(2) failed");
    }

  // Fill in the parameters from the linger structure:

  isLingering    = linfo.l_onoff;
  nLingerSeconds = linfo.l_linger; 
}
/*!
  Purpose:
    Determines the service which corresponds to a service string.
  Service strings can either be a numerical equivalent of a service port
  number or a service name which can be looked up in the service database.

  \param rService  The service name.

  Returns:  The port number or throws:
    -CTCPNoSuchService  The service cannot be translated.
 */
unsigned short
CSocket::Service(const string& rService)
{
  unsigned short port;
  int           nport;

  if(sscanf(rService.c_str(), "%d", &nport) != 1) { // Non numeric service number.
    struct servent* pEntry;
    {                                              //  getservbyname not recursive
      pEntry = getservbyname(rService.c_str(), "tcp");
      if(pEntry) port = pEntry->s_port;
    }
    if(!pEntry) throw CTCPNoSuchService(rService, 
				     "Null return from getservbyname(3)");
  }
  else {
    port = nport;
    port = htons(port);
  }
  return port;
}

/*!
  Purpose:
     Given an IP address in in_addr format, returns a string describing
     the address.  First gethostbyaddr(3) is used to attempt to get the
     primary DNS name of the address.  If this fails, then inet_ntoa(3)
     is called to get a dotted string.  

     \param peer - The network address of a host in in_addr, network byte order
                   form.

 */
string
CSocket:: AddressToHostString(in_addr peer)
{
  string result;
  {
    struct hostent* pEntry = gethostbyaddr((const char*)&peer, 
					   sizeof(in_addr), AF_INET);
    if(pEntry) {		// Reverse DNS worked...
      result = pEntry->h_name;
      
    }
    else {			// Need to return dotted address.
      result = inet_ntoa(peer);
    }
  }

  return result;
}
/*!
  Purpose:
    Given a state, returns a text string which describes the state.
    \param state - State to describe.

    NOTE: Since this is a static member which can be called before any
    instances of CSocket have been created, we call StockStateMap().

 */
string
CSocket::StateName(CSocket::State state)
{
  StockStateMap();
  return m_StateNames[state];
}
/*!
  Purpose:
    Stocks the static member: m_StateNames with the states and their names.
    This is only done if the map is empty.  This function must be updated if
    the set of states is modified.
 */
void 
CSocket::StockStateMap()
{
  if(!m_StateNames.empty()) {
    m_StateNames[Disconnected] = "Disconnected";
    m_StateNames[Bound]        = "Bound";
    m_StateNames[Listening]    = "Listening";
    m_StateNames[Connected]    = "Connected";
  }
}
/*!
  Opens the socket on a TCP/IP endpoint.
  */
void
CSocket::OpenSocket()
{
  int             protocol;
  struct protoent  entry;
  char             entryFields[1024];
  struct protoent* pEntry;
  StockStateMap();		// Ensure the state name map is stocked.
  int status;

  status = getprotobyname_r("TCP",  &entry, entryFields, sizeof(entryFields),
			    &pEntry);


  if(!pEntry) throw 
	      CErrnoException("Getting TCP protocol num from getprotoent(3)");
  protocol = pEntry->p_proto;
  m_Fd = socket(PF_INET, SOCK_STREAM, protocol);
  if(m_Fd == -1) throw
		   CErrnoException("Creating socket with socket(2)");

}
/*!
   Flush buffered input from the socket.
*/
void
CSocket::Flush()
{
  struct pollfd myfd;
  myfd.fd = m_Fd;
  myfd.events = POLLHUP | POLLIN;
  myfd.revents = 0;
  int result;
  while(result = poll(&myfd,1, 0)) {
    if(result < 0) {		// probably disconnect...
      return;
    }
    if(myfd.revents & POLLIN) {
      char in;
      Read(&in, 1);		// Flush a character.
    }
    if(myfd.revents & POLLHUP) { // Probably disconnect...
      return;
    }
    myfd.revents = 0;
  }
  
}
/**
 * throwIfIncorrectState
 *    If the current state does not match a required state
 *    a CTCPBadSocketState is throw indicting that.
 * @param required - the required state.
 * @param doing    - what was being done (part of exception string.)
 * @throws CTCPBadSocketState if m_State != required.
 */
void
CSocket::throwIfIncorrectState(State required, const char* doing)
{
 if(m_State != required) {
    vector<CSocket::State> allowedStates;
    allowedStates.push_back(required);
    throw CTCPBadSocketState(m_State, allowedStates, 
			     doing);
  }

}
/**
 * dropConnection
 *    Drop any connection from our side.
 *    - shutdown is called.
 *    - the file descriptor is closed.
 *    - We're set to disconnected.
 *    - A nwe fd is opened in preparation to re-use.
 */
void
CSocket::dropConnection()
{
  m_State = Disconnected;
  shutdown(m_Fd, SHUT_RDWR);
  m_Fd = -1;			// Mark the socket closed.
  OpenSocket();

}
