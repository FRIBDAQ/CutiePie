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


// Author:
//   Ron Fox
//   NSCL
//   Michigan State University
//   East Lansing, MI 48824-1321
//   mailto:fox@nscl.msu.edu
//
// Copyright 
//   NSCL All rights reserved.
//
#ifndef CSOCKET_H  //Required for current class
#define CSOCKET_H

//
// Include files:
//

#include <string>
#include <map>
#include <vector>
#ifndef _WIN64
#include <netinet/in.h>
#else
#include <winsock2.h>
#endif

#ifndef TRUE
#define TRUE 1
#endif


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
  
  - Disconnected:  The socket is not connected to anything.
  - Bound:         The socket is a server socket which is
                   not connected, but has been bound to
                   a service port.
  - Listening:     The socket is a server port which is
                   listening and can therefore accept
                   connections
  - Connected      The socket is either a client or a 
                   server instance and is connected
                   to it's counterpart.
  
  */
class CSocket
{
public:

  //! Captures the state of the socket.  See general remarks for more info
  enum State {
    Disconnected,
    Bound,
    Listening,
    Connected
  };

private:
  
  // Private Member data:
#ifdef _WIN64
    SOCKET m_Fd;
#else
  int m_Fd;			//!<  Socket  
#endif
  CSocket::State m_State;	//!<  State of socket.  
  static std::map<CSocket::State, std::string> m_StateNames;  //!< State name lookup tbl.
#ifdef _WIN64 
  static bool m_initialized; //!< Windows socket layer requires initialization.
#endif
   
  // Public nested data types:



public:
  CSocket ();
  CSocket (    int am_Fd,  CSocket::State am_State  ) ;

  virtual ~CSocket ( );
  
  //! Copy Constructor forbidden.
private:
  CSocket (const CSocket& aCSocket );
  
  //! Assignment Operator Forbidden
  CSocket& operator= (const CSocket& aCSocket);
 
  //! Equality Operator Forbidden
  int operator== (const CSocket& aCSocket) const;
public:

// Selectors:

public:

  //! Get accessor function socket file descriptor.
  int getSocketFd() const
  { return m_Fd;
  }  

  //! Get accessor function for socket state.
  CSocket::State getState() const
  { return m_State;
  }   

// Attribute mutators:

protected:

  //! Set accessor function for Socket file descriptor.
  void setSocketFd(const int am_Fd)
  { m_Fd = am_Fd;
  }  
  //! Set accessor function for current socket state.
  void setState (const CSocket::State am_State)
  { m_State = am_State;
  }   

  // Class operations:

public:

  void Connect (const std::string&     host,      const std::string&  service);
  void Connect (unsigned long int IpAddress, unsigned short service);
  void Bind (const std::string& service)   ;
  void Listen (unsigned int nBacklog=5)   ;
  CSocket*  Accept (std::string& client)   ;
  void Shutdown ()   ;
  int Read (void* pBuffer, size_t nBytes)   ;
  int Write (const void* pBuffer, size_t nBytes)   ;
  void getPeer (unsigned short& port, std::string& peer)   ;
  void OOBInline (bool State=TRUE)   ;
  bool isOOBInline ()   ;
  void setRcvLowWaterMark (size_t nBytes)   ;
  size_t getRcvLowWaterMark ()   ;
  void setSndLowWaterMark (size_t nBytes)   ;
  size_t getSndLowWaterMark ()   ;
  void setRcvTimeout (unsigned int nMs)   ;
  unsigned int getRcvTimeout ()   ;
  void setSndTimeout (unsigned int nMs)   ;
  unsigned int getSndTimeout ()   ;
  void Debug (bool fState=TRUE)   ;
  bool isDebug ()   ;
  void SetNotRoutable (bool fRoutable=TRUE)   ;
  bool isNotRoutable ()   ;
  void setSndBufSize (size_t nBufferSize)   ;
  size_t getSndBufSize ()   ;
  void setRcvBufSize (size_t nBytes)   ;
  size_t getRcvBufSize ()   ;
  void setLinger (bool lOn, int nLingerSeconds)   ;
  void getLinger (bool& isLingering, int& nLingerSeconds)   ;

  static std::string StateName(CSocket::State state);

  void Flush();			//!< Get rid of buffered input.

  // Protected utility functions.
protected:
  static void StockStateMap();
  unsigned short Service(const std::string& rService);
  std::string         AddressToHostString(in_addr peer);
  void           OpenSocket();
  void throwIfIncorrectState(State required, const char* doing);
  void dropConnection();

};

#endif

