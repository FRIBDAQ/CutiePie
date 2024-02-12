/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2005.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Author:
             Ron Fox
	     NSCL
	     Michigan State University
	     East Lansing, MI 48824-1321
*/


//
// Author:
//   Ron Fox
//   NSCL
//   Michigan State University
//   East Lansing, MI 48824-1321
//   mailto:fox@nscl.msu.edu
//
#ifndef CTCPCONNECTIONLOST_H
#define CTCPCONNECTIONLOST_H

#include "ErrnoException.h"
#include <string>

// Forward class references:

class CSocket;

/*!
  Encapsulate the connection lost exception.  This exception is thrown
  when a read is attempted on a busted socket.
  */
class CTCPConnectionLost : public CErrnoException
{
  // Private data members:

  std::string m_Host;
  std::string m_Port;

  mutable std::string m_ReasonText;

  // Constructors and related members:

public:
  CTCPConnectionLost(CSocket* pSocket, const char* pDoing);
  CTCPConnectionLost(const CTCPConnectionLost& rhs);
  virtual ~CTCPConnectionLost(){}

  CTCPConnectionLost& operator=(const CTCPConnectionLost& rhs);
  int                 operator==(const CTCPConnectionLost& rhs);

  // Selectors:
  

public:
  std::string getHost() const {
    return m_Host; }
  std::string getPort() const {
    return m_Port; }

  // Mutators:

protected:
  void setHost(const std::string& rHost) {
    m_Host = rHost;
  }
  void setPort(const std::string& rPort) {
    m_Port = rPort;
  }

  // Object operations:

public:
  virtual const char* ReasonText() const;

  // Utility functions:

protected:
  void Host(CSocket* pSocket);
  void Port(CSocket* pSocket);

};

#endif
