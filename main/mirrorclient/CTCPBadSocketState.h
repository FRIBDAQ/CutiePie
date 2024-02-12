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


// Author
//   Ron Fox
//   NSCL
//   Michigan State University
//   East Lansing, MI 48824-1321
//   mailto:fox@nscl.msu.edu
//
// Copyright 
//   NSCL All rights reserved.
//
#ifndef CTPBADSOCKETSTATE_H
#define CTPBADSOCKETSTATE_H

// Headers:

#include "Exception.h"
#include "CSocket.h"
#include <vector>
#include <string>

/*!
  Encapsulates an exception which will be thrown whenever a CSocket member
  is called when the socket is in an invalid state.  The exception
  recognizes that there may be a list of valid states which the socket
  can be in and will indicate in the exception message the set of valid
  states.
  */
class CTCPBadSocketState : public CException
{
  // Private members:
  
  CSocket::State          m_BadState;	 //!< Incorrect state at time of throw.
  std::vector<CSocket::State>  m_ValidStates; //!< States which would have been ok.
  mutable std::string          m_Message;     //!< Full error message built up here

  // Constructors and related functions:
public:
  CTCPBadSocketState(CSocket::State badState,
		     std::vector<CSocket::State> okStates,
		     const char* pDoing);
  CTCPBadSocketState(const CTCPBadSocketState& rhs);
  virtual ~CTCPBadSocketState() {}

  CTCPBadSocketState& operator=(const CTCPBadSocketState& rhs);
  int                 operator==(const CTCPBadSocketState& rhs);


  // Selectors:
public:
  CSocket::State getBadState() const 
  { return m_BadState; }
  std::vector<CSocket::State> getValidStates() const
  { return m_ValidStates; }

  // Mutators:
protected:

  void setBadState(CSocket::State newState)
  { m_BadState = newState; }
  void setValidStates(const std::vector<CSocket::State>& newStates)
  { m_ValidStates = newStates; }

  // Operations on the class:

public:
  virtual const char* ReasonText() const;
    
};

#endif
