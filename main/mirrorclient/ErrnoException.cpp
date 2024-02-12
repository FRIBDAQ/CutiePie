/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2014.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Authors:
             Ron Fox
             Jeromy Tompkins 
	     NSCL
	     Michigan State University
	     East Lansing, MI 48824-1321
*/


//  CErrnoException.cpp
//    Encapsulates exceptions which are thrown due to bad
//    errno values.
//
//
//   Author:
//      Ron Fox
//      NSCL
//      Michigan State University
//      East Lansing, MI 48824-1321
//      mailto:fox@nscl.msu.edu
//
//////////////////////////.cpp file///////////////////////////////////////////

//
// Header Files:
//

#include "ErrnoException.h"                               
#include <errno.h>
#include <string.h>
static const char* Copyright = 
"CErrnoException.cpp: Copyright 1999 NSCL, All rights reserved\n";

// Functions for class CErrnoException

//////////////////////////////////////////////////////////////////////////
//
//  Function:   
//    const char*  ReasonText (  )
//  Operation Type:
//     Selector.
//
const char*
CErrnoException::ReasonText() const 
{
  return strerror(m_nErrno);

}
//////////////////////////////////////////////////////////////////////////
//
//  Function:   
//    int ReasonCode (  )
//  Operation Type:
//     Selector.
//
int 
CErrnoException::ReasonCode() const 
{
  return m_nErrno;
}
