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
#ifndef IO_H
#define IO_H

// This is needed in order to get VC to ignore this header
// otherwise it will insist on precompiling this header during
// sip-build and this header is definitely C++
#ifdef __cplusplus
#include <stdint.h>
#ifdef _WIN64
#include <WinSock2.h>
#endif

/**
 * @file io.h
 * @brief Commonly used I/O method definitions.
 * @author Ron Fox
 */

namespace io {
void writeData (
#ifdef _WIN64
    SOCKET fd,
#else 
      int fd, 
#endif
    const void* pData , size_t size);
  size_t readData (
#ifdef _WIN64
      SOCKET fd,
#else
      int fd,
#endif
      void* pBuffer,  size_t nBytes);
  
}
#endif

#endif
