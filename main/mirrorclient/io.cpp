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

/**
 * @file io.cpp
 * @brief Commonly used I/O methods.
 * @author Ron Fox.
 */

#include "io.h"

#include <errno.h>
#include <stdint.h>
#include <set>
#include <sstream>
#include <iostream>
#include <stdlib.h>

#ifdef _WIN64
#include <io.h>
#include <WinSock2.h>  // We're using this for socket I/O here.
typedef int ssize_t;
#else 
#include <unistd.h>
#include <sys/types.h>
#endif

// #define DEBUGGING
#ifdef DEBUGGING
#define DEBUG(msg) std::cout << msg << std::endl; std::cout.flush()
#else
#define DEBUG(msg)
#endif

static std::set<int>                 okErrors;	// Acceptable errors in I/O operations.

/**
 * Return true if an I/O errno is not an allowed one.
 * 
 * @param error - the  errno to check.
 *
 * @return bool - True if the error is a bad one.
 *
 * @note okErrors is a set that will contain the 'allowed' errors.
 */
static bool 
badError(int error)
{
  // Stock the okErrors set if empty:

  if (okErrors.empty())
  {
    okErrors.insert(EAGAIN);
    okErrors.insert(EWOULDBLOCK);
    okErrors.insert(EINTR);
  }

  // Not in the set -> true.

  return (okErrors.count(error) == 0);
}

static long maxIovecSize(-1);

namespace io {
  
/**
 * Write a block of data to a file descriptor.
 * As with getBuffer, multiple writes are done..until either the entire data
 * are written or
 * *  A write ends in an eof condition.
 * *  A write ends in an error condition that is not EAGAIN, EWOUDLDBLOCK or EINTR.
 *
 * @param fd    - file descriptor to which the write goes:
 * @param pData - Pointer to the data to write.
 * @param size  - Number of words of data to write.
 * 
 * @throw int 0 - I/O showed eof on output.
 * @throw int errno - An error and why.
 */

void writeData (
#ifdef _WIN64
    SOCKET fd,
#else 
    int fd,
#endif   
    const void* pData , size_t size)
{
  const uint8_t* pSrc(reinterpret_cast<const uint8_t*>(pData));
  size_t   residual(size);
  ssize_t  nWritten;
  
  while (residual) {
#ifdef _WIN64
      nWritten = send(fd, (char*)pSrc, residual, 0);
#else
    nWritten = write(fd, pSrc, residual);
#endif
    if (nWritten == 0) {
      throw 0;
    }
    if ((nWritten == -1) && badError(errno)) {
      throw errno;
    }
    // If an error now it must be a 'good' error... set the nWritten to 0 as no data was
    // transferred:

    if (nWritten < 0)
    {
      nWritten = 0;
    }
    // adjust the pointers, and residuals:


    residual -= nWritten;
    pSrc     += nWritten;
  }

}
/**
 * Get a buffer of data from  a file descritor.
 * If necessary multiple read() operation are performed to deal
 * with potential buffering between the source an us (e.g. we are typically
 * on the ass end of a pipe where the pipe buffer may be smaller than an
 * event buffer.
 * @param fd      - File descriptor to read from.
 * @param pBuffer - Pointer to a buffer big enough to hold the event buffer.
 * @param size    - Number of bytes in the buffer.
 *
 * @return size_t - Number of bytes read (might be fewer than nBytes if the EOF was hit
 *                  during the read.
 *
 * @throw int - errno on error.
 */
  size_t readData (
#ifdef _WIN64
      SOCKET fd,
#else 
      int fd,
#endif
      void* pBuffer,  size_t nBytes)
{
  uint8_t* pDest(reinterpret_cast<uint8_t*>(pBuffer));
  size_t    residual(nBytes);
  ssize_t   nRead;

  DEBUG("readData for " << nBytes);
  
  // Read the buffer until :
  //  error other than EAGAIN, EWOULDBLOCK  or EINTR
  //  zero bytes read (end of file).
  //  Regardless of how all this ends, we are going to emit a message on sterr.
  //

  while (residual) {
    DEBUG(residual << " bytes left");
#ifdef _WIN64
      nRead = recv(fd, (char*)pDest, residual, MSG_WAITALL);
#else
    nRead = read(fd, pDest, residual);
#endif
    DEBUG(nRead << " bytes read");
    if (nRead == 0)		// EOF
    {
      return nBytes - residual;
    }
    if ((nRead < 0) && badError(errno) )
    {
      DEBUG("Bad read with " << errno);
#ifdef _WIN64
      throw WSAGetLastError();
#else
      throw errno;
#endif
    }
    // If we got here and nread < 0, we need to set it to zero.
    
    if (nRead < 0)
    {
      nRead = 0;
    }

    // Adjust all the pointers and counts for what we read:

    residual -= nRead;
    pDest  += nRead;
  }
  // If we get here the read worked:

  return nBytes;		// Complete read.
}

}
