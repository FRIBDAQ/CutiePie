/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2009.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Author:
             Ron Fox
	     NSCL
	     Michigan State University
	     East Lansing, MI 48824-1321
*/
#include "os.h"
#include <ErrnoException.h>
#include <pwd.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <time.h>
#include <signal.h>
#include <stdexcept>
#include <netdb.h>


static const unsigned NSEC_PER_SEC(1000000000); // nanoseconds/second.


/**
 * Get name of current user.
 * @return std::string
 */
std::string
Os::whoami()
{
  struct passwd  Entry;
  struct passwd* pEntry;
  char   dataStorage[1024];	// Storage used by getpwuid_r(3).
  uid_t  uid = getuid();

  if (getpwuid_r(uid, &Entry, dataStorage, sizeof(dataStorage), &pEntry)) {
    int errorCode = errno;
    std::string errorMessage = 
      "Unable to determine the current username in CTheApplication::destinationRing: ";
    errorMessage += strerror(errorCode);
    throw errorMessage;
    
  }
  return std::string(Entry.pw_name);
}


/**
 * Os::usleep
 *
 *    Wrapper for nanosleep since usleep is deprecated in POSIX
 *    but nanosleep is consider good.
 *
 * @param usec - Number of microseconds to sleep.
 * @return int - Status (0 on success, -1 on error)
 * 
 * @note No attempt is made to map errnos from usleep -> nanosleep...so you'll 
 *       get then nanosleep errnos directly.
 * @note We assume useconds_t is an unsigned int like type.
 */
int
Os::usleep(useconds_t usec)
{
  // usec must be converted to nanoseconds and then busted into
  // seconds and remaning nanoseconds
  // we're going to assume there's no overflow from this:
  
  useconds_t nsec = usec* 1000;			// 1000 ns in a microsecond.

  // Construct the nanosleep specification:

  struct timespec delay;
  delay.tv_sec  = nsec/NSEC_PER_SEC;
  delay.tv_nsec = nsec % NSEC_PER_SEC;


  struct timespec remaining;
  int stat;

  // Usleep is interrupted with no clue left about the remainnig time:

  return nanosleep(&delay, &remaining);

 
}
/**
 * Os::blockSignal
 *   Blocks the specified signal.
 *
 * @param sigNum - Number of the signal to block.
 *
 * @return value from sigaction
 */
int
Os::blockSignal(int sigNum)
{

  // Build the sigaction struct:

  struct sigaction action;
  action.sa_handler = 0;		// No signal handler.
  sigemptyset(&action.sa_mask);
  sigaddset(&action.sa_mask, sigNum);
  action.sa_flags = 0 ;

  struct sigaction oldAction;

  return sigaction(sigNum, &action, &oldAction);

  
}

int Os::checkStatus(int returnStatus, int checkedStatus, std::string msg) 
{
  if (returnStatus!=checkedStatus) {
    throw std::runtime_error(msg);
  }
  //otherwise, we should just return the status
  return returnStatus;
}

int Os::checkNegativeStatus(int returnStatus) 
{
  if (returnStatus<0) {
    throw CErrnoException(strerror(errno));
  }

  return returnStatus;
}
/**
 * getfqdn
 *    Return the fully qualified domain name of a host.
 *
 *  @param host - the host name to lookup.  Note that this can be either
 *                a partially or fully qualified domain name.
 *  @return std::string - Fully qualified domain name.
 *  @throw std::string  - If the gethostbyname_r call fails.  This
 *                        could happen for a few reasons including:
 *                        *  No DNS server available.
 *                        *  The input host does not exist.
 */
std::string
Os::getfqdn(const char* host)
{
  struct addrinfo  hints = {AI_CANONNAME | AI_V4MAPPED | AI_ADDRCONFIG,
			    AF_UNSPEC, 0, 0, 
			    0, NULL, NULL, NULL};
			    
  struct addrinfo* hostInfo;
  checkNegativeStatus( getaddrinfo(host, NULL, &hints, &hostInfo) );

  std::string fqhostname(hostInfo->ai_canonname);
  freeaddrinfo(hostInfo);

  return fqhostname;
}


CPosixOperatingSystem&
CPosixOperatingSystem::operator=(const CPosixOperatingSystem& rhs)
{
    // the class has no state associated with it, so this is trivial.
    return *this;
}

