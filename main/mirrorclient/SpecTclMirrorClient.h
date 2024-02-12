/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2017.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Authors:
             Ron Fox
             Giordano Cerriza
	     NSCL
	     Michigan State University
	     East Lansing, MI 48824-1321
*/

/** @file:  SpecTclMirrorClient.h
 *  @brief: API to SpecTcl mirrroring system.
 */
#ifndef SPECTCLMIRRORCLIENT_H
#define SPECTCLMIRRORCLIENT_H

/**
 * This header provides the application programming interface for clients
 * programs that want access to SpecTcl spectrum shared memory.  SpecTcl runs
 * as many as two servers:
 *   *  A REST server provides HTTP access to pretty much all of SpecTcl's functions.
 *      This is needed by this API to provide the size of the local shared memory,
 *      and to determine if a shared memory mirror in this host already exists.
 *   *  A Mirror server supports on demand updates of the contents of the SpecTcl
 *      spectrum shared memory.  This is needed for obvious resons.
 *
 *  From the application's point of view we want to hide all of the negotiations
 *  that go on at the REST level to determine if there is a mirror and all of the
 *  junk needed to run the mirrorclient application and just provide a single,
 *  simple function that can be called which will return a pointer to the
 *  mirrored shared memory, regardless how it was gotten. 
 *
 * @note the API in this softwre is usable from C or C++.
 */

#ifdef _WIN64
#define EXPORT __declspec(dllexport)
#else
#define EXPORT 
#endif

#ifdef __cplusplus
extern "C" {
#endif
/**
 *  getSpecTclMemory
 *     Returns a pointer to a SpecTcl display memory mirror.
 *
 *  @param host - the host in which SpecTcl is running.
 *  @param rest - The Rest service.  This can be a port number or an NSCLDAQ
 *                advertised service name.
 *  @param mirror - The mirror service.  This can be a port number or an NSCLDAQ
 *                advertised service name.
 *  @param user - If neither rest nor mirror are NSCLDAQ services, this optional argument
 *               is ignored.  If either is a service:
 *               *  A nullptr will qualifiy service discovery by the name of the
 *                  user running this program.
 *               *  Anything else is a username that's assumed to be running SpecTcl.
 *                  This supports, in a collaborative environment such as an
 *                  experiment, user a looking at spectra accumulated by the
 *                  SpecTcl run by user b.
 * @return void* - Pointer to the shared memory region that holds the mirror.
 * @retval nullptr - The mirror, for some reason, could not be created.
 */
EXPORT void*
getSpecTclMemory(const char* host, const char* rest, const char* mirror, const char*user = 0);


/**
 * errorCode
 *    Can only be called after a failed call to getSpecTclMemory - returns
 *    the error code that describes the failure.  These are given symbolically
 *    towards the bottom of this file.
 *  
 *  @return int  - Error status from the failed getSpecTclMemory call.
 */
EXPORT int
Mirror_errorCode();

/**
 * errorString
 *     Returns a human readable description of the error from the code gotten
 *     via errorCode().
 *
 * @param code - the error code gotteen from errorCode().
 * @return const char*  - Pointer to the static error message string.
 */
EXPORT const char*
Mirror_errorString(unsigned code);

/*------------------------------------------------------------------------*/
/*  Error code symbolic values:                                           */

static const unsigned MIRROR_SUCCESS = 0;   // Successful completion.
static const unsigned MIRROR_NORESTSVC=1;   // REST service not advertised.
static const unsigned MIRROR_NOMIRRORSVC=2; // Mirror service not advertised.
static const unsigned MIRROR_CANTGETUSERNAME=3; // getlogin failed.
static const unsigned MIRROR_CANTGETSIZE=4;
static const unsigned MIRROR_CANTGETMIRRORS=5;
static const unsigned MIRROR_SETUPFAILED=6;
static const unsigned MIRROR_CANTGETHOSTNAME = 7;

#ifdef __cplusplus
}
#endif


#endif