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

/** @file:  MirrorClientInternals.h
 *  @brief: Internal functions/methods for the MirrorClient API.
 *  
 */
#ifndef MIRRORCLIENTINTERNALS_H
#define MIRRORCLIENTINTERNALS_H

#include <vector>
#include <string>
#include <sstream>
typedef struct _MirrorInfo {
    std::string m_host;
    std::string m_memoryName;
    
} MirrorInfo, *pMirrorInfo;

/**
 *   GetMirrorList
 * @param host   - Host on which the rest server for SpecTcl is running.
 * @param port   - Port on which the REST server for SpecTcl is listening for connections.
 * @return std::vector<MirrorInfo> - Information about all mirrors.
 */
std::vector<MirrorInfo>
GetMirrorList(const char* host, int port);

/**
 * GetSpectrumSize
 *
 * @param host - host in which the rest server for SpecTcl is running.
 * @param port - Port on which the SpecTcl REST server is listening.
 * @return size_t - number of bytes in SpecTcl spectrum storage soup.
 */
size_t
GetSpectrumSize(const char* host, int port);

/**
 * GetSpecTclSharedMemory
 *    Return the name of the SpecTcl shared memory region.
 *
 *  @param host - host in which the SpecTcl REST server is running.
 *  @param port - Port on wich the SpecTcl REST server is listening.
 *  @return std::string - name of the region or throws std::runtime_error.
 */
std::string
GetSpecTclSharedMemory(const char* host, int port);

/**
 * LookupPort
 *    Lookup a port in the NSCLDAQ port manager.
 * @param host - host in which we do the lookup.
 * @param service - service for which we do the lookup.
 * @param user - user for which we do the lookup...defaults to the running user.
 * @return int  - Port number on which the service is listening.
 */
int
LookupPort(const char* host, const char* service, const char* user=nullptr);

// Windows functions:

#ifdef _WIN64
const char* getlogin();
#endif
#endif