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

/** @file:  SpecTclMirrorClient.cpp
 *  @brief: Implements the APi described in SpecTclMirrorClient.h
 */

#include "SpecTclMirrorClient.h"
#include "MirrorClientInternals.h"

#include <stdlib.h>
#include <stdexcept>
#include <string.h>
#include <iostream>
#include <stdio.h>
#include <stdio.h>
#include <thread>

#ifdef DEBUGGING
#define DEBUG(msg) std::cout << msg << std::endl; std::cout.flush()
#else
#define DEBUG(msg)
#endif


static unsigned lastError = MIRROR_SUCCESS;
static const int UPDATE_INTERVAL = 2;

#ifndef _WIN64

#include <client.h>
#include <os.h>



#include <sys/types.h>
#include <unistd.h>



static const char* ExecDirs=SPECTCL_BIN;
static const unsigned MAP_RETRY_SECS=1;
static const unsigned MAP_RETRIES=10;



// Utility functions.

/**
 * sameHost
 *   Determine if two host names refer to the same host.
 *   We match the fqdns of the names.  Note that this can miss that localhost
 *   is the same as some named host but that's been dealt with elsewhere.
 *  @param h1 - first host.
 *  @param h2 - second host
 *  @return bool - true if the two hosts have the same fqdn.
 */
static bool
sameHost(const char* h1, const char* h2)
{
    std::string strH1 = Os::getfqdn(h1);
    std::string strH2 = Os::getfqdn(h2);
    
    return strH1 == strH2;
}


/**
 * isLocalHost
 *    Determines if a host name is the same as the localhost.
 *    -  If host == localhost it's local.
 *    -  if host == results of gethostname() it's local.
 *    -  If host == the fqdn it's local.
 *  @param host - hostname to check.
 *  @return bool - true if this system is host.
 *  
 */
static bool
isLocalHost(const char* host)
{
    std::string strHost(host);
    if (strHost == "localhost" ) return true;
    char gottenhostname[1024];
    memset(gottenhostname, 0, 1024);  // ensure null termination.
    if (gethostname(gottenhostname, sizeof(gottenhostname) - 1)) {
        lastError = MIRROR_CANTGETHOSTNAME;
        throw std::runtime_error("gethostname() failed");
    }
    if (strHost == gottenhostname) return true;
    if (strHost == Os::getfqdn(host)) return true;
    
    
    return false;
    
    
}
/**
 * MapMemory
 *    Map memory that's local.
 * @param name - key for the memory.
 * @param size - bytes of spectrum memoryt.
 * @return void* pointer to the memory.
 * @retval nullptr - could not map.
 */
static void*
MapMemory(const char* name, size_t size)
{
    volatile Xamine_shared* pResult;
    int status =
            Xamine_MapMemory(const_cast<char*>(name), size, &pResult);
    if (status && (pResult != reinterpret_cast<void*>(-1))) {
        return const_cast<Xamine_shared*>(pResult);
    } else {
        return nullptr;
    }
    return nullptr;                  // Can't actually get here but g++ complained.
}

/**
 * mapSpecTclLocalMemory
 *   Find out what the SpecTcl shared memory name is and map it.
 *
 *  @param host  - name of the host.
 *  @param port  - REST server port.
 *  @param size  - Bytes of spectrum memory.
 *  @return void* - Pointer to the spectrum  memory.
 *  @retval nullptr - failed to map.
 */
static void*
mapSpecTclLocalMemory(const char* host, int port, size_t size)
{
    try {
        auto key = GetSpecTclSharedMemory(host, port);
        Xamine_shared* pResult;
        return MapMemory(key.c_str(), size);
    }
    catch(...) {
        return nullptr;
    }
}
/**
 * mapExistingMirror
 *    Maps an existing mirror, if it really does exist.
 * @param mirrors - mirror list.
 * @param size    - memory size.
 * @return void*  - nullptr if no matching mirror else pointer to mirror map.
 */
static void*
mapExistingMirror(const std::vector<MirrorInfo>& mirrors, size_t size)
{
    char gottenhostname[1024];
    memset(gottenhostname, 0, 1024);  // ensure null termination.
    if (gethostname(gottenhostname, sizeof(gottenhostname) - 1)) {
        lastError = MIRROR_CANTGETHOSTNAME;
        throw std::runtime_error("gethostname() failed");
    }
    for (auto item : mirrors) {
            if (gottenhostname, item.m_host.c_str()) {
                return MapMemory(item.m_memoryName.c_str(), size);
            }
    }
    return nullptr;
}

/**
 * getMirrorIfLocal
 *    Given the lists of mirrors that SpecTcl is exporting, including the
 *    shared memory it created for itself, if one matches our needs,
 *    map it and return a pointer to that map.  There are a few special cases,
 *    however:
 *    - SpecTcl is running locally - in that case rather than attempting to
 *      create a mirror, we can just map to SpecTcl's own shared memory.
 *    - SpecTcl is running locally in a persistent container - in that
 *      case maps to SpecTcl's shared memory will fail because
 *      the container has a separate SYS-V IPC namespace so SpecTcl's
 *      shared memory is invisibl.  In that case we do need to treat this
 *      like mirroring
 * @param host - Host on which SpecTcl is running.
 * @param rest    - REST port number.
 * @param mirrors - mirrors currently being maintained.
 * @param size    - Spectrum memory size.
 * @return void*  - Pointer to the shared memory.
 * @retval nullptr - If local mapping is not possible (mirror needs to be setup).
 *
 */
static void*
getMirrorIfLocal(
    const char* host, int rest, 
    const std::vector<MirrorInfo>& mirrors, size_t size
)
{
    // Handle the special case of SpecTcl run locally (both of them).
    if (isLocalHost(host)) {
        void* pResult = mapSpecTclLocalMemory(host, rest, size);
        if (!pResult) {
            pResult = mapExistingMirror(mirrors, size);
        }
        return pResult;
    } else {
        return mapExistingMirror(mirrors, size);
    }
    // No match
    
    return nullptr;
}
/**
 * startMirroring
 *    -  Run the mirrorclient program to start mirroring.
 *    -  Wait a bit to let the mirrorclient produce its shared memory.
 *    -  Get mirror information and use getMirrorIfLocal to map to it.
 *       This bit of waiting and mapping can be repeated a few times.
 * @param host - host in which SpecTcl is running.
 * @param mirror - Port on which the SpecTcl mirror server is listening.
 * @param rest   - Port on which the SpecTcl REST server is listening.
 * @param size   - Spectrum bytes.
 * @return void*   - Pointer to specTcl mirrored memory.
 * @retval nullptr - if we can't do all this stuff.
 */
void*
startMirroring(const char* host, int mirror, int rest, size_t size)
{
    pid_t child = fork();
    if (child == -1) {
        lastError = MIRROR_SETUPFAILED;
        return nullptr;
    }
    if (child) {
        // Parent
        
        void* pResult(nullptr);
        for (int i =0; i < MAP_RETRIES; i++) {
            sleep(MAP_RETRY_SECS);
            auto mirrors = GetMirrorList(host, rest);
            pResult = getMirrorIfLocal(host, rest, mirrors, size);
            if (pResult) break;
        }
        if (!pResult) lastError = MIRROR_SETUPFAILED;
        return pResult;
    } else {
        // child
        
        // Close stdin,out,error
        
        //close(0);
        //close(1);
        //close(2);
        pid_t session = setsid();          // Create a new session.
        std::cerr << "Child session " << session << std::endl;
        if (session < 0) {
            exit(EXIT_FAILURE);            // failed
        }
        
        // formulate the mirrorclient command and arguments.
        // Since we're passing integer ports we don't need --user.
        
        std::string program(SPECTCL_BIN);
        program += "/mirrorclient";
        
        std::string hostarg = "--host=";
        hostarg            +=  host;
        
        std::string mirrorarg = "--mirrorport=";
        mirrorarg += std::to_string(mirror);
        
        std::string restarg = "--restport=";
        restarg +=  std::to_string(rest);
        
        std::cerr << program << " " << hostarg << " " << mirrorarg << " " << restarg <<std::endl;
        
        execl(
            program.c_str(), program.c_str(),
            hostarg.c_str(), mirrorarg.c_str(), restarg.c_str(),
            nullptr
        );
        // If we got here the execl failed.
        perror("Failed to execl");
        exit(EXIT_FAILURE);
    }
}
#else
#include "Mirrorclient.h"
#include <Windows.h>

/**
*    We're going to keep a registry of mirrors we are running.. the assumption is that the
*    number of mirrors is small....and that a single host/rest port will not have more than one
*    mirror port, so a mirror will be defined by the host and rest port and pointer triplet.
*    getMirrorIfLocal is now repurposed to see if we're already mirroring that guy.
*/
struct _RunningMirrorInfo {
    std::string s_hostName;
    int s_restPort;
    void* s_MirrorPointer;
};
typedef _RunningMirrorInfo RunningMirrorInfo;

std::vector<RunningMirrorInfo> runningMirrors;

/**
*  MirrorThread
*    Just update the mirror until it fails:
*   @param pClient client of the mirror server.
*   @param interval - seconds between updates.
*/
void MirrorThread(MirrorClient* pClient, unsigned interval) {
    pClient->initialize();
    while (true) {
        
        try {
            pClient->update();
        }
        catch (...) {
            delete pClient;
            break;
        }
        Sleep(interval * 1000);
    }
}
 
/**
 * getMirrorIfLocal
 *    Given the lists of mirrors that SpecTcl is exporting, including the
 *    shared memory it created for itself, if one matches our needs,
 *    map it and return a pointer to that map.  There are a few special cases,
 *    however:
 *    - SpecTcl is running locally - in that case rather than attempting to
 *      create a mirror, we can just map to SpecTcl's own shared memory.
 *    - SpecTcl is running locally in a persistent container - in that
 *      case maps to SpecTcl's shared memory will fail because
 *      the container has a separate SYS-V IPC namespace so SpecTcl's
 *      shared memory is invisibl.  In that case we do need to treat this
 *      like mirroring
 * @param host - Host on which SpecTcl is running.
 * @param rest    - REST port number.
 * @param mirrors - mirrors currently being maintained.
 * @param size    - Spectrum memory size.
 * @return void*  - Pointer to the shared memory.
 * @retval nullptr - If local mapping is not possible (mirror needs to be setup).
 *
 *  In windows we, we don't worry about multiple clients; we just get data to a local soup:
 * 
 */
static void*
getMirrorIfLocal(
    const char* host, int rest,
    const std::vector<MirrorInfo>& mirrors, size_t size
)
{
    for (auto item : runningMirrors) {
        if ((item.s_hostName == host) && (item.s_restPort == rest)) {
            return item.s_MirrorPointer;
        }
    }
    return nullptr;
}

/**
 * startMirroring
 *    -  Run the mirrorclient program to start mirroring.
 *    -  Wait a bit to let the mirrorclient produce its shared memory.
 *    -  Get mirror information and use getMirrorIfLocal to map to it.
 *       This bit of waiting and mapping can be repeated a few times.
 * @param host - host in which SpecTcl is running.
 * @param mirror - Port on which the SpecTcl mirror server is listening.
 * @param rest   - Port on which the SpecTcl REST server is listening.
 * @param size   - Spectrum bytes.
 * @return void*   - Pointer to specTcl mirrored memory.
 * @retval nullptr - if we can't do all this stuff.
 * 
 * STUB STUB STUB
 */
void*
startMirroring(const char* host, int mirror, int rest, size_t size)
{
    // First we need a pot of memory for the spectra.  This is just a pile of bytes:

    char* result = new char[size];
    std::thread* pThread(0);
    if (result) {
        // Get the initial contents and start a thread to do updates:

        MirrorClient* pClient(0);
        try {
            std::string shost(host);
            pClient = new MirrorClient(shost, mirror, result);

            // Now start an update thread:
            pThread = new std::thread(MirrorThread, pClient, UPDATE_INTERVAL);
            Sleep(1000);               // Wait for the mirror to populate first.
        } 
        catch (...) {
            delete[]result;
            delete pClient;
            return nullptr;
        }

    }
    // Enter the mirror in our directory:

    RunningMirrorInfo info;
    info.s_hostName = host;
    info.s_restPort = rest;
    info.s_MirrorPointer = result;
    runningMirrors.push_back(info);


    return result;
}
#endif


// Target independent code:

/**
 * translatePort
 *    Static function to take a port number and translate it:
 *
 *  @param host  - Host on which the translation is done (see port argument)
 *  @param port  - Port to translate.
 *  @param user  - User that's advertised the port if so.
 *  @param status - Status to set in lastError  on failure.
 *  @return int  - port number
 *  @throws std::runtime_error if not able to translate.
 *  Translation is as follows:
 *  -   If the port is numerical, It is converted to an integer and returned.
 *  -   If not, the host, user and port string are used to attempt to do
 *      a translation via the DAQ port manager in that system and the
 *      result is returned.
 */
static int
translatePort(const char* host, const char* port, const char* user, unsigned status)
{
    // Try to convert the string to an integer.
    char* endptr;
    unsigned result = strtoul(port, &endptr, 0);
    if (endptr != port) {
        // successful conversion:

        lastError = MIRROR_SUCCESS;
        return result;
    }
    // Use the port manager.

    lastError = status;             // LookupPort throws:
    result = LookupPort(host, port, user);
    lastError = MIRROR_SUCCESS;     // NO exception thrown.
    return result;
}

/**
*   getSpecTclMemory
*     Return a pointer to a SpecTcl mirror memory.
*     @param host - host on which SpecTcl is running.
*     @param rest - Port on which rest server is running.  If this can be
*                   translated to a number it's treated as a numeric port number.
*                   If not, we interact with the NSCLDAQ port manager in host to
*                   resolve the port number.
*     @param mirror - Mirror port, treated identically to rest.
*     @param user   - If not null, this is the user running SpecTcl otherwise
*                     the current user is used.  This is noly important
*                     for service name translations.
*     
*/
extern "C" {
EXPORT void*
getSpecTclMemory(const char* host, const char* rest, const char* mirror, const char*user)
{
    DEBUG("getSpecTclMemory");
    if (!user) {
        user = getlogin();
        if (!user) {
            DEBUG("Failed to get username");
            lastError = MIRROR_CANTGETUSERNAME;
            return nullptr;
        }
    }
    int restPort, mirrorPort;
    try {
        restPort = translatePort(host, rest, user, MIRROR_NORESTSVC);
        mirrorPort = translatePort(host, mirror, user, MIRROR_NOMIRRORSVC);
    }
    catch (...) {
        return nullptr;             // translatePort returns the
    }
    DEBUG("Translated the ports");
    
    // Now that the ports are numeric, we can get the memory size and
    // see if there's already a local mirror:
    
    size_t spectrumBytes;
    try {
        DEBUG("asking for spectrum size");
        spectrumBytes = GetSpectrumSize(host, restPort);
    }
    catch (std::exception& e) {
        lastError = MIRROR_CANTGETSIZE;
        return nullptr;
    }
    catch (...) {
        DEBUG("UNanticipated exception in getting mirror size");
        lastError = MIRROR_CANTGETSIZE;
        return nullptr;
    }
    std::vector<MirrorInfo> mirrors;
    try {
        mirrors = GetMirrorList(host, restPort);
    }
    catch (...) {
        lastError = MIRROR_CANTGETMIRRORS;
        return nullptr;
    }

    void* result =  getMirrorIfLocal(host, restPort, mirrors, spectrumBytes);    // If local map 
    if (result) return result;
   
    try {
        auto p = startMirroring(host, mirrorPort, restPort, spectrumBytes);
        return p;
    } catch(...) {
        lastError = MIRROR_SETUPFAILED;
    }
    return nullptr;
}

}









static const char* ppMessages[] = {
    "Successful completion",
    "The specified REST service name is not advertised in that host",
    "The specified MIRROR service name is not advertised in that host",
    "Unable to get the name of the logged in user",
    "Unable to retrieve memory size",
    "Unable to retrieve the list of existing mirrors.",
    "Unable to set up the mirror client",
    "gethostname failed"

};

static const unsigned nMsgs = sizeof(ppMessages) / sizeof(const char*);

extern "C" {
    EXPORT int
        Mirror_errorCode()
    {
        return lastError;
    }
}

extern "C" {
    EXPORT const char*
        Mirror_errorString(unsigned code)
    {
        if (code < nMsgs) {
            return ppMessages[lastError];
        }
        else {
            return "Invalid error code";
        }
    }
}

