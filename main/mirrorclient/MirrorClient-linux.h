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

/** @file:  MirrorClient.h
 *  @brief: Class that provides the messaging for a shared memory mirror client.
 */
#ifndef MIRRORCLIENT_H
#define MIRRORCLIENT_H

#include <utility>
#include <stddef.h>

class CSocket;

/**
 * @class MirrorClient
 *    This class provides a client driven mirror client for the SpecTcl
 *    shared memory mirroring system.  While this is intended to provide
 *    a shared memory mirror it need not do so. The class is policy free
 *    in the sense that it's own client determines where mirrored data goes.
 *    The client is expected to:
 *    - Instantiate this class.
 *    - Request connection to the server (which includes the optional transmission
 *      of a memory information message).
 *    - Request updates, supplying sufficient buffer memory to hold the data.
 *      The assumption is that the client has queried the SpecTcl REST server
 *      about the size of the Spectrum memory and set aside that amount
 *      (in a shared memory or locally) and that this buffer will be used
 *      for all update requests.   The client will indicate if the reply
 *      was a partial or full update.
 *    - When finished destroy the object which, in turn, closes the connection.
 *
 *  This choice of architecture is not only intended to support unit testing
 *  where shared memory is neither needed nor wanted, but unforeseen needs
 *  such as a program somewhere that wants its own internal mirror (e.g.
 *  in the controls world for display memor from the separator SpecTcl(s)).
 */
class MirrorClient
{
private:
    CSocket* m_pSocket;
public:
    MirrorClient();
    virtual ~MirrorClient();
    
    void connect(const char* host, int port, const char* key=nullptr);
    bool update(void* pMemory);
    
    // Utility methods.
private:
    void sendKeyInfo(const char* key);
    std::pair<bool, size_t> requestUpdate();
};

#endif
