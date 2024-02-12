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

/** @file: MirrorMessages.h
 *  @brief: Provide definitions of the memory mirror message types.
 */
#ifndef MIRRORMESSAGES_H
#define MIRRORMESSAGES_H
#include <stdint.h>

// Message types:

namespace Mirror {

    /** Message types:
    *    - MSG_TYPE_SHMINFO client -> server memory key.
    *    - MSG_TYPE_REQUEST_UPDATE client -> server - ask for new data.
    *    - MSG_TYPE_FULL_UPDATE server -> client - used contents of shared memory.
    *    - MSG_TYPE_PARTIAL_UPDATE server -> client - only used spectrum memory.
    */

    static const uint32_t MSG_TYPE_SHMINFO = 1;
    static const uint32_t MSG_TYPE_REQUEST_UPDATE = 2;
    static const uint32_t MSG_TYPE_FULL_UPDATE = 3;
    static const uint32_t MSG_TYPE_PARTIAL_UPDATE = 4;

    struct MessageHeader {
        uint32_t s_messageSize;
        uint32_t s_messageType;
    };
    
    typedef char MemoryKey[4];
    
    
}                                // Namespace Mirror.


#endif