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

/** @file:  MirrorClient.cpp
 *  @brief: Implement the SpecTcl mirror client protocol.
 */
#include "MirrorClient-linux.h"
#include <CSocket.h>
#include <MirrorMessages.h>
#include <string>
#include <sstream>
#include "xamineDataTypes.h"
#include <string.h>

using namespace Xamine;

/**
 * MirrorClient - constructor
 *   Null out the socket pointer.
 */
MirrorClient::MirrorClient() :
    m_pSocket(nullptr)
{
    
}
/**
 * MirrorClient - destructor
 *   If the socket exists, destroy it.  That closes it if it is open.
 */
MirrorClient::~MirrorClient()
{
    if (m_pSocket) {
        try {
            m_pSocket->Shutdown();
        } catch(...) {}
        delete m_pSocket;
    }
}

/**
 * connect
 *   Connects to the server and, optionally, sends a memory description message.
 *   this message should only be sent if the memory used to hold updated
 *   spectra will be shared.  In that case it should unambiguously identify
 *   the shared memory region to other local clients.  Note that at present,
 *   we only support 4 byte SYS-V shared memory keys.
 *
 *  @param host  - host to which we connect (must run a mirror server).
 *  @param port  - numerical port on which the server is running. If this is
 *                 advertised in the NSCLDAQ port manager it's up to the client
 *                 to interact to translate the service name/user into a numeric
 *                 port.
 *  @param key  - If provided, this must be a pointer to a 4 byte SYS-V shared
 *                memory key and a message will be sent to the server to
 *                indicate it is mirroring into that shared memory.
 *  @note all errors are reported via exceptions and typically the correct
 *        action is to report the error and destroy this object.
 */
void
MirrorClient::connect(const char* host, int port, const char* key)
{
    std::stringstream sport;
    sport << port;
    std::string strPort = sport.str();
    std::string strHost(host);
    
    m_pSocket = new CSocket;
    m_pSocket->Connect(strHost, strPort);
    
    if (key) {
        sendKeyInfo(key);
    }
}
/**
 * update
 *   Send an update request and get the updated data.
 *
 *  @param pMemory - pointer to an Xamine_shared memory block that will be updated.
 *  @return  bool - true if full update is done else false.
 */
bool
MirrorClient::update(void* pMemory)
{
    std::pair<bool, size_t> info =  requestUpdate();
    bool result     = info.first;
    size_t nBytes   = info.second;
    
    // For a full update we just read nBytes into pMemory
    // For a partial, we need to read it into the spectrum soup part of that
    // storage:
    
    if (result) {
        m_pSocket->Read(pMemory, nBytes);
    } else {
        Xamine_shared* pXamine = reinterpret_cast<Xamine_shared*>(pMemory);
        m_pSocket->Read(pXamine->dsp_spectra.XAMINE_b, nBytes);
    }
    
    return result;
}
////////////////////////////////////////////////////////////////////////
// Private utilities:

/**
 * sendKeyInfo
 *   Send the server a Mirror::MSG_TYPE_SHMINFO message. No reply
 *   is expected back from the server.
 * @param key - 4 character key.
 */
void
MirrorClient::sendKeyInfo(const char* key)
{
    struct {
        Mirror::MessageHeader hdr;
        Mirror::MemoryKey     body;
    } message;
    
    message.hdr.s_messageType = Mirror::MSG_TYPE_SHMINFO;
    message.hdr.s_messageSize = sizeof(message);
    memcpy(message.body, key, 4);
    
    m_pSocket->Write(&message, sizeof(message));
}
/**
 * requestUpdate
 *    Requests an update of the contents of the mirror server's spectrum
 *    memory.
 * @return std::pair<bool, size_t>  .first is a flag that, if true,
 *            indicates the server replied with a Mirror::MSG_TYPE_FULL_UPDATE
 *            if false, the server replied with a Mirror::MSG_TYPE_PARTIAL_UPDATE
 * @note  The payload of the server message is left for the caller to read into the
 *  appropriate chunk of memory
 */
std::pair<bool, size_t>
MirrorClient::requestUpdate()
{
    Mirror::MessageHeader req;
    req.s_messageSize = sizeof(req);
    req.s_messageType = Mirror::MSG_TYPE_REQUEST_UPDATE;
    
    m_pSocket->Write(&req, sizeof(req));
    
    Mirror::MessageHeader rep;
    m_pSocket->Read(&rep, sizeof(rep));
    
    std::pair<bool, size_t> result;
    
    result.first = rep.s_messageType == Mirror::MSG_TYPE_FULL_UPDATE;
    result.second = rep.s_messageSize - sizeof(rep);   // Remaining bytes.
    
    return result;
}
