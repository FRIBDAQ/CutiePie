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

/** 
*   Implementation of the mirror client class.
* 
*/
#include "MirrorClient.h"
#include "CSocket.h"
#include "Exception.h"
#include "xamineDataTypes.h"

#include <stdexcept>
#include <sstream>
#include <iostream>



/**
*   constructor - just saves what we need:
* 
* @param host - host that is running the mirror server. 
*               since we use CSocket this can be a DNS name
*               or a dotted IP.
* @param port - The port on which the mirror server is listening.
* @param pStorage - Pointer to storage the caller has allocated 
*               into which the mirroring is done.
*/
MirrorClient::MirrorClient(const std::string& host, unsigned short port, char* pStorage) :
    m_pMirrorStorage(pStorage),
    m_pSocket(nullptr),
    m_hostname(host),
    m_port(port)
{}

/**
* destructor.
*   Assumption : pStorage and m_pSocket can be targets of delete.
*/
MirrorClient::~MirrorClient() {
    delete[]m_pMirrorStorage;
    delete m_pSocket;
}
/**
*   initialize
*     -   Create a socket.
*     -   Connect to the mirror server.
*     -   INvoke update.
* 
* @note If we catch an exception, we set our state to
*       uninitialized (nullptr for socket having deleted
*       any socket we might have created.
* 
* @throws
*    * std::runtime_error - If the connection fails.
*/
void MirrorClient::initialize() {
    m_pSocket = new CSocket();
    std::stringstream portS;
    portS << m_port;
    auto portname = portS.str();
    try {
        m_pSocket->Connect(m_hostname, portname);
        sendKey(0);
        update();
    }
    catch (CException& e) {
        delete m_pSocket;
        m_pSocket = nullptr;
        throw std::runtime_error(e.ReasonText());
    }
    catch (std::exception& e) {
        delete m_pSocket;
        m_pSocket = nullptr;
        throw e;
    }
    catch (...) {
        delete m_pSocket;
        throw;
    }
}

/**
*    Update the mirror:
*     - Send the update request.
*     - Read the reply header.
*     - If the reply header is full update just read the 
*         Rest of the data into m_pMirrorStorage.
*     - If the reply header is a partial update, read the rest of the
*         message into storage past the Xamine_Header.
*  @throws:
*     *  std::logic_error - we don't have a connected socket.
*     *  std::runtime_error - something bad happend in the client/server interaction.
*/
void MirrorClient::update() {
    // We need a connected socket:

    if (m_pSocket) {
        if (m_pSocket->getState() == CSocket::Connected) {

        }
        else {
            // So initialize doesn't leak.
            delete m_pSocket;
            m_pSocket = nullptr;
            throw std::logic_error("The client socket evidently is not connected");
        }
        try {
            sendUpdateRequest();
            auto hdr = readResponseHeader();
            void* pDest(m_pMirrorStorage);
            size_t nBytes = hdr.s_messageSize - sizeof(hdr);
            if (nBytes > 0) {
                if (hdr.s_messageType == Mirror::MSG_TYPE_FULL_UPDATE) {
                    pDest = m_pMirrorStorage;
                }
                else if (hdr.s_messageType == Mirror::MSG_TYPE_PARTIAL_UPDATE) {
                    pDest = m_pMirrorStorage + sizeof(Xamine::Xamine_Header);
                }
                // Remember that s_messageSize inludes the header:
                m_pSocket->Read(pDest, hdr.s_messageSize - sizeof(hdr));
            }
        }
        catch (std::exception& e) {
            throw e;
        }
        catch (CException& e) {
            throw std::runtime_error(e.ReasonText());
        }
    }
    else {
        throw std::logic_error("You must successfully run the initialize() method before updating");
    }
    
}   

/**
*   Send a memory message key to the server.
*  
* @param key - SYSV key to send.
* 
* @todo should this be public and called by the class client?
*/
void MirrorClient::sendKey(int key) {
    
    struct KeyMsg {
        Mirror::MessageHeader s_hdr;
        uint32_t key;                  // Bit of a cheat.
    } msg;
    msg.s_hdr.s_messageSize = sizeof(msg);
    msg.s_hdr.s_messageType = Mirror::MSG_TYPE_SHMINFO;
    msg.key = key;

    try {
        m_pSocket->Write(&msg, sizeof(msg));
        
    } catch (CException& e) {
        throw std::runtime_error(e.ReasonText());
    }

}
/**
*    send and update request to the mirror server.  This triggers
* it to respond with data from which to populate the mirror.
* 
* @throws std::runtime_error if the send fails.
*/
void MirrorClient::sendUpdateRequest() {
    Mirror::MessageHeader hdr;
    hdr.s_messageSize = sizeof(hdr);
    hdr.s_messageType = Mirror::MSG_TYPE_REQUEST_UPDATE;
    
    try {
        m_pSocket->Write(&hdr, sizeof(hdr));
    }
    catch (CException& e) {
        throw std::runtime_error(e.ReasonText());
    }
}

/**
*   Read the header of response from the mirror client Note that
* the server only replies to update requests.
* 
* @throw std::runtime_error if the read fails.
* @note if an exception is thrown you should not rely on the header returned.
*/
Mirror::MessageHeader MirrorClient::readResponseHeader() {

    Mirror::MessageHeader hdr;
    try {
        m_pSocket->Read(&hdr, sizeof(hdr));
    }
    catch (CException& e) {
        throw std::runtime_error(e.ReasonText());
    }
    return hdr;
}
