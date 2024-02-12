#ifndef MIRRORCLIENT_H
#define MIRRORCLIENT_H
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
#include "MirrorMessages.h"
#include <string>

#ifdef _WIN64
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif


class CSocket;

/**
*  CMirrorClient provides a client to the SpecTcl/Rustogrammer
* display  memory mirroring server.
* in order to support exception generation two stage construction
* is needed.
*   * Construction supplies:
*       -  The host and port of the mirror server.
*       -  A pointer to suitably sized storage to hold the mirror.
*          this storage is assumed to have been created via
*          new char[some-size].  Ownership of this storage passes
*          to the object being constructed.
*   * Initialization:
*       - Connects to the mirror server throwing an exception 
*         on failure (std::runtime_error).
*       - Invokes update to initially populate the storage.
*   
* the **update** method shall request an update from the mirror server
* and update the contents of the mirror.
* 
*  Destruction:
*    *   Closes the connection to the mirror server.
*    *   uses delete to free the storage passed to us.
* 
* As used by the mirror server, the main thread consructs and intializes.
* This ensures the caller gets populated storage.
* A thread is then spawned off that does something like:
* 
* 
*     while(1) {
*       sleep(for-update-interval)
*       mirror.update()
*     }
* 
* where this is surrounded in a try/catch block that emits a nasty
* error if the update failes.
*/
class EXPORT MirrorClient {
private:
	char*       m_pMirrorStorage;           //< Our mirror.
	CSocket*    m_pSocket;                  //< Socket to server.
	std::string m_hostname;                 //< Mirroserver host.
	unsigned short m_port;                  //< Mirror server port.
public:
	MirrorClient(const std::string& host, unsigned short port, char* pStorage);
	virtual ~MirrorClient();    // Maybe this class isn't final (in the JAVA sense).

	void initialize();
	void update();
	void* getMirror();
private:
    void sendKey(int key);          // Send memory key.
    void sendUpdateRequest();
    Mirror::MessageHeader readResponseHeader();
};


#endif