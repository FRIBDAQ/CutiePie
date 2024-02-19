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

/** @file:  MirrorCommand.cpp
 *  @brief:  Implement the Mirror::mirror command.
 */
#include "MirrorCommand.h"
#include "MirrorClient-linux.h"
#include "xamineDataTypes.h"
#include <TCLInterpreter.h>
#include <TCLObject.h>
#include <Exception.h>
#include <ErrnoException.h>
#include <stdexcept>
#include <string.h>

using namespace Xamine;

/**
 * constructor
 *   @param interp - interpreter on which the Mirror::mirror command will be
 *                    registered.
 */
MirrorCommand::MirrorCommand(CTCLInterpreter& interp) :
    CTCLObjectProcessor(interp, "Mirror::mirror", TCLPLUS::kfTRUE),
    m_mirrorMemory(0),
    m_pClient(0)
{}
/**
 * destructor:
 *    Clean up.
 */
MirrorCommand::~MirrorCommand()
{
    delete m_pClient;          // Remove mirror from SpecTcl's dictionary.
    m_pClient = nullptr;
    if (m_mirrorMemory) {
        Xamine_DetachSharedMemory();
        m_mirrorMemory = nullptr;
        Xamine_KillSharedMemory();
    }
    Tcl_DeleteExitHandler(MirrorCommand::exitHandler, this);
}
/**
 * operator()
 *   Gains control when the base command is entered or found in a script.
 *   - setup  exception based error handling/reporting.
 *   - Ensure there is at least a subcommand.
 *   - Dispatch dependingon the actual subcommand and report invalid subcommands.
 * @param interp - interpreter running the command.
 * @param objv   - Command words (including the verb).
 * @return int   - TCL_OK On success, TCL_ERROR on failure.
 */
int
MirrorCommand::operator()(CTCLInterpreter& interp, std::vector<CTCLObject>& objv)
{
    bindAll(interp, objv);
    try {
        requireAtLeast(objv, 2, "Usage: Mirror::mirror subcommand ?...?");
        std::string subcommand = objv[1];
        if (subcommand == "create") {
            create(interp, objv);
        } else if (subcommand == "update") {
            update(interp, objv);
        } else if (subcommand == "destroy") {
            destroy(interp, objv);
        } else {
            std::string emsg(subcommand);
            emsg += " - invalid subcommand";
            throw emsg;
        }
    }
    catch (std::string msg) {
        interp.setResult(msg);
        return TCL_ERROR;
    }
    catch (CException& e) {
        interp.setResult(e.ReasonText());
        return  TCL_ERROR;
    }
    catch (std::exception& e) {
        interp.setResult(e.what());
        return TCL_ERROR;
    }
    catch (...) {
        interp.setResult("MirrorCommand::operator() - unanticipated exceptiontype");
        return TCL_ERROR;
    }
    return TCL_OK;  
}
//////////////////////////////////////////////////////////////////////////////
// Command handlers.

/**
 * create
 *   - Create the mirror memory
 *   - Create the mirror client.
 * @param interp  - interpreter on which the command is running.
 * @param objv    - Command parameters these must be:
 *            - [0] - THe verb.
 *            - [1] - 'create'
 *            - [2] - Spectrum bytes in shared memory.
 *            - [3] - Host running SpecTcl.
 *            - [4] - Port on which the mirror server accepts connections.
 * @note On success, the result is the shared memory key name.
 */
void
MirrorCommand::create(CTCLInterpreter& interp, std::vector<CTCLObject>& objv)
{
    if (m_mirrorMemory || m_pClient) {
        throw std::string(
            "It looks like the mirror memory and/or client are already set up."
        );
    }
    
    requireExactly(objv, 5, "Mirror::mirror create specbytes host port");
    
    int specBytes = objv[2];
    std::string host = objv[3];
    int port = objv[4];
    
    char key[5];
    memset(key, 0, sizeof(key));          // So the key has a trailing null.
    
    if (!Xamine_CreateSharedMemory(specBytes, &m_mirrorMemory)) {
        throw CErrnoException("Unable to create shared memory mirror");       
    }
    Xamine_GetMemoryName(key);             // For the client creation.
    
    try {
        m_pClient = new MirrorClient;
        m_pClient->connect(host.c_str(), port, key);
        
    }
    catch(...) {
        // Cleanup uf we cikd bit establish our client-ness:
        
        Xamine_DetachSharedMemory();
        m_mirrorMemory = nullptr;
        throw;
    }
    // Set up an exit handler to clean this all up:
    
    Tcl_CreateExitHandler(MirrorCommand::exitHandler, this);
    
    // Set the key as the result.
    interp.setResult(key);
}
/**
 * update
 *     Update the contents of the mirror memory.
 *  @param interp - interpreter running the command.
 *  @param objv   - Command words.
 */
void
MirrorCommand::update(CTCLInterpreter& interp, std::vector<CTCLObject>& objv)
{
    throwIfNotSetup();
    
    // Try the update.  If it fails, kill off the client and detach the memory.
    
    try {
        m_pClient->update(const_cast<Xamine_shared*>(m_mirrorMemory));
    }
    catch (...) {
        Xamine_DetachSharedMemory();
        m_mirrorMemory = nullptr;
        try {
            delete m_pClient;
        }
        catch (...) {}
        m_pClient = nullptr;
        throw;
    }
}
/**
 * destroy
 *    Destroy the client and unmap the shared memory.
 *  @param interp - interpreter running the command.
 *  @param objv   - Command words.
 */
void
MirrorCommand::destroy(CTCLInterpreter& interp, std::vector<CTCLObject>& objv)
{
    throwIfNotSetup();
    Xamine_KillSharedMemory();
    m_mirrorMemory = nullptr;
    try {
        delete m_pClient;    
    } catch(...) {}
    m_pClient = nullptr;
    
    // Kill off the exit hander:
    
    Tcl_DeleteExitHandler(MirrorCommand::exitHandler, this);
}
/////////////////////////////////////////////////////////////////////////////
// utilities

/**
 * throwIfNotSetup
 *    @throw std::string - if the memory or client are not set up.
 */
void
MirrorCommand::throwIfNotSetup()
{
    if ((!m_pClient) || (!m_mirrorMemory)) {
        throw std::string("Either the client or the mirror memory are not setup");
    }
    
   
}
/**
 * exitHandler
 *    This is establshed by MirrorCommand::destroy and killed off both by
 *    MirrorCommand::destroy and the destructor.
 *    We're called on exit.  If mirroring is active we shut it down and
 *    by closing the socket. and unmapping the memory.  We
 *    we don't destroy the object that'll happen naturally enough.
 * @param obj - Actually a pointer to the command object.
 */
void
MirrorCommand::exitHandler(ClientData obj)
{
    MirrorCommand* p = reinterpret_cast<MirrorCommand*>(obj);
    if (p->m_mirrorMemory) {
        Xamine_KillSharedMemory();
        p->m_mirrorMemory = nullptr;
    }
    if (p->m_pClient) {
        
        try {
            delete p->m_pClient;
        } catch(...) {}
        p->m_pClient;
    }
}
