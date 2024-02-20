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

/** @file:  MirrorCommand.h
 *  @brief: Define the command Mirror::mirror command
 */
#ifndef MIRRORCOMMAND_H
#define MIRRORCOMMAND_H

#include <TCLObjectProcessor.h>
#include "xamineDataTypes.h"
#include <tcl.h>

using namespace Xamine;
class MirrorClient;
class CTCLInterpreter;
class CTCLObject;

/**
 *  @class MirrorCommand
 *     Provides the class to create and maintain the mirror.  This implements
 *     a command ensemble with the base name Mirror::mirror
 *     Subcommands:
 *     -   create specbytes serverhost serverport
 *              create the shared memory and mirror client.
 *     -   update - Update the mirror contents from the server.
 *     -   destroy- Destroy the client and detach the mirror memory.
 */
class MirrorCommand : public CTCLObjectProcessor
{
private:
    volatile Xamine_shared* m_mirrorMemory;
    MirrorClient*   m_pClient;
public:
    MirrorCommand(CTCLInterpreter& interp);
    virtual ~MirrorCommand();
    
    virtual int operator()(CTCLInterpreter& interp, std::vector<CTCLObject>& objv);
    
    // subcommand executors:
private:
    void create(CTCLInterpreter& interp, std::vector<CTCLObject>& objv);
    void update(CTCLInterpreter& interp, std::vector<CTCLObject>& objv);
    void destroy(CTCLInterpreter& interp, std::vector<CTCLObject>& objv);
    
    // Utilities:
private:
    void throwIfNotSetup();
    
    static void exitHandler(ClientData obj);
};

#endif
