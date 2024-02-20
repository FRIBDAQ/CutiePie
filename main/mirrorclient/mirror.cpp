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

/** @file:  mirror.cpp
 *  @brief: Entry point for mirror.
 */

/**
 * processes command line parameters,
 * starts up the Tcl intepreter
 * and transfers control to the mirror.tcl startup script.
 */
#include "cmdline.h"
#include <tcl.h>
#include <TCLInterpreter.h>
#include "CmdInfo.h"
#include "MirrorCommand.h"
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <Exception.h>
#include <iostream>
#include "os.h"
const char* TclLibPath = SPECTCL_TCLLIBS;

static struct gengetopt_args_info parsed;

static int AppInit(Tcl_Interp* pInterp)
{
    // Do basic interpreter initialization:
    
    if (Tcl_Init(pInterp) != TCL_OK) {
        return TCL_ERROR;
    }
    
    // Also get an encapulated interpreter:
    
     CTCLInterpreter* pOInterp = new CTCLInterpreter(pInterp);
    
    // Add the TclLibPath for this spectcl to the auto_path:
    
    Tcl_Obj* autopath = Tcl_ObjGetVar2(
        pInterp, Tcl_NewStringObj("auto_path", -1), NULL, TCL_GLOBAL_ONLY
    );
    if (!autopath) {
        pOInterp->setResult("Cant' read auto_path to include SpecTcl directories");
        return TCL_ERROR;
    }
    
    int status = Tcl_ListObjAppendElement(
        pInterp, autopath, Tcl_NewStringObj(TclLibPath, -1)
    );
    if (status != TCL_OK) {
        pOInterp->setResult("Failed to lappend Tcl Library path to auto_path");
    }
    Tcl_ObjSetVar2(
        pInterp, Tcl_NewStringObj("auto_path", -1), NULL, autopath,
        TCL_GLOBAL_ONLY
    );
    
    
    
    // Create the Mirror namespace in which our commands will live:
    
    if (!Tcl_CreateNamespace(pInterp, "Mirror", nullptr, nullptr)) {
        return TCL_ERROR;
    }
    
    // These commands provide script access to the command options.
    
    
    new CmdInfo(*pOInterp, "Mirror::gethost", parsed.host_arg);
    new CmdInfo(*pOInterp, "Mirror::getrestport", parsed.restport_arg);
    new CmdInfo(*pOInterp, "Mirror::getmirrorport", parsed.mirrorport_arg);
    const char* pUser;
    std::string strUser = Os::whoami();
    if (parsed.user_given) {
        pUser = parsed.user_arg;
    } else {
      pUser = strUser.c_str();
        if (!pUser) {
            perror("Unable to get current username use the --user option to supply it.");
            exit(-1);
        }
    }new CmdInfo(*pOInterp, "Mirror::getuser", pUser);
    
    new MirrorCommand(*pOInterp);
    
    //  Set up the mirrorclientscript.tcl in @bindir@ to be our init script:
    
    std::string startScript = TclLibPath;
    startScript += "/../bin/mirrorclientscript.tcl";
   
    Tcl_ObjSetVar2(pInterp, Tcl_NewStringObj("tcl_rcFileName", -1), NULL,
                   Tcl_NewStringObj(startScript.c_str(), -1), TCL_GLOBAL_ONLY);
    
    return TCL_OK;
}


/**
 * main
 *    Program entry point
 *    - Parse the command arguments
 *    - start up the interpreter where we pick up again with AppInit.
 *
 */
int
main(int argc, char** argv)
{
    cmdline_parser(argc, argv, &parsed);
    
    // Start up teh TCLInterpreter
    
    Tcl_Main(argc, argv, AppInit);
}
