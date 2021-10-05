/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2014.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Authors:
             Ron Fox
             Jeromy Tompkins 
	     NSCL
	     Michigan State University
	     East Lansing, MI 48824-1321
*/


// Class: CApplyCommand                     //ANSI C++
//
// Implements the apply command.
// apply applies a single gate to a set of spectra.
// The syntax of this command is:
//    apply gatename spectrum1 [spectrum2 ...]
//        to apply a gate to several spectra or:
//    apply -list  spectrum [spectrum2...]
//        to list the gate applied on each of the selected
//       spectra.
//  Note that spectra can only have a single gate applied.
//  Applying a gate to a spectrum which is gated, replaces that
//  spectrum's gate... Since gates can be arbitrarily complex entities,
//  in practice, this is not a problem.
//
     
//
// Author:
//     Ron Fox
//     NSCL
//     Michigan State University
//     East Lansing, MI 48824-1321
//     mailto: fox@nscl.msu.edu
// 
// (c) Copyright NSCL 1999, All rights reserved ApplyCommand.h
//

#ifndef APPLYCOMMAND_H  //Required for current class
#define APPLYCOMMAND_H

#include "TCLPackagedCommand.h"
#include <string>
   
class CGatePackage;		// Forward class type definition.
class CTCLInterpreter;		// Forward class type definition
           
class CApplyCommand  : public CTCLPackagedCommand        
{                       


public:

   // Constructor:  

  CApplyCommand (CTCLInterpreter* pInterp, CTCLCommandPackage& rPack, const char* cmd="apply") :
    CTCLPackagedCommand(cmd, pInterp, rPack)
  { 
  } 
  ~CApplyCommand ( ) { }  //Destructor 
  
   //Copy constructor: Illegal

private:
  CApplyCommand (const CApplyCommand& aCApplyCommand );
public:
   // Operator= Assignment Operator: Illegal
private:
  CApplyCommand& operator= (const CApplyCommand& aCApplyCommand);
  int operator==(const CApplyCommand& rhs) const;
  int operator!=(const CApplyCommand& rhs) const;
public:

public:

  virtual   int operator() (CTCLInterpreter& rInterp, 
			    CTCLResult& rResult, int nParams, 
			    char* pParams[])    ;

  TCLPLUS::Int_t ApplyGate (CTCLInterpreter& rInterp, CTCLResult& rResult, 
		   TCLPLUS::UInt_t nArgs, char*  pArgs[])    ;
  
  TCLPLUS::Int_t ListApplications (CTCLInterpreter& rInterp, CTCLResult& rResult,
			  TCLPLUS::UInt_t nArgs, char* pArgs[])    ;

  // Utility functions.
 
protected:
  static std::string Usage();	// Produces usage information.
};

#endif
