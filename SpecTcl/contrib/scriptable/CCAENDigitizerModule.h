//! \class: CCAENDigitizerModule           
//! \file:  .h
// Author:
//   Ron Fox
//   NSCL
//   Michigan State University
//   East Lansing, MI 48824-1321
//   mailto:fox@nscl.msu.edu
//
// Copyright 

#ifndef __CCAENDIGITIZERMODULE_H  //Required for current class
#define __CCAENDIGITIZERMODULE_H

//
// Include files:
//

                               //Required for base classes
#ifndef __CMODULE_H     //CModule
#include "CModule.h"
#endif

#ifndef __STL_STRING
#include <string>
#define __STL_STRING
#endif

#ifndef __CRTL_STRING_H
#include <string.h>
#define __CRTL_STRING_H
#endif

#ifndef __TRANSLATORPOINTER_H
#include <TranslatorPointer.h>
#endif

// Forward Class Defintions:

class CTCLInterpreter;
class CTCLResult;
class CAnalyzer;
class CBufferDecoder;
class CEvent;
 
/*!
Unpacks data from a CAEN digitizer.
*/
class CCAENDigitizerModule  : public CModule        
{
private:
  
  // Private Member data:
    int m_aParameterMap[32];  //!  Maps channels to spectcl parameter ids.  
    int m_nCrate;                   //!  Crate number (from crate register).  
    int m_nSlot;                     //!  Slot number from GEO address or register.  
    CIntConfigParam* m_pCrateConfig;	//!< Pointer to "crate" parameter
    CIntConfigParam* m_pSlotConfig;    	//!< Pointer to "slot" parameter.
    CStringArrayparam* m_pParamConfig;  //!< Pointer to "parameters" param.
   
public:
   // Constructors and other canonical functions
    CCAENDigitizerModule (CTCLInterpreter& rInterp, 
				    const string& rName);
    virtual ~ CCAENDigitizerModule ( );
private:
 
    CCAENDigitizerModule (const CCAENDigitizerModule& aCCAENDigitizerModule );
    CCAENDigitizerModule& operator= (const CCAENDigitizerModule& aCCAENDigitizerModule);
    int operator== (const CCAENDigitizerModule& aCCAENDigitizerModule) const;
    int operator!=(const CCAENDigitizerModule& rhs) const;
public:
// Selectors:

public:

          //Get accessor function for non-static attribute data member
  const int* getParameterMap() const
  { 
       return m_aParameterMap;
  }  
            //Get accessor function for non-static attribute data member
  int getCrate() const
  { 
      return m_nCrate;
  }  
            //Get accessor function for non-static attribute data member
  int getSlot() const
  { 
      return m_nSlot;
  }   

// Attribute mutators:

protected:

          //Set accessor function for non-static attribute data member
  void setParameterMap (const int* am_aParameterMap) 
  { 
    memcpy(m_aParameterMap, am_aParameterMap, 
		sizeof(m_aParameterMap));
  }  
            //Set accessor function for non-static attribute data member
  void setCrate (const int am_nCrate)
  { m_nCrate = am_nCrate;
  }  
            //Set accessor function for non-static attribute data member
  void setSlot (const int am_nSlot)
  { m_nSlot = am_nSlot;
  }   

  // Class operations:

public:
  virtual  string getType() const;
  virtual   void Setup (CAnalyzer& rAnalyzer, CHistogrammer& rHistogrammer)   ; // 
  virtual   TranslatorPointer<UShort_t> 
    Unpack (TranslatorPointer<UShort_t> pEvent, 
	    CEvent& rEvent, 
	    CAnalyzer& rAnalyzer, CBufferDecoder& rDecoder)   ; // 
  
};

#endif