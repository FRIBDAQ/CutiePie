/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2021.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Authors:
             Giordano Cerizza
             Ron Fox
             FRIB
             Michigan State University
             East Lansing, MI 48824-1321
*/

/** @file:  dataRetriever.h
 *  @brief: API to set and retrieve shared memory information
 */

#include <string>
#include <vector>
#include "dataAccess.h"
#include "dataTypes.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Test methods (not for public use). Shared memory requests have to go through the mirrorclient.
// These function were thought and deployed BEFORE the SpecTclMirrorClient was even an idea. So please don't use them
// because they fullfil the needs of something now obsolete
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class dataRetriever{
  
 private:
     spec_shared* shmem;

  // Canonical for the singleton class
  dataRetriever() : shmem(0) {};  // Private so that it can  not be called
  dataRetriever(dataRetriever const&){};             // copy constructor is private
  dataRetriever& operator=(dataRetriever const&){ return *this; };  // assignment operator is private
  static dataRetriever* m_pInstance;

  

  // Test methods (not for public use). Shared memory requests have to go through the mirrorclient.
  // These function were thought and deployed BEFORE the SpecTclMirrorClient was even an idea. So please don't use them
  // because they fullfil the needs of something now obsolete

  std::vector<std::string> shm;
  std::string key;
  unsigned int size;

  std::string _hostname;
  std::string _port;

  spec_shared *spectra;
  
 public:

  static dataRetriever* getInstance();
  void SetShMem(spec_shared* p);
  spec_shared* GetShMem();
//  static void PrintOffsets();

  
};
