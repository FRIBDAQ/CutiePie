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

/** @file:  CPyConverter.h
 *  @brief: Class for creation of Python extension module
 */

#ifndef CPYCONVERTER_H
#define CPYCONVERTER_H
#include <Python.h>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include "dataAccess.h"
#include "SpecTclMirrorClient.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// CPyConverter is a class defined to provide the defintion for a python extension in QtPy. SIP is a collection of tools that makes it
// very easy to create Python bindings for C and C++ libraries. SIP comprises a set of build tools and a sip module. The build tools process
// a set of specification files and generates C or C++ code which is then compiled to create the bindings extension module. See PyQtGUI/sip
// for the implementation.
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class CPyConverter {

 public:
  CPyConverter();
  virtual ~CPyConverter();

  std::vector<int> m_id;  
  std::vector<std::string> m_names;
  std::vector<int> m_dim;
  std::vector<int> m_binx;
  std::vector<float> m_minx;
  std::vector<float> m_maxx;
  std::vector<int> m_biny;      
  std::vector<float> m_miny;      
  std::vector<float> m_maxy;        
  std::vector<int> m_types;        

  void extractInfo(char* speclist);
  PyObject* getShMem(char* hostname, char* port, char* mirror, char* user, bool reset);
  PyObject* ShMemToNpArray(void* addr, int dim, int nbinx, int nbiny, int type);

  PyObject* vectorToList_Int(const std::vector<int> &data);  
  PyObject* vectorToList_Float(const std::vector<float> &data);
  PyObject* vectorToList_String(const std::vector<std::string> &data);  
  PyObject* DebugFillSpectra();


  
};

#endif
