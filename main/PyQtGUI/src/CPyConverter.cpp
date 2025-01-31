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

/** @file:  CPyConverter.cpp
 *  @brief: Class for creation of Python extension module
 */

#include "CPyConverter.h"
#include <stdexcept>
#include <sys/stat.h>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include "dataRetriever.h"
#include "dataTypes.h"
#include "dataAccess.h"
#include <numpy/arrayobject.h>
#include <memory>

bool debug = true;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// CPyConverter is a class defined to provide the defintion for a python extension in QtPy. SIP is a collection of tools that makes it
// very easy to create Python bindings for C and C++ libraries. SIP comprises a set of build tools and a sip module. The build tools process
// a set of specification files and generates C or C++ code which is then compiled to create the bindings extension module. See PyQtGUI/sip
// for the implementation.
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


CPyConverter::CPyConverter()
{}

CPyConverter::~CPyConverter()
{}

/////////////////////////////////////////////////////////////////////////////////////////////
//
// Formatting of the spectra information for the creation of PyObject. Each std::vector
// will the be finally converted into PyTuple
//
/////////////////////////////////////////////////////////////////////////////////////////////

void
CPyConverter::extractInfo(char* speclist)
{
  std::stringstream ss;
  int id, dim, binx, biny, type;
  float minx, miny, maxx, maxy;
  std::string name;
  ss << std::string(speclist);  
  while (!ss.eof() &&
	 (ss >> id >> name >> type >> dim >> binx >> minx >> maxx >> biny >> miny >> maxy)) {
    if (debug)
      std::cout << id << " " << name << ": " << speclist << std::endl;
    m_id.push_back(id);
    m_names.push_back(name);
    m_types.push_back(type);
    m_dim.push_back(dim);
    m_binx.push_back(binx);
    m_biny.push_back(biny);
    m_minx.push_back(minx);
    m_miny.push_back(miny);      
    m_maxx.push_back(maxx);
    m_maxy.push_back(maxy);     
  }    
}

/////////////////////////////////////////////////////////////////////////////////////////////
//
// Access function for the shared memory. This function is associated to the "GetData" signal
// of the QtPy GUI. The functions utilizes the SpecTclMirrorClient API
//
//   getSpecTclMemory
//     Return a pointer to a SpecTcl mirror memory.
//     @param host - host on which SpecTcl is running.
//     @param rest - Port on which rest server is running.  If this can be
//                   translated to a number it's treated as a numeric port number.
//                   If not, we interact with the NSCLDAQ port manager in host to
//                   resolve the port number.
//     @param mirror - Mirror port, treated identically to rest.
//     @param user   - If not null, this is the user running SpecTcl otherwise
//                     the current user is used.  This is noly important
//                     for service name translations.
//
/////////////////////////////////////////////////////////////////////////////////////////////

PyObject*
CPyConverter::getShMem(char* hostname, char* port, char* mirror, char* user, bool reset=false)
{
  if (debug){
    std::cout << "Inside CPyConverter::getShMem()" << std::endl;
    std::cout << "hostname: " << hostname << " port: " << port << " mirror: " << mirror << " user: " << user << std::endl;
  }
  
  std::string _hostname = hostname;
  std::string _port = port;  
  std::string _mirror = mirror;
  std::string _user = user;    
  
  if (debug){
    std::cout << "Hostname --> " << _hostname << " Port --> " << _port << std::endl;
    std::cout << "Mirror --> " << _mirror << " User --> " << _user << std::endl;    
  }

  if (dataRetriever::getInstance() != nullptr && reset) {
    dataRetriever::getInstance()->deleteInstance();
  }
  dataRetriever* d = dataRetriever::getInstance();  
  if (debug) {
      std::cout << "About to get shared memory...\n";
  }
 
  // The pointer won't move around so we only need to do this if the memory point 
  // has not been set...and doing this twice on windows is fatal.

  if (!d->GetShMem()) {
      if (debug) {
          std::cout << "Need to get the mirror pointer:\n";
      }
      spec_shared* p = reinterpret_cast<spec_shared*>(getSpecTclMemory(_hostname.c_str(), _port.c_str(), _mirror.c_str(), _user.c_str()));
      if (debug) {
          std::cout << " got " << std::hex << p << std::dec << std::endl;
      }
      d->SetShMem(p);
  }
  
  spec_shared* p = d->GetShMem();
  
  
  // Create list of spectra
  char **speclist;
  int lsize;
  lsize = p->GetSpectrumList(&speclist);

  Address_t addr;
  

  // PyObject* data[lsize];
  PyObject** data(new PyObject*[lsize]);
  PyObject* listData = PyList_New(lsize);
  
for (int i = 0; i < lsize; i++){
    extractInfo(speclist[i]);
    // access histogram of id i
    addr = p->CreateSpectrum(m_id.at(i));
    // convert memory to np array
    data[i] = ShMemToNpArray(addr, m_dim[i], m_binx[i], m_biny[i], m_types[i]);
    // add nparray to a list
    PyList_SetItem(listData, i, data[i]);
  }

  PyObject* result = PyTuple_New(10);
  PyTuple_SetItem(result, 0, vectorToList_Int(m_id));
  PyTuple_SetItem(result, 1, vectorToList_String(m_names));
  PyTuple_SetItem(result, 2, vectorToList_Int(m_dim));  
  PyTuple_SetItem(result, 3, vectorToList_Int(m_binx));
  PyTuple_SetItem(result, 4, vectorToList_Float(m_minx));
  PyTuple_SetItem(result, 5, vectorToList_Float(m_maxx));
  PyTuple_SetItem(result, 6, vectorToList_Int(m_biny));
  PyTuple_SetItem(result, 7, vectorToList_Float(m_miny));
  PyTuple_SetItem(result, 8, vectorToList_Float(m_maxy));
  PyTuple_SetItem(result, 9, listData);
  // PyTuple_SetItem(result, 10, vectorToList_Int(m_types));
  delete[]data;
  
  return result;
  
}


PyObject*
CPyConverter::ShMemToNpArray(void* addr, int dim, int nbinx, int nbiny, int type)
{
  PyArrayObject* data;
  import_array();
  npy_intp specType = {type};

  if (dim == 1){
    npy_intp dims = {nbinx};
    if (specType == _onedlong){
      data = (PyArrayObject*)PyArray_SimpleNewFromData(dim, &dims, NPY_INT, addr);
    }
    else if (specType == _onedword){
      data = (PyArrayObject*)PyArray_SimpleNewFromData(dim, &dims, NPY_SHORT, addr);
    }
  }
  else {
    npy_intp dims[2] = {nbiny, nbinx};
    if (specType == _twodlong){
      data = (PyArrayObject*)PyArray_SimpleNewFromData(dim, dims, NPY_INT, addr);
    }
    else if (specType == _twodword){
      data = (PyArrayObject*)PyArray_SimpleNewFromData(dim, dims, NPY_SHORT, addr);
    }
    else if (specType == _twodbyte){
      data = (PyArrayObject*)PyArray_SimpleNewFromData(dim, dims, NPY_BYTE, addr);
    }
  }

  return (PyObject*)data;
}


PyObject*
CPyConverter::vectorToList_Int(const std::vector<int> &data) {
  PyObject* listObj = PyList_New(data.size() );
  if (!listObj) throw std::logic_error("Unable to allocate memory for Python list");
  for (unsigned int i = 0; i < data.size(); i++) {
    PyObject *num = PyLong_FromLong(data[i]);
    if (!num) {
      Py_DECREF(listObj);
      throw std::logic_error("Unable to allocate memory for Python list");
    }
    PyList_SET_ITEM(listObj, i, num);
  }
  return listObj;
}

PyObject*
CPyConverter::vectorToList_Float(const std::vector<float> &data) {
  PyObject* listObj = PyList_New(data.size() );
  if (!listObj) throw std::logic_error("Unable to allocate memory for Python list");
  for (unsigned int i = 0; i < data.size(); i++) {
    PyObject *num = PyFloat_FromDouble( (double) data[i]);
    if (!num) {
      Py_DECREF(listObj);
      throw std::logic_error("Unable to allocate memory for Python list");
    }
    PyList_SET_ITEM(listObj, i, num);
  }
  return listObj;
}

PyObject*
CPyConverter::vectorToList_String(const std::vector<std::string> &data) {
  PyObject* listObj = PyList_New(data.size() );
  if (!listObj) throw std::logic_error("Unable to allocate memory for Python list");  
  for (unsigned int i = 0; i < data.size(); i++) {
    PyObject* newstring;
    newstring = PyUnicode_DecodeUTF8(data[i].c_str(), data[i].length(), NULL);
    if(newstring == NULL) {
      Py_DECREF(listObj);
      throw std::logic_error("Unable to allocate memory for Python list");      
    }
    PyList_SET_ITEM(listObj, i, newstring);  
  }
  return listObj;
}
