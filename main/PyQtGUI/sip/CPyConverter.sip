// Define the SIP wrapper to the CPyConverter library.

%Module CPyConverter

class CPyConverter {

%TypeHeaderCode
#include "CPyConverter.h"
#include "SpecTclMirrorClient.h"
#include "dataAccess.h"
#include <Python.h>
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <chrono>
%End

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
   
   void extractInfo(char* speclist);
   PyObject* Update(char* hostname, char* port, char* mirror, char* user);
   PyObject* ShMemToNpArray(void* addr, int size, int nbinx, int nbiny, int type); 

   PyObject* vectorToList_Int(const std::vector<int> &data);
   PyObject* vectorToList_Float(const std::vector<float> &data);
   PyObject* vectorToList_String(const std::vector<std::string> &data);
      
};

// ****************************************************
// Template for std::vector<TYPE>
// ****************************************************

template<TYPE *>
%MappedType std::vector<TYPE *> {
%TypeHeaderCode
#include <vector>
%End

%ConvertFromTypeCode
  PyObject *l;

  const sipTypeDef* kpTypeDef = getSipType("TYPE");

  if (!kpTypeDef) {
    return NULL;
  }

  // Create the Python vector of the correct length.
  if ((l = PyList_New(sipCpp -> size())) == NULL) {
    return NULL;
  }

  int i = 0;
  for(std::vector<TYPE *>::iterator iter = sipCpp->begin(); iter != sipCpp->end(); iter++) {
    TYPE *cpp = *iter;
    PyObject *pobj;

    // Get the Python wrapper for the Type instance, creating a new
    // one if necessary, and handle any ownership transfer.
    if ((pobj = sipConvertFromType(cpp, kpTypeDef, sipTransferObj)) == NULL) {
       // There was an error so garbage collect the Python list.
       Py_XDECREF(l);
       return NULL;
    }

    // Add the wrapper to the list.
    PyList_SET_ITEM(l, i++, pobj);
  }

  // Return the Python list.
  return l;
%End

%ConvertToTypeCode

  const sipTypeDef* kpTypeDef = getSipType("TYPE");

  if (!kpTypeDef) {
     return 0;
  }

  // Check if type is compatible
  if (sipIsErr == NULL) {
    if (!PyList_Check(sipPy)) {
     return 0;
    }
    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(sipPy); ++i) {
      PyObject *item = PyList_GET_ITEM(sipPy, i);
      if (!sipCanConvertToType(item, kpTypeDef, SIP_NOT_NONE)) {
        //printErrorMessage("TypeError : object in list of type " + std::string(item->ob_type->tp_name) + " can not be converted to TYPE");
        return 0;
      }
    }
    return 1;
  }

  // Convert Python list of TYPE to  std::vector<TYPE*>
  std::vector<TYPE*> *l = new std::vector<TYPE*>();

  for (Py_ssize_t i = 0; i < PyList_GET_SIZE(sipPy); ++i) {
    int state;
    PyObject *item = PyList_GET_ITEM(sipPy, i);
    TYPE* p = static_cast<TYPE*>(sipConvertToType(item, kpTypeDef, NULL, SIP_NOT_NONE, &state, sipIsErr));

    if (*sipIsErr) {
      sipReleaseType(p, kpTypeDef, state);
      delete l;
      return 0;
    }

    sipTransferTo(item, item);
    l->push_back(p);

    sipReleaseType(p, kpTypeDef, state);
  }
																									  *sipCppPtr = l;
  return sipGetState(sipTransferObj);
%End
};

template<TYPE>
%MappedType std::vector<TYPE> {
%TypeHeaderCode
#include <vector>
%End

%ConvertFromTypeCode

  PyObject *l;

  const sipTypeDef* kpTypeDef = getSipType("TYPE");
  if (!kpTypeDef) {
     return NULL;
  }

  // Create the Python list of the correct length.
  if ((l = PyList_New(sipCpp -> size())) == NULL) {
     return NULL;
  }

  int i = 0;
  for(std::vector<TYPE>::iterator iter = sipCpp->begin(); iter != sipCpp->end(); iter++) {
    TYPE *cpp = new TYPE(*iter);
    PyObject *pobj;

    // Get the Python wrapper for the Type instance, creating a new
    // one if necessary, and handle any ownership transfer.

    if ((pobj = sipConvertFromNewType(cpp, kpTypeDef, sipTransferObj)) == NULL) {
      // There was an error so garbage collect the Python list.
      Py_XDECREF(l);
      return NULL;
    }

    // Add the wrapper to the list.
    PyList_SET_ITEM(l, i++, pobj);
  }
  // Return the Python list.

  return l;

%End

%ConvertToTypeCode
 const sipTypeDef* kpTypeDef = getSipType("TYPE");
  if (!kpTypeDef) {
     return 0;
  }

  // Check if type is compatible
  if (sipIsErr == NULL) {
    if (!PyList_Check(sipPy)) {
     return 0;
    }
    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(sipPy); ++i) {
      PyObject *item = PyList_GET_ITEM(sipPy, i);
      if (!sipCanConvertToType(item, kpTypeDef, SIP_NOT_NONE)) {
        //printErrorMessage("TypeError : object in list of type " + std::string(item->ob_type->tp_name) + " can not be converted to TYPE");
        return 0;
      }
   }
   return 1;
  }

  // Convert Python list of TYPE to  std::vector<TYPE>
  std::vector<TYPE> *l = new std::vector<TYPE>();
  for (Py_ssize_t i = 0; i < PyList_GET_SIZE(sipPy); ++i) {
    int state;
    TYPE* p = static_cast<TYPE*>(sipConvertToType(PyList_GET_ITEM(sipPy, i), kpTypeDef, sipTransferObj, SIP_NOT_NONE, &state, sipIsErr));

    if (*sipIsErr) {
      sipReleaseType(p, kpTypeDef, state);
      delete l;
      return 0;
    }

    l->push_back(*p);
    sipReleaseType(p, kpTypeDef, state);
  }

  *sipCppPtr = l;
  return sipGetState(sipTransferObj);

%End
};

// ****************************************************
// Specialization for std::vector<double>
// ****************************************************

%MappedType std::vector<double> {
%TypeHeaderCode
#include <vector>
%End

%ConvertFromTypeCode
  PyObject *l;

  // Create the Python list of the correct length.
  if ((l = PyList_New(sipCpp -> size())) == NULL) {
    return NULL;
  }

  // Go through each element in the C++ instance and convert it to a
  // wrapped object.
  int i = 0;
  for (std::vector<double>::iterator iter = sipCpp->begin(); iter != sipCpp->end(); ++iter) {
    // Add the wrapper to the list.
    PyList_SET_ITEM(l, i++, PyFloat_FromDouble(*iter));
  }
  // Return the Python list.
  return l;
%End

%ConvertToTypeCode
  // Check if type is compatible
  if (sipIsErr == NULL) {
   if (!PyList_Check(sipPy)) {
      return 0;
   }
    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(sipPy); ++i) {
      PyObject *item = PyList_GET_ITEM(sipPy, i);
      if (!PyNumber_Check(item)) {
        //printErrorMessage("TypeError : object in list of type " + std::string(item->ob_type->tp_name) + " can not be converted to double");
        return 0;
      }
    }
    return 1;
  }

  // Convert Python list of numbers to a std::vector<double>
  std::vector<double> *v = new std::vector<double>();
  for (Py_ssize_t i = 0; i < PyList_GET_SIZE(sipPy); ++i) {
    PyObject *f = PyNumber_Float(PyList_GET_ITEM(sipPy, i));
    v->push_back(PyFloat_AsDouble(f));
    Py_XDECREF(f);
  }

  *sipCppPtr = v;
  return sipGetState(sipTransferObj);
%End
};

// ****************************************************
// Specialization for std::vector<float>
// ****************************************************

%MappedType std::vector<float> 
{
%TypeHeaderCode
#include <vector>
%End

%ConvertFromTypeCode
 PyObject *l;

 // Create the Python list of the correct length.
 if ((l = PyList_New(sipCpp -> size())) == NULL) {
  return NULL;
 }

 // Go through each element in the C++ instance and convert it to a
 // wrapped object.
 int i = 0;
 for (std::vector<float>::iterator iter = sipCpp->begin(); iter != sipCpp->end(); ++iter) {
     // Add the wrapper to the list.
     PyList_SET_ITEM(l, i++, PyFloat_FromDouble(*iter));
 }

 // Return the Python list.
 return l;
%End

%ConvertToTypeCode
 // Check if type is compatible
 if (sipIsErr == NULL) {
   if (!PyList_Check(sipPy)) {
     return 0;
   }
   for (Py_ssize_t i = 0; i < PyList_GET_SIZE(sipPy); ++i) {
     PyObject *item = PyList_GET_ITEM(sipPy, i);
     if (!PyNumber_Check(item)) {
       //printErrorMessage("TypeError : object in list of type " + std::string(item->ob_type->tp_name) + " can not be converted to float");
       return 0;
     }
   }
   return 1;
 }

 // Convert Python list of numbers to a std::vector<double>
 std::vector<float> *v = new std::vector<float>();
 for (Py_ssize_t i = 0; i < PyList_GET_SIZE(sipPy); ++i) {
   PyObject *f = PyNumber_Float(PyList_GET_ITEM(sipPy, i));
   v->push_back(float(PyFloat_AsDouble(f)));
   Py_XDECREF(f);
 }

 *sipCppPtr = v;
 return sipGetState(sipTransferObj);
%End
};

// ****************************************************
// Specialization for std::vector<int>
// ****************************************************

%MappedType std::vector<int> {
%TypeHeaderCode
#include <vector>
%End

%ConvertFromTypeCode
  PyObject *l;

  // Create the Python list of the correct length.
  if ((l = PyList_New(sipCpp -> size())) == NULL) {
    return NULL;
  }

  // Go through each element in the C++ instance and convert it to a
  // wrapped object.
  int i = 0;
  for (std::vector<int>::iterator iter = sipCpp->begin(); iter != sipCpp->end(); ++iter) {
    // Add the wrapper to the list.
    PyList_SET_ITEM(l, i++, PyLong_FromLong(*iter));
  }

  // Return the Python list.
  return l;
%End

%ConvertToTypeCode
  // Check if type is compatible
  if (sipIsErr == NULL) {
    if (!PyList_Check(sipPy)) {
      return 0;
    }
    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(sipPy); ++i) {
      PyObject *item = PyList_GET_ITEM(sipPy, i);
#if PY_MAJOR_VERSION >= 3
      if (!PyLong_Check(item)) {
#else
        if (!PyLong_Check(item) && !PyInt_Check(item)) {
#endif
          //printErrorMessage("TypeError : object in list of type " + std::string(item->ob_type->tp_name) + " can not be converted to int");
          return 0;
        }
      }
      return 1;
    }

    // Convert Python list of integers to a std::vector<int>
    std::vector<int> *v = new std::vector<int>();
    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(sipPy); ++i) {
#if PY_MAJOR_VERSION >= 3
      v->push_back(int(PyLong_AsLong(PyList_GET_ITEM(sipPy, i))));
#else
      PyObject *item = PyList_GET_ITEM(sipPy, i);
      if (PyLong_Check(item)) {
        v->push_back(int(PyLong_AsLong(item)));
      } else {
        v->push_back(int(PyInt_AsLong(item)));
      }
#endif
    }

    *sipCppPtr = v;
    return sipGetState(sipTransferObj);
%End
};
																	
// ****************************************************
// Specialization for std::vector<unsigned int>
// ****************************************************

%MappedType std::vector<unsigned int> {
%TypeHeaderCode
#include <vector>
%End

%ConvertFromTypeCode
  PyObject *l;

  // Create the Python list of the correct length.
  if ((l = PyList_New(sipCpp -> size())) == NULL) {
    return NULL;
  }

  // Go through each element in the C++ instance and convert it to a
  // wrapped object.
  int i = 0;
  for (std::vector<unsigned int>::iterator iter = sipCpp->begin(); iter != sipCpp->end(); ++iter) {
    // Add the wrapper to the list.
    PyList_SET_ITEM(l, i++, PyLong_FromUnsignedLong(*iter));
  }

  // Return the Python list.
  return l;
%End

%ConvertToTypeCode
  // Check if type is compatible
  if (sipIsErr == NULL) {
    if (!PyList_Check(sipPy)) {
      return 0;
    }
    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(sipPy); ++i) {
      PyObject *item = PyList_GET_ITEM(sipPy, i);
#if PY_MAJOR_VERSION >= 3
     if (!PyLong_Check(item)) {
#else
        if (!PyLong_Check(item) && !PyInt_Check(item)) {
#endif
          //printErrorMessage("TypeError : object in list of type " + std::string(item->ob_type->tp_name) + " can not be converted to int");
          return 0;
        }
      }
      return 1;
    }

    // Convert Python list of integers to a std::vector<int>
    std::vector<unsigned int> *v = new std::vector<unsigned int>();
    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(sipPy); ++i) {
#if PY_MAJOR_VERSION >= 3
      v->push_back(uint(PyLong_AsUnsignedLong(PyList_GET_ITEM(sipPy, i))));
#else
      PyObject *item = PyList_GET_ITEM(sipPy, i);
      if (PyLong_Check(item)) {
        v->push_back(uint(PyLong_AsUnsignedLong(item)));
      } else {
        v->push_back(uint(PyInt_AsUnsignedLongMask(item)));
      }
#endif
    }

    *sipCppPtr = v;
    return sipGetState(sipTransferObj);
%End
};

// ****************************************************
// Specialization for std::vector<long>
// ****************************************************

%MappedType std::vector<long> {
%TypeHeaderCode
#include <vector>
%End

%ConvertFromTypeCode
  PyObject *l;

  // Create the Python list of the correct length.
  if ((l = PyList_New(sipCpp -> size())) == NULL) {
    return NULL;
  }

  // Go through each element in the C++ instance and convert it to a
  // wrapped object.
  int i = 0;
  for (std::vector<long>::iterator iter = sipCpp->begin(); iter != sipCpp->end(); ++iter) {
    // Add the wrapper to the list.
    PyList_SET_ITEM(l, i++, PyLong_FromLong(*iter));
  }

  // Return the Python list.
  return l;
%End

  %ConvertToTypeCode
  // Check if type is compatible
  if (sipIsErr == NULL) {
    if (!PyList_Check(sipPy)) {
      return 0;
    }
    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(sipPy); ++i) {
      PyObject *item = PyList_GET_ITEM(sipPy, i);
#if PY_MAJOR_VERSION >= 3
      if (!PyLong_Check(item)) {
#else
        if (!PyLong_Check(item) && !PyInt_Check(item)) {
#endif
          //printErrorMessage("TypeError : object in list of type " + std::string(item->ob_type->tp_name) + " can not be converted to long");
          return 0;
        }
      }
      return 1;
    }

    // Convert Python list of integers to a std::vector<int>
    std::vector<long> *v = new std::vector<long>();
    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(sipPy); ++i) {
#if PY_MAJOR_VERSION >= 3
      v->push_back(PyLong_AsLong(PyList_GET_ITEM(sipPy, i)));
#else
      PyObject *item = PyList_GET_ITEM(sipPy, i);
      if (PyLong_Check(item)) {
        v->push_back(PyLong_AsLong(item));
      } else {
        v->push_back(PyInt_AsLong(item));
      }
#endif
    }

    *sipCppPtr = v;
    return sipGetState(sipTransferObj);
%End
};

// ****************************************************
// Specialization for std::vector<unsigned long>
// ****************************************************

%MappedType std::vector<unsigned long> {
%TypeHeaderCode
#include <vector>
%End

%ConvertFromTypeCode
  PyObject *l;

  // Create the Python list of the correct length.
  if ((l = PyList_New(sipCpp -> size())) == NULL) {
    return NULL;
  }

  // Go through each element in the C++ instance and convert it to a
  // wrapped object.
  int i = 0;
  for (std::vector<unsigned long>::iterator iter = sipCpp->begin(); iter != sipCpp->end(); ++iter) {
    // Add the wrapper to the list.
    PyList_SET_ITEM(l, i++, PyLong_FromUnsignedLong(*iter));
  }

  // Return the Python list.
  return l;
%End

%ConvertToTypeCode
  // Check if type is compatible
  if (sipIsErr == NULL) {
    if (!PyList_Check(sipPy)) {
      return 0;
    }
    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(sipPy); ++i) {
      PyObject *item = PyList_GET_ITEM(sipPy, i);
      #if PY_MAJOR_VERSION >= 3
      	  if (!PyLong_Check(item)) {
	  #else
          if (!PyLong_Check(item) && !PyInt_Check(item)) {
	  #endif
          //printErrorMessage("TypeError : object in list of type " + std::string(item->ob_type->tp_name) + " can not be converted to unsigned long");
          return 0;
        }
      }
      return 1;
    }

    // Convert Python list of integers to a std::vector<int>
    std::vector<unsigned long> *v = new std::vector<unsigned long>();
    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(sipPy); ++i) {
     #if PY_MAJOR_VERSION >= 3
      	 v->push_back(PyLong_AsUnsignedLong(PyList_GET_ITEM(sipPy, i)));
     #else
      PyObject *item = PyList_GET_ITEM(sipPy, i);
      if (PyLong_Check(item)) {
        v->push_back(PyLong_AsUnsignedLong(item));
      } else {
        v->push_back(PyInt_AsUnsignedLongMask(item));
      }
     #endif
    }

    *sipCppPtr = v;
    return sipGetState(sipTransferObj);
%End
};

// ****************************************************
// Specialization for std::vector<std::string>
// ****************************************************

%MappedType std::vector<std::string> {
%TypeHeaderCode
#include <vector>
%End

%ConvertFromTypeCode
  PyObject *l;

  // Create the Python list of the correct length.
  if ((l = PyList_New(sipCpp -> size())) == NULL) {
    return NULL;
  }

  // Go through each element in the C++ instance and convert it to a
  // wrapped object.
  int i = 0;
  for (std::vector<std::string>::iterator iter = sipCpp->begin(); iter != sipCpp->end(); ++iter) {
    PyObject* newstring;
    newstring = PyUnicode_DecodeUTF8(iter->c_str(), iter->length(), NULL);
    if(newstring == NULL) {
        PyErr_Clear();
        newstring = PyBytes_FromString(iter->c_str());
    }
    // Add the wrapper to the list.    
    PyList_SET_ITEM(l, i, newstring);
  }

  // Return the Python list.
  return l;
  %End

%ConvertToTypeCode
  // Check if type is compatible
  if (sipIsErr == NULL) {
    if (!PyList_Check(sipPy)) {
      return 0;
    }
    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(sipPy); ++i) {
      PyObject *item = PyList_GET_ITEM(sipPy, i);
      if (!PyBool_Check(item)) {
        //printErrorMessage("TypeError : object in list of type " + std::string(item->ob_type->tp_name) + " can not be converted to bool");
        return 0;
      }
    }
    return 1;
  }

  // Convert Python list of string to a std::vector<std::string>
  std::vector<std::string> *v = new std::vector<std::string>();
  for (Py_ssize_t i = 0; i < PyList_GET_SIZE(sipPy); ++i) {
    PyObject *item = PyList_GET_ITEM(sipPy, i);	      
    v->push_back((std::string)PyBytes_AsString(item));
  }

  *sipCppPtr = v;
  return sipGetState(sipTransferObj);
%End
};


// ****************************************************
// Specialization for std::string
// ****************************************************

%MappedType std::string
{
%TypeHeaderCode
#include <string>
%End

%ConvertFromTypeCode
    // convert an std::string to a Python (unicode) string
    PyObject* newstring;
    newstring = PyUnicode_DecodeUTF8(sipCpp->c_str(), sipCpp->length(), NULL);
    if(newstring == NULL) {
        PyErr_Clear();
        newstring = PyBytes_FromString(sipCpp->c_str());
    }
    return newstring;
%End

%ConvertToTypeCode
    // Allow a Python string (or a unicode string) whenever a string is
    // expected.
    // If argument is a Unicode string, just decode it to UTF-8
    // If argument is a Python string, assume it's UTF-8
    if (sipIsErr == NULL)
        return (PyBytes_Check(sipPy) || PyUnicode_Check(sipPy));
    if (sipPy == Py_None) {
        *sipCppPtr = new std::string;
        return 1;
    }
    if (PyUnicode_Check(sipPy)) {
        PyObject* s = PyUnicode_AsEncodedString(sipPy, "UTF-8", "");
        *sipCppPtr = new std::string(PyBytes_AS_STRING(s));
        Py_DECREF(s);
        return 1;
    }
    if (PyBytes_Check(sipPy)) {
        *sipCppPtr = new std::string(PyBytes_AS_STRING(sipPy));
        return 1;
    }
    return 0;
%End
};
