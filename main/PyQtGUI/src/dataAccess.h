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

/** @file:  dataAccess.h
 *  @brief: Methods to access shared memory information
 */

#ifndef DATAACCESS_H
#define DATAACCESS_H

#include "dataTypes.h"

#include <string.h>
#include <stdio.h>
#include <iostream>
#include <sys/types.h>

#include <string>

typedef void (*Address_t);          
static const Address_t kpNULL = (Address_t)0;

struct spec_shared : shared_memory
{
  // general info on size of spectra and channels
  int getspectrumcount() {return MAXSPEC; }
  int getmaxchannels() {return SPECBYTES; }
  
  // dimensions and n channels
  int getxdim(int id) { return dsp_xy[id]._xchans; }
  int getxdim(char *name) 
  { 
    int id;
    id = getspecid(name); 
    return ((id < 0) ? -1: getxdim(id)); 
  }

  int getydim(int id) { return dsp_xy[id]._ychans; }
  int getydim(char *name) 
  { 
    int id;
    id = getspecid(name);
    return ((id < 0) ? -1: getydim(id)); 
  }

  // access channel via id
  int getchannels(int id) 
  { 
    return (((dsp_types[id] == _onedlong) ||
	     (dsp_types[id] == _onedword))? 
	    dsp_xy[id]._xchans : dsp_xy[id]._xchans * dsp_xy[id]._ychans); 
  }

  // access channel via name
  int getchannels(char *name) 
  {
    int id;
    id = getspecid(name);
    return ((id < 0)? -1: getchannels(id)); 
  }
  
  // axis max/min and label
  float getxmax_map(int id) { return dsp_map[id]._xmax; }
  float getxmin_map(int id) { return dsp_map[id]._xmin; }
  float getymax_map(int id) { return dsp_map[id]._ymax; }
  float getymin_map(int id) { return dsp_map[id]._ymin; }

  char *getxlabel_map(spec_label label, int id);
  char *getylabel_map(spec_label label, int id);

  // spectrum type
  spec_type gettype(int id);
  spec_type gettype(char *name) 
  { 
    int id;
    id = getspecid(name);
    return ((id < 0)? _undefined : gettype(id)); 
  }

  // other info
  int getspecid(char *name);
  char *getname(spec_title name, int id);
  char *getinfo(spec_title info, int id);

  bool isOned(int type);

  // summary info
  int GetSpectrumList(char ***list);
  int GetSpectrumId(char *name);
  int GetSpectrumId(std::string name);
  spec_type GetSpectrumType(int id);
  
  Address_t CreateSpectrum(int id);
  
};

#endif
