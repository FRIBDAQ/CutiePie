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

/** @file:  dataAccess.cpp
 *  @brief: Methods to access shared memory information
 */

#include <stdio.h>
#ifdef _WIN64
#else
#include <strings.h>
#endif
#include <memory.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>
#include <map>
#include <iostream>

#include "dataAccess.h"
#include "dataRetriever.h"
extern bool debug;


#define dbgprint(s) if(debug) {std::cerr << s << std::endl; }

typedef char chooser_name[128];
chooser_name names[MAXSPEC];
char *(namelist[MAXSPEC]);

typedef std::pair<int, std::string> NumberAndName;

// title formatting
char *upcase(char *s)
{
  char *st;
  st = s;
  while(*st != '\0') {
    *st = toupper(*st);
    st++;
  }
  return s;
}

static char* cvttitle(char *dest, volatile char *src, int upper)
{
  char *p;

  memcpy(dest,(char*)src, sizeof(spec_title));  /* Grab the whole string */
  dest[sizeof(spec_title)-1] = '\0';     /* Ensure null termination */
  if(strlen(dest) > 0) {
    p = dest + strlen(dest)-1;/* Point to last character of string */

    while(p != (dest)) {
      if(isspace(*p)) {
	*p--  = '\0';
      }
      else
	break;
    }
    if(upper) p = upcase(dest);
    else       p = dest;
  }
  return dest;
}

char*
spec_shared::getxlabel_map(spec_label label, int id)
{
  cvttitle(label, dsp_map[id]._xlabel, 0);
  return label;
}
char*
spec_shared::getylabel_map(spec_label label, int id)
{
  cvttitle(label, dsp_map[id]._ylabel, 0);
  return label;
}

spec_type
spec_shared::gettype(int id) 
{
  if( (id < 0) || (id > MAXSPEC))
    return _undefined;
  return dsp_types[id];
}

int
spec_shared::getspecid(char *name)
{
  spec_title search;
  spec_title current;
  int id;

  memcpy(search, name, sizeof(search));
  search[sizeof(search)-1] = '\0';
  upcase(search); // Make an upper case version of the name 
  
  for (id = 0; id < MAXSPEC; id++) {
    if(dsp_types[id] != _undefined) {
      cvttitle(current, dsp_titles[id], 1);
      if(strncmp(current, search, sizeof(spec_title)) == 0) return id+1;
    }
  }
  return -1;
}

char*
spec_shared::getname(spec_title name, int id)
{
  /* cvttitle(name, dsp_titles[id],1); */
  cvttitle(name, dsp_titles[id],0);
  return name;
}

char*
spec_shared::getinfo(spec_title info, int id)
{
  cvttitle(info, dsp_info[id], 1);
  return info;
}

// returns the id of a spectrum given the name
int
spec_shared::GetSpectrumId(char *name)
{
  char *valstart;
  int  id;
  chooser_name n;

  strcpy(n, name);
  valstart = strchr(n, '['); // Look for open 
  if(valstart == NULL) return -1; // not a valid name
  valstart += 1; // point to the digits
  id = atoi(valstart);
  if(id < 0)
    return -1; 
  return id;
}

int
spec_shared::GetSpectrumId(std::string name)
{
  for (int i = 0; i < MAXSPEC; i++) {
    if (dataRetriever::getInstance()->GetShMem()->gettype(i) != _undefined)  {
      spec_title aname;
      dataRetriever::getInstance()->GetShMem()->getname(aname, i);
      if (name == aname) {
	       return i;
      }
    }
  }
  return -1;
}

// returns size and spectrum list
int
spec_shared::GetSpectrumList(char ***list)
{
  dbgprint("spec_shared::GetSpectrumList note undefined is: " << _undefined);
  spec_title aname;
  std::map<std::string, NumberAndName> NameInfo;

  // First a list of spectrum name pointers is generated for the defined spectra:
  int i;
  for(i = 0; i < MAXSPEC; i++) {
    dbgprint("Spectrum " << i << "type is " << dataRetriever::getInstance()->GetShMem()->gettype(i));
    if(dataRetriever::getInstance()->GetShMem()->gettype(i) != _undefined) {
      dataRetriever::getInstance()->GetShMem()->getname(aname, i); 
      if(strlen(aname) == 0) {        // spectra don't require names...
	      strcpy(aname, "<Untitled>");
      }
      dbgprint("Spectrum name: " << aname);
      NameInfo[std::string(aname)] = NumberAndName(i, std::string(aname));
    }
  }
  // Pull the names out of the NumberAndName map (they'll be sorted) and encode them into names:

  int nspec =0;
  int type, nbins_x, nbins_y;
  float xmin, xmax, ymin, ymax;
  spec_shared* d = dataRetriever::getInstance()->GetShMem();
  for(std::map<std::string,NumberAndName>::iterator i = NameInfo.begin(); i != NameInfo.end(); i++) {
    type = d->GetSpectrumType(i->second.first);
    nbins_x = d->getxdim(i->second.first);
    xmin = d->getxmin_map(i->second.first);
    xmax = d->getxmax_map(i->second.first);    
    if (d->getydim(i->second.first) == 0){
      nbins_y = 0;
      ymin = 0;
      ymax = 0;    
    } else {
      nbins_y = d->getydim(i->second.first);    
      ymin = d->getymin_map(i->second.first);
      ymax = d->getymax_map(i->second.first);
    }
    sprintf(names[nspec], "%d %s %d %d %f %f %d %f %f", i->second.first, i->first.c_str(), type, nbins_x, xmin, xmax, nbins_y, ymin, ymax);
    dbgprint("Encoded name for " << nspec << " " << names[nspec]);
    nspec++;
  }

  // Build the pointer name list

  for(i = 0; i < nspec; i ++){
    namelist[i] = names[i];
    //    std::cout << names[i] << std::endl;
  }
  *list = namelist;
  return nspec;

}

spec_type
spec_shared::GetSpectrumType(int id)
{
  return (spec_type)(dataRetriever::getInstance()->GetShMem()->gettype(id)-3);
}

Address_t
spec_shared::CreateSpectrum(int id)
{
  uint32_t nOffset = dataRetriever::getInstance()->GetShMem()->dsp_offsets[id];
  
  switch(dataRetriever::getInstance()->GetShMem()->dsp_types[id]) { 
  case _twodlong:
  case _onedlong:
    nOffset = nOffset*sizeof(int32_t);
    break;
  case _onedword:
  case _twodword:
    nOffset = nOffset*sizeof(int16_t);
    break;
  case _twodbyte:
    break;
  default:
    return NULL;
  }

  Address_t p_storage = (Address_t)(dataRetriever::getInstance()->GetShMem()->dsp_spectra._b + nOffset);

  return p_storage;
}
