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

/** @file:  dataRetriever.cpp
 *  @brief: API to set and retrieve shared memory information
 */


#include <cstring>
#include <algorithm>
#include <iostream>
#include <array>
#include <memory>
#include <stddef.h>  // defines NULL
#include <stdlib.h>
#include <sys/types.h>
#ifndef _WIN64
#include <unistd.h>
#include <bits/stdc++.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#endif
#include "dataRetriever.h"

dataRetriever* dataRetriever::m_pInstance = NULL;
static int memsize;
bool dbg = false;

dataRetriever*
dataRetriever::getInstance()
{
  if (!m_pInstance)   
    m_pInstance = new dataRetriever;

  return m_pInstance;
}

void
dataRetriever::SetShMem(spec_shared* p)
{
  shmem = p;
}
  
spec_shared*
dataRetriever::GetShMem()
{
  return shmem;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Test methods (not for public use). Shared memory requests have to go through the mirrorclient.
// These function were thought and deployed BEFORE the SpecTclMirrorClient was even an idea. So please don't use them
// because they fullfil the needs of something now obsolete
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//void
//dataRetriever::PrintOffsets()
//{
//  spec_shared *p(0);
//  printf("Offsets into shared mem: \n");
// printf("  dsp_xy      = %p\n", (void*)p->dsp_xy);
//  printf("  dsp_titles  = %p\n", (void*)p->dsp_titles);
//  printf("  dsp_types   = %p\n", (void*)p->dsp_types);
// printf("  dsp_map     = %p\n", (void*)p->dsp_map);
//  printf("  dsp_spectra = %p\n", (void*)&(p->dsp_spectra));
//  printf("  Total size  = %d\n", sizeof(spec_shared));

//}



