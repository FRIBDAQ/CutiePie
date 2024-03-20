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

#include <bits/stdc++.h>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <array>
#include <memory>
#include <stddef.h>  // defines NULL
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/ipc.h>
#include <sys/shm.h>

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

void
dataRetriever::SetHostPortTest(std::string host, std::string port)
{
  if (dbg) {
      std::cout << "Inside dataRetriever::SetHostPort()" << std::endl;
      std::cout << host << "," << port << std::endl;  
    }
  _hostname = host;
  _port = port;
}

void
dataRetriever::InitShMemTest()
{
  if (dbg) {
    std::cout << "Inside dataRetriever::InitShMem()" << std::endl;
    std::cout << _hostname << "," << _port << std::endl;  
  }
  
  std::array<char, 512> buffer;
  std::string filename, tmp;

  std::string command = "python3 ";
  std::string pwd(INSTALLED_IN);
  filename = std::string(pwd)+"/Script/shm_parser.py";
  if(getenv("INSTDIR"))
    command += filename+" "+_hostname+" "+_port;
  else
    command += filename;  
  if (dbg) {
    std::cout << "command --> " << command << std::endl;
  }
  FILE* pipe = popen(command.c_str(), "r");

  if (!pipe)
    std::cerr << "Couldn't start command." << std::endl;

  while (fgets(buffer.data(), 512, pipe) != NULL) {
    tmp += buffer.data();
  }
  auto returnCode = pclose(pipe);

  tmp.erase(std::remove(tmp.begin(), tmp.end(), '\n'), tmp.end());

  std::istringstream ss(tmp);
  do {
    std::string word;
    ss >> word;
    shm.push_back(word);
  } while (ss); 
  
  key = shm.at(0);
  char* name = new char[key.length()+1];
  std::strcpy(name, key.c_str());
  if(name == NULL) {
    perror("Could not translate shared memory name");
    exit(errno);
  }
  size = stoi(shm.at(1));
  if(size == 0) {
    fprintf(stderr, "Shared memory size string illegal");
    exit(-1);
  }
  
  // mapping to memory
  memsize = sizeof(spec_shared) - SPECBYTES + size;

  shmem = dataRetriever::mapmem(name, memsize);
  spectra = shmem;
  if(spectra == (shared_memory *)NULL) {
    perror("Map to shared memory failed!!");
    exit(errno);
  }
  
}

spec_shared*
dataRetriever::mapmem(char* name, unsigned int size)
#ifdef HAVE_SHM_OPEN
{
  int fd = shm_open(name, O_RDWR, S_IRUSR | S_IWUSR);
  if(fd < 0) {
    perror("shm_open failed!");
    return NULL;
  }

  void* pMem;
#ifdef HAVE_MMAP
  pMem = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
#else
  read(fd, pMem, size);
#endif

  if(pMem == (char*)-1) {
    perror("mmap failed!");
    return NULL;
  }
  close(fd);
  shm_unlink(name);
  return (spec_shared*)pMem;
}
#else
{
  key_t key; // Shared memory key.
  int   id; // Shared memory size. 
  char *memory;

  memcpy(&key, name, sizeof(key));


  id  = shmget(key, size, 0); // Get the memory key
  if(id < 0) {
    return (spec_shared *)NULL;
  }

  memory = (char *)shmat(id, NULL, SHM_RDONLY);

  return (spec_shared *)memory;
}
#endif

char*
dataRetriever::MemoryTop()
{
  char *bottom = (char *)shmem;
  bottom      += (memsize-1);
  return      bottom;
}

void
dataRetriever::PrintOffsets()
{
  spec_shared *p(0);
  printf("Offsets into shared mem: \n");
  printf("  dsp_xy      = %p\n", (void*)p->dsp_xy);
  printf("  dsp_titles  = %p\n", (void*)p->dsp_titles);
  printf("  dsp_types   = %p\n", (void*)p->dsp_types);
  printf("  dsp_map     = %p\n", (void*)p->dsp_map);
  printf("  dsp_spectra = %p\n", (void*)&(p->dsp_spectra));
  printf("  Total size  = %d\n", sizeof(spec_shared));

}



