/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2019.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Author:
             Giordano Cerizza
             NSCL
             Michigan State University
             East Lansing, MI 48824-1321
*/

#include <mutex>
#include "SubManagerAPI.h"
#include "SubAPI.h"

zmq::context_t* SubManagerAPI::m_pContextSingleton(nullptr);
SubManagerAPI* SubManagerAPI::m_pInstance = 0;

SubManagerAPI*
SubManagerAPI::getInstance()
{
  if (!m_pInstance) 
    m_pInstance = new SubManagerAPI;

  return m_pInstance;
}

zmq::context_t*
SubManagerAPI::getContext()
{
  if (!m_pContextSingleton) {
    m_pContextSingleton =
      new zmq::context_t(1);
  }
  return m_pContextSingleton;
}

void
SubManagerAPI::CreateThread()
{
  // Creating subscriber task
  if (debug)
    std::cout << "Setting up subscriber..." << std::endl; 
  pthread_create(&subscriber, nullptr, SubAPI::subscriber_task,  (void*)SubManagerAPI::getInstance()->getContext());  

  if (debug)
    std::cout << "...Done! " << std::endl;
}

void
SubManagerAPI::JoinThread()
{
  if (debug)
    std::cout << "Joining threads" << std::endl;
  pthread_join(subscriber, nullptr);
}

void
SubManagerAPI::DetachThread()
{
  if (debug)
    std::cout << "Detaching threads" << std::endl;
  pthread_detach(subscriber);
}

void
SubManagerAPI::Destroy()
{
  delete m_pInstance;
  m_pInstance = NULL;
}
