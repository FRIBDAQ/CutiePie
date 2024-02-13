/*
** Facility:
**   Xamine  - NSCL display program.
** Abstract:
**   client.c:
**     This file contains client side code for the AEDTSK. It will be put
**     into a library to which clients can link.  There are two sets of
**     calls in this file:
**        C callable  (these are Xamine_xxxx only).
**        native f77 callable (These are f77xamine_xxx(_) only).
**        See client.f for the AED compatibility library.
** Author:
**   Ron Fox
**   NSCL
**   Michigan State University
**   East Lansing, MI 48824-1321
*/


/*
** Includes:
*/

#include <config.h>

#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>



#include <sys/types.h>
#include <unistd.h>


#include <sys/ipc.h>
#include <sys/shm.h>

#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <signal.h>

#include "xamineDataTypes.h"

/*
** Definitions.
*/

#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif

#define NAME_FORMAT "XA%02x"
#define SHARENV_FORMAT "XAMINE_SHMEM=%s" /* Environment names/logical names. */
#define SIZEENV_FORMAT "XAMINE_SHMEM_SIZE=%uld"

#define XAMINEENV_FILENAME "XAMINE_IMAGE"

#ifndef HOME
#define XAMINE_PATH "/daq/bin/%s"
#endif

/*
** DEFINE's.  These DEFINE's are used to derive the names of the message
** queues used by this module:
*/


/*
** Exported variables;
*/
using namespace Xamine;

volatile Xamine_shared *Xamine_memory;
size_t            Xamine_memsize;

/*
** Local storage:
*/

static pid_t Xamine_Pid = 0;
static int   Xamine_Memid = -1;
static void* Xamine_LastMemory = NULL;           /* So we can detach ... */
static pid_t Memwatcher_Pid;








/*
** Functional Description:
**   Xamine_MapMemory:
**     This function is used to map to an existing shared memory region
**     given it's name.  The idea is that a histogrammer could fire up Xamine
**     and publish the name and size of the shared memory region using
**     Xamine_GetMemoryName.
** Formal Parameters:
**    char *name:
**       Name of the shared memory region.
**    int specbytes:
**       Number of bytes of spectrum storage.
**    Xamine_shared **ptr:
**       In VMS, this is an input pointer to the desired map location.
**       In UNIX, the map location is determined by the map operation
**        and this argument is an output only.
**       The structure of the INLCUDE files in FORTRAN make this transparent
**       if always used as an input.  In C, conditional compilation allow
**       common sources.
** Returns:
**    1 - If successful.
**    0 - If failed.  Error reason is in the usual errno crap.
*/

/*
** First we take care of the Fortran call interface: 
*/



/*
** The code which follows is the native C implementation of 
**  Xamine_MapMemory which is described by the comment header way up there
** It's completely system dependent.
*/
int Xamine_MapMemory(const char *name, size_t specbytes,  struct _Xamine_Header** ptr)

{
  size_t memsize;
  key_t key;
  int shmid;

  /* First generate the size of the shared memory region in bytes: */

  memsize = sizeof(Xamine_shared) - XAMINE_SPECBYTES + specbytes;
  key  = *(key_t *)name;		/* Convert name to key. */

  /* get the shared memory id from the key and size: */

  shmid = shmget(key, memsize,0);
  if(shmid == -1) return 0;

  /* Now map to the memory read/write */

  *ptr = (Xamine_shared *)shmat(shmid, 0, 0);
  
  /* Return success if shmat returned a non null pointer */

  Xamine_memory = *ptr;		/* Save memory pointer for mgmnt rtns. */
  return (*ptr ? 1 : 0);
}
