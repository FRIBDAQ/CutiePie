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
#include <sys/wait.h>
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
static char *(env[8]) = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
/*
** Functional Description:
**   genname:
**      This local function generates a name for the shared memory region
** Formal Parameters:
**   char *name:
**      Buffer for shared memory region name.
** Returns:
**   TRUE -- success.
*/
static int genname(char *name)
{
  pid_t pid;

  pid = getpid();		/* Get the process id. */
  pid = (pid & 0xff);		/* Only take the bottom byte of pid. */
  sprintf(name, NAME_FORMAT, (int)pid);	/* Format the name. */
  return 1;
}

/*
** Functional Description:
**   genmem:
**     Function to generate the shared memory region and map to it.
** Formal Parameters:
**   char *name:
**     Name of region.
**   void **ptr:
**     Pointer to base buffer (set in VMS to desired base).
**   unsigned int size:
**     Total number of bytes in the region.
** Returns:
**    True - Success
**    False- Failure.
**  NOTE: In the Unix case, the **ptr value is modified to indicate where
**        the shared memory was allocated.
*/
static int genmem(char *name, volatile void **ptr, unsigned int size)
{				/* UNIX implementation. */
  key_t key;
  int   memid;
  char *base;
  pid_t pid;

  /* If we're already attached to a memory region detach.  That let's our forked guy die: */

  if (Xamine_LastMemory) {
    shmdt(Xamine_LastMemory);
    Xamine_Memid=-1;
    Xamine_LastMemory = NULL;
  }
  
  /* Create the shared memory region: */


  memcpy(&key, name, sizeof(key));

  memid = shmget(key, size,
 	         (IPC_CREAT | IPC_EXCL) | S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH); /* Owner rd/wr everyone else, read only.*/
  if(memid == -1) {
    return 0;
  }
  fprintf(stderr, "Created %d\n", memid);
  
  /*
    spawn a daemon that will clean up shared memory when no more processes
    are attached to it.
  */
  pid = fork();
  if (pid == 0) {
      struct shmid_ds stat;

    /* child */

      /* detach the child from the parent */
      int sid = setsid();
      shmctl(memid, IPC_STAT, &stat);

      while (stat.shm_nattch != 0) {
          sleep(1);
          shmctl(memid, IPC_STAT, &stat);
      }
      fprintf(stderr, "killing mem %d\n", memid);
      shmctl(memid, IPC_RMID, 0);
      exit(EXIT_SUCCESS);
  }
  Memwatcher_Pid = pid;
  /* Attach to the shared memory region: */

  base = (char *)shmat(memid, NULL, 0);
  if(base == NULL) {
    return 0;
  }

  Xamine_Memid = memid;		/* Save the memory id. for Atexit<. */
  Xamine_LastMemory = base;


  *ptr = (void *)base;
  return -1;			/* Indicate successful finish. */
}				/* Unix implementation. */

/*
** Functional Description:
**  genenv:
**    Generate the environment strings needed by Xamine when it is run.
**    The strings are placed in this process' environment so that they can
**    be inherited by Xamine when it is run as a child process.
**    This is done by building up the env array and passing strings one by
**    one to putenv.  On unix, putenv is a library function however 
**    on VMS it is a module local function to create a process wide logical
**    name.
** Formal Parameters:
**   char *name:
**      Name of the global section/shared memory region.
**   int specbytes:
**      number of bytes in the shared memory region.
** Returns:
**    True    - Success
**    False   - Failure
*/

static int genenv(const char *name, size_t specbytes)
{
  /* Allocate persistent storage for the strings */

  env[0] = (char*)malloc(strlen(name) + strlen(SHARENV_FORMAT) + 1);
  if(env[0] == NULL)
    return 0;

  env[1] = (char*)malloc(strlen(SIZEENV_FORMAT) + 20);
  if(env[1] == NULL) {
    free(env[0]);
    return 0;
  }
  /* Generate the environment variables: */

  sprintf(env[0], SHARENV_FORMAT, name);
  sprintf(env[1], SIZEENV_FORMAT, specbytes);

  if(putenv(env[0])) {
    free(env[0]);
    free(env[1]);
    return 0;
  }

  if(putenv(env[1])) {
    free(env[1]);
    return 0;
  }
  return 1;
}





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
int Xamine_MapMemory(const char *name, size_t specbytes,  Xamine_Header** ptr)

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
int Xamine_DetachSharedMemory()
{


  return shmdt((const void*)Xamine_memory);
}
int Xamine_CreateSharedMemory(size_t specbytes,volatile Xamine_shared **ptr)
{

  char name[33];

  if(!genname(name))		/* Generate the shared memory name. */
    return 0;

  if(!genmem(name, 
	     (volatile void **)ptr,	/* Gen shared memory region. */
             sizeof(Xamine_shared) - XAMINE_SPECBYTES + specbytes)) {
    return 0;
  }

  if(!genenv(name, specbytes))	/* Generate the subprocess environment. */
    return 0;

  Xamine_memsize = specbytes;
  Xamine_memory  = *ptr;		/* Save poinyter to memory for mgmnt rtns. */
  return 1;			/* set the success code. */
}
/*
** Functional Description:
**   killmem:
**      This UNIX only function is a cleanup function that's called to
**      ensure that shared memory segments will be cleaned up after
**      exit.
** Formal Parameters:
**   NONE:
*/
void killmem()

{
  if(Xamine_Memid > 0) {
     struct shmid_ds stat;
     if (Xamine_LastMemory) {
       int status;
       int stat;
       fprintf(stderr, "detatching previously attached memory %d\n", Xamine_Memid) ;
       shmdt(Xamine_LastMemory);	/* Detach to stop the watcher. */
       Xamine_LastMemory = NULL;
       do {
	 fprintf(stderr, "Reaping %d\n", Memwatcher_Pid);
	 stat =  waitpid(Memwatcher_Pid,  &status, 0);
	 if (stat == -1) {
	   fprintf(stderr, "Waitpid failed %d\n", errno);
	   break;
	 }
	 sleep(1);		/*  System cleanup deletion of memory is async(?) */
       } while(!WIFEXITED(stat));
     }

     Xamine_Memid = -1;

  }
}
void Xamine_KillSharedMemory()
{
  killmem();
}

/*
** Functional Description:
**   Xamine_GetMemoryName:
**     This function retuzrns the name of the shared memory region
**     that will be used for Xamine.  This allows the client to publish the
**     name in a way that allows other processes to get at it to manipulate
**     the same memory region.
** Formal Parameters:
**   char *namebuffer:
**      Buffer for the name (must be large enough).
*/
void Xamine_GetMemoryName(char *namebuffer)
{
  genname(namebuffer);
}
