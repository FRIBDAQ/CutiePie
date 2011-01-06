/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2008

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Author:
             Ron Fox
	     NSCL
	     Michigan State University
	     East Lansing, MI 48824-1321
*/
#include "spectcl_experiment.h"
#include <uuid/uuid.h>
#include <sqlite3.h>
#include <stdlib.h>
#include <string.h>
/**
 * This file implements the UUID specific functions of the spectcl experiment database.
 */


/*---------------------------------------------------------------------------------
** Public entries;
----------------------------------------------------------------------------------*/

/**
 ** Return the UUID of an experiment as a uuid_t
 ** the uiud_t has been dynamically allocated and therefore
 ** must be free'd at some point.,
 ** @param db   - Experiment database.
 ** @return uuid_t*
 ** @retval NULL - Unable to get the UUID, spectcl_experiment_errno has the reason for that.
 ** @retval other- Pointer to dynamically allocated uuid_t that contains the parsed UUID.
 */
uuid_t*
spectcl_experiment_uuid(spectcl_experiment db)
{
  const char*           pSql = "SELECT config_value FROM configuration_values \
                                                    WHERE config_item = 'uuid'";
  sqlite3_stmt*         stmt;
  int                   status;
  uuid_t*               result = NULL;
  uuid_t                uuid;
  const unsigned char*  uuidText;

  status = sqlite3_prepare_v2(db,
			      pSql, -1, &stmt, NULL);
  if(status != SQLITE_OK) {
    spectcl_experiment_errno = status;
    return NULL;
  }
  
  status = sqlite3_step(stmt);
  if (status != SQLITE_ROW) {
    spectcl_experiment_errno = status;
    return NULL;
  }
  uuidText = sqlite3_column_text(stmt, 0);
  result = malloc(sizeof(uuid_t));
  uuid_parse((char*)uuidText, *result);

  sqlite3_finalize(stmt);

  return result;
}
/**
 ** Determine if a UUID matches that of the experiment.
 ** @param db  - experiment database.
 ** @param uuid - Uuid to compare with.
 ** @return bool
 ** @retval TRUE -matches
 ** @retval FALSE- nomatch.
 */
bool
spectcl_correct_experiment(spectcl_experiment db, uuid_t* uuid)
{
  uuid_t* myuuid = spectcl_experiment_uuid(db);
  int     result = (uuid_compare(*myuuid, *uuid) == 0);

  free(myuuid);

  return result;

}