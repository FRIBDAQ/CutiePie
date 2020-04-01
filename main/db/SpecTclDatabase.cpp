/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2017.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Authors:
             Ron Fox
             Giordano Cerriza
	     NSCL
	     Michigan State University
	     East Lansing, MI 48824-1321
*/

/** @file:  SpecTclDatabase.cpp
 *  @brief: Implementation of the SpecTclDatabase class
 */

#include "SpecTclDatabase.h"
#include "CSqlite.h"
#include "CSqliteStatement.h"
#include "CSqliteException.h"

namespace SpecTcl {
    
    
    
//////////////////////////////////////////////////////
//  Static methods:
//
/**
 * create
 *    Static method to create a new database in a file.
 *    If the file does not exist then sqlite3 will create it.
 *    If the file exists, Sqlite3 will create the schema
 *    into it.  If there are errors an CSqliteException is
 *    thrown.  This is derived from std::exception.
 * @param database - Path to the database.
 */
void
CDatabase::create(const char* database)
{
    CSqlite connection(database);
    
      // Top level savesets table:
      
    CSqliteStatement::execute(
        connection,
        "CREATE TABLE IF NOT EXISTS  save_sets    \
        (id  INTEGER PRIMARY KEY,                 \
         name TEXT UNIQUE,                        \
         timestamp INTEGER)"
    );
        // Parameter definitions
    
    CSqliteStatement::execute(
        connection,
        "CREATE TABLE IF NOT EXISTS parameter_defs        \
        (id      INTEGER PRIMARY KEY,                     \
         save_id INTEGER NOT NULL,  -- foreign key to save_sets.id \
         name    TEXT NOT NULL,                           \
         number  INTEGER NOT NULL,                        \
         low     REAL,                                    \
         high    REAL,                                    \
         bins    INTEGER,                                 \
         units   TEXT)"
    );
    CSqliteStatement::execute(
        connection,
        "CREATE INDEX IF NOT EXISTS pdef_save_id  \
            ON parameter_defs (save_id)"
    );
    CSqliteStatement::execute(
        connection,
        "CREATE INDEX IF NOT EXISTS pdef_name ON parameter_defs (name)"
    );
    // Spectrum definitions.
    
    CSqliteStatement::execute(
        connection,
        "CREATE TABLE IF NOT EXISTS spectrum_defs   \
        (id      INTEGER PRIMARY KEY,               \
         save_id INTEGER NOT NULL,     -- Foreign key to save_sets.id \
         name    TEXT NOT NULL,                     \
         type    TEXT NOT NULL,                     \
         datatype TEXT NOT NULL                     \
        )"
    );
    CSqliteStatement::execute(
        connection,
        "CREATE INDEX IF NOT EXISTS sdef_save_id \
            ON spectrum_defs (save_id)"
    );
    
    CSqliteStatement::execute(
        connection,
        " CREATE TABLE IF NOT EXISTS axis_defs     \
        (   id           INTEGER PRIMARY KEY,      \
            spectrum_id  INTEGER NOT NULL,  -- FK to spectrum_defs.id \
            low          REAL NOT NULL,            \
            high         REAL NOT NULL,            \
            bins         INTEGER NOT NULL          \
        )"
    );
    CSqliteStatement::execute(
        connection,
        "CREATE INDEX IF NOT EXISTS adef_specid  \
            ON axis_defs (spectrum_id)"
    );
    
    CSqliteStatement::execute(
        connection,
        "CREATE TABLE IF NOT EXISTS spectrum_params   \
        (   id          INTEGER PRIMARY KEY,          \
            spectrum_id INTEGER NOT NULL,    -- FK to spectrum_defs.id \
            parameter_id INTEGER NOT NULL    -- Fk to parameter_defs.id \
        )"
    );
    CSqliteStatement::execute(
        connection,
        "CREATE INDEX IF NOT EXISTS sparams_spectrum_id \
                ON spectrum_params (spectrum_id)"
    );
    
    // Spectrum contents.
    
    CSqliteStatement::execute(
        connection,
        "CREATE TABLE IF NOT EXISTS spectrum_contents   \
        (   id             INTEGER PRIMARY KEY,         \
            spectrum_id    INTEGER NOT NULL,      -- FK to spectrum_defs.id. \
            xbin           INTEGER NOT NULL,            \
            ybin           INTEGER,                     \
            value          INTEGER NOT NULL             \
        )"
    );
    CSqliteStatement::execute(
        connection,
        "CREATE INDEX IF NOT EXISTS scontents_spectrum_id \
                ON spectrum_contents (spectrum_id)"
    );
    
    // Gate definitions:
    
    CSqliteStatement::execute(
        connection,
        "CREATE TABLE IF NOT EXISTS gate_defs       \
            (   id          INTEGER PRIMARY KEY,    \
                saveset_id  INTEGER NOT NULL,   -- FK to save_sets.id \
                name        TEXT NOT NULL,          \
                type        TEXT NOT NULL           \
            )"
    );
    //     Primitive gates have points and parameter ids.
    
    CSqliteStatement::execute(
        connection,
        "CREATE TABLE IF NOT EXISTS gate_points    \
            (   id          INTEGER PRIMARY KEY,   \
                gate_id     INTEGER NOT NULL,           -- FK to gate_defs.id \
                x           REAL,                       -- Point x coordinate. \
                y           REAL                        -- Point y coords.     \
            )"
    );
    CSqliteStatement::execute(
        connection,
        "CREATE INDEX IF NOT EXISTS gate_points_gatidx \
                ON gate_points (gate_id)"
    );
    
    CSqliteStatement::execute(
        connection,
        "CREATE TABLE IF NOT EXISTS gate_parameters \
            (   id   INTEGER PRIMARY KEY,           \
                parent_gate INTEGER NOT NULL,     -- FK to gate_defs.id \
                parameter_id INTEGER NOT NULL     -- FK for parameter_defs.id \
            )"
    );
    CSqliteStatement::execute(
        connection,
        "CREATE INDEX IF NOT EXISTS gate_params_parentidx \
                ON gate_parameters (parent_gate)"
    );
    CSqliteStatement::execute(
        connection,
        "CREATE INDEX IF NOT EXISTS gate_params_paramidx  \
                ON gate_parameters (parameter_id)"
    );
    
    //     compound gates have other gates.
    
    CSqliteStatement::execute(
        connection,
        "CREATE TABLE IF NOT EXISTS component_gates       \
            (                                             \
                id          INTEGER PRIMARY KEY,          \
                parent_gate INTEGER NOT NULL,      -- gate_defs.id of owner. \
                child_gate  INTEGER NOT NULL       -- Gate parent_gate depends on. \
            )"
    );
    CSqliteStatement::execute(
        connection,
        "CREATE INDEX IF NOT EXISTS component_gates_parentidx  \
                ON component_gates (parent_gate)"
    );
    
    //     bitmask gates have masks.
    
    CSqliteStatement::execute(
        connection,
        "CREATE TABLE IF NOT EXISTS gate_masks       \
            (   id          INTEGER PRIMARY KEY,     \
                parent_gate INTEGER NOT NULL,        \
                mask        INTEGER NOT NULL         \
            )"
    );
    CSqliteStatement::execute(
        connection,
        "CREATE INDEX IF NOT EXISTS gate_mask_parentix \
            ON gate_masks (parent_gate)"
    );
    
    // Gate applications:
    
    CSqliteStatement::execute(
        connection,
        "CREATE TABLE IF NOT EXISTS gate_applications ( \
                id                INTEGER PRIMARY KEY,  \
                spectrum_id       INTEGER NOT NULL,      -- fk to spectrum_defs \
                gate_id           INTEGER NOT NULL       -- fk to gate_defs \
            )"
    );
    // Tree Variables:
    
    CSqliteStatement::execute(
        connection,
        "CREATE TABLE IF NOT EXISTS treevariables (   \
                id             INTEGER PRIMARY KEY,   \
                save_id        INTEGER NOT NULL,      \
                name           TEXT NOT NULL,         \
                value          DOUBLE NOT NULL,       \
                units          TEXT                   \
            )"
    );
    CSqliteStatement::execute(
        connection,
        "CREATE INDEX IF NOT EXISTS treevariables_saveidx \
            ON treevariables (save_id)"
    );
    // Support putting run data in the database:
    
    CSqliteStatement::execute(
        connection,
        " CREATE TABLE IF NOT EXISTS runs (    -- Runs that were saved. \
                id         INTEGER PRIMARY KEY,                         \
                config_id  INTEGER,              -- Configuration at begin of run. \
                run_number INTEGER UNIQUE NOT NULL,                     \
                title      TEXT NOT NULL,                               \
                start_time INTEGER,                                     \
                stop_time  INTEGER              -- End of run time      \
            )"
    );
    CSqliteStatement::execute(
        connection,
        "CREATE INDEX IF NOT EXISTS run_num_idx ON runs (run_number)"
    );
    CSqliteStatement::execute(
        connection,
        "REATE TABLE IF NOT EXISTS events (      \
                id         INTEGER PRIMARY KEY,  \
                run_id     INTEGER NOT NULL,   -- fk to runs(id). \
                event_number INTEGER NOT NULL, -- Event number in the run. \
                parameter_count INTEGER NOT NULL, -- Hints at the event blob size. \
                event_data  BLOB NOT NULL        \
            )"
    );
    CSqliteStatement::execute(
        connection,
        "CREATE TABLE IF NOT EXISTS scaler_readouts (   \
                id            INTEGER PRIMARY KEY,      \
                run_id        INTEGER NOT NULL,      -- fk for runs. \
                source_id     INTEGER NOT NULL,      -- Event builder source. \
                start_offset  INTEGER NOT NULL,         \
                stop_offset   INTEGER NOT NULL,         \
                divisor       INTEGER NOT NULL,         \
                clock_time    INTEGER NOT NULL          \
            )"
    );
    CSqliteStatement::execute(
        connection,
        " CREATE TABLE IF NOT EXISTS scaler_channels (   \
                id          INTEGER PRIMARY KEY,         \
                readout_id  INTEGER NOT NULL,      -- fk for scaler_readouts. \
                channel     INTEGER NOT NULL,      -- Channel number.         \
                value       INTEGER NOT NULL       -- Channel Value.          \
            )"
    );
    

}

}