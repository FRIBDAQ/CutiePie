/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2005.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Author:
             Ron Fox
	     NSCL
	     Michigan State University
	     East Lansing, MI 48824-1321
*/

/*
   This file provides a refactoring of several identical
   typedefs and constant defs that occur throughout the
   Xamine header and implementation files.  Yes, I know it
   was originally >bad bad bad< to have done this... that's
   why I'm fixing it.
*/

// Cut down for mirror clients:

#ifndef XAMINEDATATYPES_H
#define XAMINEDATATYPES_H




#include <stdint.h>




#define XAMINE_MAXSPEC 10000	/* Maximum spectrum count. */


/* Graphical object limits etc. */

#define GROBJ_NAMELEN 80
#define GROBJ_MAXPTS   50

namespace Xamine {
  typedef union _spec_spectra {
    uint8_t XAMINE_b[1];
    uint16_t XAMINE_w[1];
    uint32_t XAMINE_l[1];
  } spec_spectra;
#define XAMINE_SPECBYTES sizeof(spec_spectra)
#pragma pack(push, 1)             // both gcc and vcc support this.
    

    typedef struct {
        unsigned int xchans;
        unsigned int ychans;
    } spec_dimension;	/* Describes the channels in a spectrum. */

    typedef char spec_title[128];	/* Spectrum name string */
    typedef spec_title spec_label;	/* These two must be the same due to
                                       the implementation of cvttitle in spectra.cc */

    typedef struct _statistics {
        unsigned int   overflows[2];
        unsigned int   underflows[2];
    } Statistics, * pStatistics;

    typedef enum {
        undefined = 0,
        twodlong = 5,
        onedlong = 4,
        onedword = 2,
        twodword = 3,
        twodbyte = 1
    } spec_type;

    typedef struct {
        float xmin;
        float xmax;
        float ymin;
        float ymax;
        spec_label xlabel;
        spec_label ylabel;
    } spec_map;


    typedef struct _Xamine_Header {
        spec_dimension  dsp_xy[XAMINE_MAXSPEC];
        spec_title      dsp_titles[XAMINE_MAXSPEC];
        spec_title      dsp_info[XAMINE_MAXSPEC];    /* Associated info.  */
        unsigned int    dsp_offsets[XAMINE_MAXSPEC];
        spec_type       dsp_types[XAMINE_MAXSPEC];
        spec_map        dsp_map[XAMINE_MAXSPEC];
        Statistics      dsp_statistics[XAMINE_MAXSPEC];
      spec_spectra      dsp_spectra;

    } Xamine_Header, Xamine_shared;
}     // Xamine namespace.
#pragma pack(pop)
#ifndef _WIN64
int Xamine_MapMemory(const char* name, int size, volatile Xamine::Xamine_shared** pResult);
#endif
#endif
