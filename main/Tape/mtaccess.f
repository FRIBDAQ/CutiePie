c++
c Facility:
C	Unix support 
C Abstract:
C	mtaccess.f - This is an include file intended for use by unix
C		     users of the portable tape access routines.
C		     Portable tape access routines currently exist in
C		     ULTRIX and pSOS and will be ported to VMS.
C Author:
C	Ron Fox
C	NSCL
C	Michigan State University
C	East Lansing, MI 48824-1321
C	January 21, 1992
C
C	SCCS information:
C		Include_file 1/28/92 @(#)mtaccess.f	2.2
C--

C	First define the functions and their types:

c		Block level I/O

      INTEGER F77MTOPEN
      INTEGER F77MTCLOSE
      INTEGER F77WTDRIVE
      INTEGER F77MTCLEARERR
      INTEGER F77MTWRITE
      INTEGER F77MTLOAD
      INTEGER F77MTUNLOAD
      INTEGER F77MTREWIND
      INTEGER F77MTREAD
      INTEGER F77MTWEOF
      INTEGER F77MTSPACEF
      INTEGER F77MTSPACER

C		Message reporting
		
      INTEGER GETERRNO
      CHARACTER*80 F77MTGETMSG

C		Volume level I/O

      INTEGER F77VOLINIT
      INTEGER F77VOLMOUNT
      INTEGER F77VOLDMOUNT
      INTEGER F77VOLCREATE
      INTEGER F77VOLOPEN
      INTEGER F77VOLWRITE
      INTEGER F77VOLREAD
      INTEGER F77VOLCLOSE

C		The following are status codes that can be fed into
C		MTGETMSG to get textual error messages back:

      INTEGER MTSUCCESS,MTINSFMEM,MTBADHANDLE,MTBADRAM,MTBADPROM
      INTEGER MTINTERNAL,MTALLOCATED,MTBADUNIT,MTREQSENSE,MTNOTALLOC
      INTEGER MTLEOT,MTEOMED,MTOVERRUN
      PARAMETER  (MTSUCCESS	=0		)
      PARAMETER  (MTINSFMEM	=101		)
      PARAMETER  (MTBADHANDLE	=102		)
      PARAMETER  (MTBADRAM	=103		)
      PARAMETER  (MTBADPROM	=104		)
      PARAMETER  (MTINTERNAL	=105		)
      PARAMETER  (MTALLOCATED	=106		)
      PARAMETER  (MTBADUNIT	=107		)
      PARAMETER  (MTREQSENSE	=108		)
      PARAMETER  (MTNOTALLOC	=109		)
      PARAMETER  (MTLEOT=		110		)
      PARAMETER  (MTEOMED	=	111		)
      PARAMETER  (MTOVERRUN	=112		)

      INTEGER MTEOF,MTINVLBL,MTNOTANSI,MTWRONGVOL,MTBADPROT
      INTEGER MTNOTMOUNTED,MTFILEOPEN,MTPROTECTED,MTFNAMEREQ
      INTEGER MTFILECLOSE,MTBADLENGTH,MTOFFLINE,MTAVAILABLE,MTLEOV 
      PARAMETER  (MTEOF		=113		)
      PARAMETER  (MTINVLBL	=114		)
      PARAMETER  (MTNOTANSI	=115		)
      PARAMETER  (MTWRONGVOL	=116		)
      PARAMETER  (MTBADPROT	=117		)
      PARAMETER  (MTNOTMOUNTED	=118		)
      PARAMETER  (MTFILEOPEN	=119		)
      PARAMETER  (MTPROTECTED	=120		)
      PARAMETER  (MTFNAMEREQ	=121		)
      PARAMETER  (MTFILECLOSE	=122		)
      PARAMETER  (MTBADLENGTH	=123		)
      PARAMETER  (MTOFFLINE	=124		)
      PARAMETER  (MTAVAILABLE	=125		)
      PARAMETER  (MTLEOV	=	126)

      INTEGER MTNOTFOUND, MTIO		
      PARAMETER  (MTNOTFOUND	=127	)	
      PARAMETER  (MTIO	=	128	)	


C 		Next we define the strucutre of
C		item lists used in some of the volume handling routines.
C
      STRUCTURE/MTITEMLIST/
         INTEGER code
         INTEGER buffer
      END STRUCTURE

C       The following define codes which are used in the volume and file
C	manipulation item lists.  The ones starting with VOL_ are used in
C	volmount, and the ones starting with FILE_ are in the file creation
C	opening routines only.
C

C	Volume item list codes.

      INTEGER VOL_REQLBL, VOL_PROTECT, VOL_LABEL, VOL_ACCESS
      INTEGER VOL_OWNER, VOL_ENDLIST

      PARAMETER (VOL_REQLBL	=1	)
      PARAMETER (VOL_PROTECT	=2	)
      PARAMETER (VOL_LABEL	=3	)
      PARAMETER (VOL_ACCESS	=4	)
      PARAMETER (VOL_OWNER	=5	)
      PARAMETER (VOL_ENDLIST	=0	)
 
C	 File list item codes: 
 
      INTEGER FILE_NAMEREQ, FILE_REQRECLEN, FILE_REQPREFIX, FILE_NAME
      INTEGER FILE_CREDATE, FILE_EXPDATE, FILE_ACCESS, FILE_RECLEN
      INTEGER FILE_BLOCK, FILE_PREFIX, FILE_ENDLIST
      PARAMETER (FILE_NAMEREQ	=1	)
      PARAMETER (FILE_REQRECLEN	=2	)
      PARAMETER (FILE_REQPREFIX	=3	)
      PARAMETER (FILE_NAME	=4	)
      PARAMETER (FILE_CREDATE	=5	)
      PARAMETER (FILE_EXPDATE	=6	)
      PARAMETER (FILE_ACCESS	=7	)
      PARAMETER (FILE_RECLEN	=8	)
      PARAMETER (FILE_BLOCK	=9	)
      PARAMETER (FILE_PREFIX	=10	)
      PARAMETER (FILE_ENDLIST	=0	)

C		Volume accessibility values.

      INTEGER READABLE,WRITEABLE
      PARAMETER (READABLE	=0)
      PARAMETER (WRITEABLE	=1)

