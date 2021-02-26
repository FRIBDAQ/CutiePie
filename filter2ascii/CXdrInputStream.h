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




#ifndef CXDRINPUTSTREAM_H
#define CXDRINPUTSTREAM_H



#include <string>
#include <rpc/types.h>		/* Some systems (e.g. CYGWIN) need this. */
#ifdef BROKEN_XDR_H
#include "../Replace/xdr.h"
#else
#include <rpc/xdr.h>
#endif


/*!
   This class reads a buffered output stream in XDR format.  The user
   must know the order and type of each data item in the stream,
   however this class will do buffer management as well as XDR decoding
   of individual data items on behalf of the user.

   The format of the file is assumed to be a list of fixed length 
   buffers.  Each buffer has a one int header.  The int contains
   the offset at the end of the buffer (result from xdr_getpos when
   the buffer was flushed).

   The remainder of the data is whatever the writer stuck into the
   stream.

   The stream is written in a way that an arbitrary source of
   buffers can be used.
*/
class CXdrInputStream {
  // Private member data:
private:
  int    m_nBuffersize;		//!< Read Unit size.
  char  *m_pBuffer;		//!< The actual buffer.
  XDR    m_Xdr;			//!< XDR conversion data structs.
  std::string m_sConnectionId;	//!< Identifies data source.
  bool   m_fIsOpen;		//!< True when input stream open.
  int    m_nFid;		//!< Base class is a file!!.
  int    m_nSize;		//!< # bytes used in buffer.
public:
  // Constructors and other canonical functions:

  CXdrInputStream(int nBuffersize = 8192);
  CXdrInputStream(const std::string& connection, 
		  int nBuffersize = 8192);
  ~CXdrInputStream();
private:
  CXdrInputStream(const CXdrInputStream& rhs);
  CXdrInputStream& operator=(const CXdrInputStream& rhs);
  int operator==(const CXdrInputStream& rhs);
  int operator!=(const CXdrInputStream& rhs);
public:

  // Selectors
  
  int getBufferSize() const {
    return m_nBuffersize;
  }
  void* getBuffer() {
    return (void*)m_pBuffer;
  }
  bool isOpen() const {
    return m_fIsOpen;
  }
  std::string getConnectionId() const {
    return m_sConnectionId;
  }
  /*! Get file id: note this may be meaningless in derived classes */
  int getFid() const {
    return m_nFid;
  }

  //  Mutators:

protected:
  void setOpen(bool state) {
    m_fIsOpen = state;
  }


  // class operations


   //!  Give name and connect.
public:
  void Connect(const std::string& rName)  {
    m_sConnectionId = rName;
    Connect();
  }

  //! Get an arbitrarily typed item from the buffer:

 void Get(void* item, xdrproc_t cvt) ;

  //  Stream-like operations:

  CXdrInputStream& operator>>(int& n);	//!< Get int.
  CXdrInputStream& operator>>(float& n); //!< Get float.
  CXdrInputStream& operator>>(double& n); //!< Get double.
  CXdrInputStream& operator>>(char& c); //!< Get single char.
  CXdrInputStream& operator>>(char* c); //!< Get czstd::string
  CXdrInputStream& operator>>(std::string& s); //!< Get stl std::string
  CXdrInputStream& operator>>(bool& b); //!< Get a bool.


  // overrides:

  virtual void Connect() ;
  virtual void Disconnect(); 

protected:
  virtual int  Read(void* pBuffer);


protected:
  // Utility functions.

  void DoRead();
  void ConditionalRead() ;

};
   

#endif
