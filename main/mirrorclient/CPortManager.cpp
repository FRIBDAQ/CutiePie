//  This file contains the implementation of the CPortManager class
//  This class interacts with TCP port allocation servers in the NSCL
//  DAQ system.  For more information about this class and its interfaces
//  see CPortManager.h
//
// Author:
//   Ron Fox
//   NSCL
//   Michigan State University
//   East Lansing, MI 48824-1321
//
// (c) 2005 Copyright Board of Trustees of Michigan State University All rights Reserved
// This software is licensed under the Gnu Public License  For license
// terms see: http://www.gnu.org/licenses/gpl.txt
//

#include <config.h>
#include "CPortManager.h"
#include "CPortManagerException.h"


// headers needed for the network.

#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netinet/ip.h>

#include <assert.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <pwd.h>
#include <regex.h>
#include <stdlib.h>


#include <tcl.h>


// #include <publib.h>
#include <ctype.h>

#include <fstream>
#include <sstream>

using namespace std;

//  consts:

static const int DefaultPort(30000);      //!< If there is no listen.port file.

static const string LocalServerPortFile("/var/tmp/daqportmanager/listen.port");

// ---------------------------------------------------------------------------

/*!
 * Construct a CPortManager.  The port manager will not connect on constrution.
 * Connections will be on an as needed basis until the first successful port
 * allocation at which time the object will hold open a connection to the server
 * as required by the protocol.  
 * 
 * \param host (string [in] = "localhost")
 *     The host this port manager will interact with. 
 * \param port (int [in]):
 *     The port to connect with.  This defaults as follows, in the first
 *     form of the constructor:
 *     - If possible, the file /var/tmp/daqportmanager/listen.port is
 *       read for the port.
 *     - If /var/tmp/daqportmanager/listen.port could not be read for the port,
 *       the port defaults to DefaultPort defined above.
*/
CPortManager::CPortManager(string host) :
   m_sHost(host),
   m_nPort(DetermineDefaultPort()),
   m_fisConnected(false)
 {
 }
 CPortManager::CPortManager(string host, int port) :
 	m_sHost(host),
 	m_nPort(port),
 	m_fisConnected(false)
 {
 }
 /*!
  *  Destroy an instance of a CPortManager.  Note that contrary to what you
  * might expect, this is a no-op.  This is because there's a policy assumption
  * that the application will hold on to ports it allocated for the life of the
  * application.  In order to do this, the protocol requires that the application
  * retain an open socket to the server.  Thus, even if m_fisConnected is true,
  * we won't shutdown the socket.
  */
CPortManager::~CPortManager()
{
  if (m_fisConnected) {
    shutdown(m_nSocket, SHUT_WR);
    m_fisConnected = false;
  }
}

/*!
 *   Equality comparison.  Two port managers are considered equivalent if
 * they are talking (or will talk) to the same server on the same port.
 * 
 * \param rhs (const CPortManager& [in]):
 *     The port manager to compare *this to.
 * \return int
 * \retval 0  - *this is not equivalent to rhs.
 * \retval 1  - *this is equivalent to rhs.
 */
int
CPortManager::operator==(const CPortManager& rhs) const
{
	return ( (m_sHost   == rhs.m_sHost)     &&
	         (m_nPort   == rhs.m_nPort));
} 
/*!
 * Inequality comparison.  Two port managers are inequal if operator== 
 * returns false.
 */
int
CPortManager::operator!=(const CPortManager& rhs) const
{
	return !(*this == rhs);
}
// --------------------------------- class operations -----------------------

/*!
 * Allocate a port from the TCP/IP port server.
 * We connect to the server and send it a "GIMME" command.
 * Note that the user is responsible for any application name quoting
 * if the appliation name has spaces or other TCL List breaking characters.
 * 
 * We read back the server reply:
 * -  On FAIL we throw up a CPortManagerException with the reason
 *    decoded to one of NoPorts or NotLocal depending on the actual error
 *    message gotten from the server.
 * - On OK we decode the port and pass that back to the caller.
 * 
 * \param application (string [in]):
 *    The name of an application to identify what the port will be used for.
 *    This must be a 'valid' TCL list element.  e.g.:
 *    - SpecTcl
 *    - {This is a test}
 * \return int
 * \retval  The port that was allocated by the server.
 */
int
CPortManager::allocatePort(string application)
{

	Connect();

	
	// Build up the command and send it.
	
	string command("GIMME ");
	command    += "{";
	command    += application;
	command    += "} ";
	command    += GetUsername();
	command    += '\n';
	
	if(send(m_nSocket, command.c_str(), command.size(), 0) != command.size()) {
		int status = errno;
		
		m_fisConnected = false;        // We've probably lost our connection.
		Disconnect();
		
		string doing(" Sending: ");
		doing +=  command;
		doing += " to the port manager server : ";
		doing += strerror(status);
		throw CPortManagerException(m_sHost,
					    CPortManagerException::ConnectionFailed,
					    doing);
		
	}
	
	// Get the reply: 
	
	string reply = GetLine();

	
	// Simplest decode is with an istringstream:
	
	istringstream s(reply);
	string status;
	
	s >> status;
	if(status != string("OK")) {
		Disconnect();
		s >> status;
		CPortManagerException::Reason 
		         reason(CPortManagerException::ConnectionFailed); // impossible.
 		if(status.find("no free ports") != string::npos) {
			reason = CPortManagerException::NoPorts;
		}
		if(status.find("Only local applications get ports") != string::npos) {
			reason = CPortManagerException::NotLocal;
		}
		assert(reason != CPortManagerException::ConnectionFailed);
		throw CPortManagerException(m_sHost,
									reason,
									" Attempting to allocate a port");
	}
	
	 int port;
	 s >> port;
	 m_fisConnected = true;         // We'll hold the connection from now on.
	 
	 return port;
	
	
}
/*!
 *     Determine which ports have been allocated and send the user back
 *  a vector of portInfo structs.
 * 
 * \return vector<portInfo>
 *     The list of port usages.
 * \throw CPortManagerException
 *    Due to connection/communication problems.
 * */
 vector<CPortManager::portInfo>
 CPortManager::getPortUsage()
 {
 	Connect();
 	
 	string command("LIST\n");
 	
 	if(send(m_nSocket, command.c_str(), command.size(), 0) != command.size()) {
		int status = errno;
		
		m_fisConnected = false;        // We've probably lost our connection.
		Disconnect();
		
		string doing(" Sending: ");
		doing +=  command;
		doing += " to the port manager server : ";
		doing += strerror(status);
		throw CPortManagerException(m_sHost,
					    CPortManagerException::ConnectionFailed,
					    doing);
	 	}
 	
 	// Get the reply.. should be OK nlines followed by nlines of information.
 	
 	string line = GetLine();
 	istringstream s(line);
 	string status;
 	
 	s >> status;
 	if(status != string("OK")) {
 		Disconnect();
 		string doing("Getting port usage list from server : ");
 		doing += line;
 		
 		throw CPortManagerException(m_sHost,
 									CPortManagerException::ConnectionFailed,
 									doing);
 	}
 	// Rest of the line should be number of lines to read:
 	
 	int nlines;
 	s >> nlines;
 	vector<portInfo> result;
 	for(int i =0; i < nlines; i++) {
	  portInfo info;
	  string  data = GetLine();
	  info = GetPortInfo(data);
	  result.push_back(info);
 	}
 	Disconnect();
 	return result;
 }
// -------------------------------- Utility operations ---------------------

/*!
  Connect to the remote system.   If m_fisConnected, this is a no-op as
  we are already connected.  Otherwise, a socket is created, the address of
  the remote system determined and we connect to the remote system.
  The connected socket will be stored in m_nSocket for use by the rest of
  the object.
  
  \throw CPortManagerException
      Will be thrown if any part of the connection process fails.
*/
void
CPortManager::Connect()
{
  if(!m_fisConnected) {
    
    // Attempt to create the socket:
    
    m_nSocket = socket(PF_INET, SOCK_STREAM, tcp());
    if(m_nSocket < 0) {
      int stat = errno;              // String creation etc. can change errno
      string doing(" opening socket : ");
      doing += strerror(stat);
      
      throw CPortManagerException(m_sHost, 
				  CPortManagerException::ConnectionFailed,
				  doing);
    }
    // Translate the host and port to a sockaddr_in structure:
    
    sockaddr_in remoteAddress;
    memset(&remoteAddress, 0, sizeof(remoteAddress));
    
    try {
      GetNetworkAddress(remoteAddress);      // Throws on error.

    }
    catch (...) {
      close(m_nSocket);                       // prevent socket leaks.
      throw;
    }
    
    // connect to the server:
    
    if(connect(m_nSocket, (const sockaddr*)&remoteAddress, sizeof(remoteAddress))) {
      int    stat = errno;
      string doing("Connecting to server : ");
      doing  += strerror(stat);
      close(m_nSocket);
      throw CPortManagerException(m_sHost,
				  CPortManagerException::ConnectionFailed,
				  doing);
    }
  }
  
}
/*!
   Disconnect the socket, unless it's necessary to hold onto a persistent
   connection with the remote host.
*/
void
CPortManager::Disconnect()
{
	// m_fisConnected means holding a persistent connection.. We should only
	// call Disconnect if a socket is open.
	
	if(!m_fisConnected) {
		shutdown(m_nSocket, SHUT_RDWR);
		close(m_nSocket);                   // This may  not be needed?
	}
}
/*!
 * Determine the default port on which to connect.  This is done as follows:
 * First we try to open and read the port number from the file
 * in the const LocalServerPortFile.  If that fails, then we fall back to 
 * the server default port which is in the const DefaultPort
 * \return int
 * \retval  The port on which to connect by default.
*/
int
CPortManager::DetermineDefaultPort()
{
	ifstream portfile(LocalServerPortFile.c_str());
	
	// If we can't even open the port file fall back:
	
	if(!portfile) {
		return DefaultPort;
	}
	int port;
	
	portfile >> port;
	
	// If could not read the port, fall back
	
	if(!portfile) {
		return DefaultPort;
	}
	// Got a good port from the file return it:
	
	return port;
	
}

/*!
   Determine the network address of the peer that we will be talking to.
   this is used prior to a connection to create the sockaddr_in that
   is needed to connect(2).  The port is straightforward as it only needs
   to be converted to network byte ordering.  The host is a bit more interesting.
   - First we try to look it up via gethostbyname(3). If that's successful,
     that's what we use.
   - If the gethostbyname(3) fails, we attempt to treat the hostname as a
     dotted ip string (e.g. "11.22.33.44") and translate it using
     inet_aton(3).
   - If both of the above fail we will throw an exception to indicate the
   network address cannot be computed.  The exception thrown will be a 
   CPortManagerException with the reason ConnectionFailed.
   
   \param result (sockaddr_in& [out]):
       Buffer to hold the resulting network address.
   \throw CPortManagerException
       If the host does not translate.
       
    \note  This could be made a member function, and could cache the
      lookup after doing it once. The assumption is that there will not be
      requent interactions with the server and therefore there's no real penalty
      to looking up the hostname each time we need to connect.
    \note This implementation can only handle AF_INET .. we fail on AF_INET6.
*/
void
CPortManager::GetNetworkAddress(sockaddr_in& result) const
{
	// Do the easy stuff first.
	
	result.sin_family = AF_INET;
	result.sin_port   = htons(m_nPort);
	
	// Attempt to translate the hostname:



	// getaddrinfo is thread-safe everywhere; gethostbyname_r is glibc-only.
	struct addrinfo  hints;
	struct addrinfo* pInfo(nullptr);
	memset(&hints, 0, sizeof(hints));
	hints.ai_family   = AF_INET;
	hints.ai_socktype = SOCK_STREAM;
	if(getaddrinfo(m_sHost.c_str(), nullptr, &hints, &pInfo) == 0) {
	  const sockaddr_in* pAddr =
	    reinterpret_cast<const sockaddr_in*>(pInfo->ai_addr);
	  result.sin_addr = pAddr->sin_addr;
	  freeaddrinfo(pInfo);
	  return;
	}
	else {
	  // For whatever reason, we could not translate the host by name.
	  // try by number:
	  
	  if(!inet_aton(m_sHost.c_str(), &(result.sin_addr))) {
	    throw CPortManagerException(m_sHost,
					CPortManagerException::ConnectionFailed,
					"Host lookup failed");	
	  }	
	}
}
/*!
 * Get a line of text from the socket.  Unfortunately, the only way to do this
 * is character by character reads on the socket until the \n is seen.
 * Note that the \n is removed from the input string.
 *   Any errors in recv are treated as worth a ConnectionFailed CPortManager
 * exception
 * 
 * \return string
 * \retval  the input string gotten from the socket with the \n removed.
*/
string
CPortManager::GetLine() const
{
  string result;
  while(1) {
    char c;
    if(recv(m_nSocket, &c, 1, 0) != 1) {
      int stat = errno;
      string doing(" Reading a line of input from the server : ");
      doing += strerror(errno);
      
      throw CPortManagerException(m_sHost,
				  CPortManagerException::ConnectionFailed,
				  doing);
    }
    if(c == '\n') {
      break;
    }
    if(c != '\r') {		// Seems like tcl terminates with \r\n.
      result += c;
    }
  }
  
  return result;
}

/*!
	Break up a port description line into its components:
	- port - the port the line describes.
	- app  - The application described by the line.
	- user - The user who is running the application.
	
	\param line (string [in]):
	    The line to parse.
	\param result ( CPortManager::portInfo [out]):
	   The elements of the portInfo struct are filled in with the parsed
	        results.

	\note For now, if any of the items required brace quoting the braces will
	be returned with the strings... that is the line is just broken up into
	three chunks:  The first chunk is a leading number terminated by the
	first instance of whitespace.  The third chunk is a trailing set of 
	characters that is terminated by the last whitespace character.
	The third chunk is the stuff in the middle with leading and trailing 
	whitespace stripped off.
*/
CPortManager::portInfo 
CPortManager::GetPortInfo(string l)
{

  portInfo result;

  // Ok I tried to do this with regular expressions, but on debian
  // etch I caught regcomp in the act of smashing the heap
  // so we'll decode the line as a Tcl list which, in fact, it is.


  int argc;
  const char **argv;

  string copy;
  copy.assign(l);
  const char* list = copy.c_str();

  
  int ok = Tcl_SplitList(reinterpret_cast<Tcl_Interp*>(NULL), 
			 list,
			 &argc, &argv);

  if (ok != TCL_OK) {
    throw std::string("CPortManager::GetPortInfo - string is not a good list");
  }
  if (argc != 3) {
    throw std::string("CPortManager::GetPortInfo - list does not have 3 elements");
  }

  result.s_Port        = atoi(argv[0]);
  result.s_Application = argv[1];
  result.s_User        = argv[2];

  Tcl_Free((char*)argv);


  return result;

}
/*!  Get and return the name of the user that corresponds to the current
  effective userid.
  
  @return string
  @retval the name of the user running.

*/
string
CPortManager::GetUsername()
{
 const char* pUser = getlogin();
 if (pUser) {
  return std::string(pUser);
 } else {
  return std::string("*unknown*");
 }
  

}
/*!
  Return the Tcp protocol number.
  If this cannot be found, the function will assert fail.
*/
int
CPortManager::tcp()
{
  struct protoent* pProto = getprotobyname("tcp");
  assert(pProto);
  return pProto->p_proto;
}
