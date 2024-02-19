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

/** @file:  MirrorClientInternals.cpp
 *  @brief:  Implement mirror internals functions.
 */
#include "MirrorClientInternals.h"
#include <stdexcept>

//  #define DEBUGGING 1
#ifdef DEBUGGING
#define DEBUG(msg) std::cout << msg << std::endl; std::cout.flush()
#else
#define DEBUG(msg)
#endif

static const int HTTPSuccess = 200;

#ifndef _WIN64
#include <unistd.h>
#include "CPortManager.h"
#include <restclient-cpp/restclient.h>
#include <json/json.h>
#include <os.h>

/**
 * formatUrl
 *    @param host - host of spectcl
 *    @param port - Port of SpecTcl.
 *    @param domain - subchunk of the REST interface.,
 *    @return std::string - URL.
 */
static std::string
formatUrl(const char* host, int port, const char* domain)
{
    std::stringstream strPort;
    strPort << "http://" << host << ":" << port << "/spectcl/" << domain;
    std::string result = strPort.str();
    return result;
}
/**
 * Check a JSON status - should be OK else throw a runtime error with
 * the value in detail
 */
void
checkStatus(Json::Value& v)
{
    Json::Value s = v["status"];
    if (s.asString() != "OK") {
        Json::Value d = v["detail"];
        std::string msg = d.asString();
        throw std::runtime_error(msg);
    }
}
/**
 * GetMirrorList
 *    Get the list of current mirrors that are being maintained by
 *    a specific SpecTcl...see header.
 */
std::vector<MirrorInfo>
GetMirrorList(const char* host, int port)
{
    std::vector<MirrorInfo> result;
    auto uri = formatUrl(host, port, "mirror");
    RestClient::Response r = RestClient::get(uri);

    if (r.code != HTTPSuccess) {
        throw std::runtime_error(r.body);
    }
    // Now let's process the JSON:
    
    Json::Value root;
    std::stringstream data(r.body);
    data >> root;

    checkStatus(root);
    
    const Json::Value mirrorList = root["detail"];
    for (int i =0; i < mirrorList.size(); i++) {
        const Json::Value d = mirrorList[i];
        std::string host = d["host"].asString();
        std::string mem  = d["shmkey"].asString();
        
        MirrorInfo item = {host, mem};
        result.push_back(item);
    }
    
    return result;
}
/**
 * GetSpectrumSize
 *   Get the shared memory spectrum soup size.
 */
size_t
GetSpectrumSize(const char* host, int port)
{
    std::string uri = formatUrl(host, port, "shmem/size");
    RestClient::Response r = RestClient::get(uri);
    
    if (r.code != HTTPSuccess) {
        throw std::runtime_error(r.body);
    }
    //
    Json::Value root;
    std::stringstream data(r.body);
    data >> root;
    
    checkStatus(root);
    std::string strSize = root["detail"].asString();
    
    return atol(strSize.c_str());
}
/**
 * GetSpecTclSharedMemory
 *    Get the name of the SpecTcl shared memory.
 */
std::string
GetSpecTclSharedMemory(const char* host, int port)
{
    std::string uri = formatUrl(host, port, "shmem/key");
    RestClient::Response r = RestClient::get(uri);
    if (r.code != HTTPSuccess) {
        throw std::runtime_error(r.body);
    }
    //
    Json::Value root;
    std::stringstream data(r.body);
    data >> root;
    
    checkStatus(root);
    
    return root["detail"].asString();
}
/**
 * LookupPort
 *    See header - returns the port associated with a service name
 *    advertised in the DAQPort manager
 */

int
LookupPort(const char* host, const char* service, const char* user)
{
    if (!user) {
        user = Os::whoami();
        if (!user) {
            throw std::logic_error("Could not determine username");
        }
    }
    CPortManager pm(host);
    auto services = pm.getPortUsage();
    for (auto s : services) {
        if ((s.s_Application == service) && (s.s_User == user)) {
            return s.s_Port;
        }
    }
    // no match:
    
    throw std::logic_error("No port matches this service/username.");
}
#endif

// Windows implementation
#ifdef _WIN64

#define CURL_STATICLIB
#include <curl/curl.h>
#include <sstream>
#include <iostream>
#include <json/json.h>

static size_t my_curl_writefn(char* ptr, size_t nmemb, size_t nbytes, void* dest) {
    std::string* pbuffer = reinterpret_cast<std::string*>(dest);

    // note that ptr is not null terminated so we can't just use +=.

    pbuffer->append(ptr, nbytes);   // But this shouild work.
    return nbytes;

}

/**
 * GetSpectrumSize
 *   Get the shared memory spectrum soup size.
 * @param host - host specification (DNS name or dotted ip.).
 * @param port - port on which the REST server is running.
 */
size_t
GetSpectrumSize(const char* host, int port)
{
    DEBUG("IN getspectrum size");
    CURL* session = curl_easy_init();
    char error_text[CURL_ERROR_SIZE];

    // Generate the URL to access the shared memory size:

    std::stringstream urls;
    urls << "http://" << host << ':' << port << "/spectcl/shmem/size";
    std::string url = urls.str();

    // Set the session URL

    auto status = curl_easy_setopt(session, CURLOPT_URL, const_cast<char*>(url.c_str()));
    DEBUG("Session URL set" << url);
    if (status != CURLE_OK) {
        curl_easy_cleanup(session);
        throw std::runtime_error("Failed to set URL option in CURL");
    }
    DEBUG("Ok");
    // In order not to write a file, we need to set a write function callback.
    // In this case, the write function callback will just save the data to an std::string:

    std::string body;      // Data goes here.
    if (curl_easy_setopt(session, CURLOPT_WRITEFUNCTION, my_curl_writefn) != CURLE_OK) {
        curl_easy_cleanup(session);
        throw std::runtime_error("Could not set curl writeback");
    }
    DEBUG("Write callback set");
    if (curl_easy_setopt(session, CURLOPT_WRITEDATA, &body) != CURLE_OK) {
        curl_easy_cleanup(session);
        throw std::runtime_error("Could not set curl writeback parameter (buffer)");
    }
    DEBUG("Write Parameter eset");
    // Put a place for the error messages:

    if (curl_easy_setopt(session, CURLOPT_ERRORBUFFER, error_text) != CURLE_OK) {
        curl_easy_cleanup(session);
        throw std::runtime_error("Failed to set errro buffer");
    }
    DEBUG("Error Buffer set");

    if (curl_easy_perform(session) != CURLE_OK) {
        curl_easy_cleanup(session);     // Should wrap all in try/catch and centralize this.
        std::string reason("Failed to perform shared memory size get: ");
        reason += std::string(error_text);
        throw std::runtime_error(reason);
    }
    curl_easy_cleanup(session);
    DEBUG("Transaction performed");
    // body should have the result of the query:


    // What we got back was JSOn with:
    // status: == "OK" on success else an error message.
    // detail: an integer spectrum storage size on success.

    Json::Value root;
    DEBUG("Got back ");
    DEBUG(body);
    std::stringstream sbody(body);
    sbody >> root;
    Json::Value json_status = root["status"];

    if (json_status.asString() == "OK") {
        Json::Value detail = root["detail"];
        long size = atol(detail.asString().c_str());
        DEBUG("Returning " << size);
        return size;
    }
 
    
    return 0;
}

/** 
*  Get the list of mirrors
    STUB STUB STUB
*/
std::vector<MirrorInfo>
GetMirrorList(const char* host, int port)
{
    std::vector<MirrorInfo> result;
    return result;
}

// There is no port manager so this always fails:
//

int LookupPort(const char* host, const char* service, const char* user)
{
    throw std::runtime_error("On windows ports must be integers not service names");
}


const char* getlogin() {
    return nullptr;
}
#endif