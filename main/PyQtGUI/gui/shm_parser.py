#!/usr/bin/python3
"""Print the SpecTcl shared-memory key and size, one line, space separated.

A standalone command, not a module: run it, do not import it. Nothing in the
GUI imports this file and nothing should — CutiePie reaches shared memory
through CPyConverter, and this exists so a human (or a shell script) can ask
the same question from a terminal when the mirror is misbehaving.

    ./shm_parser.py <host> <port>
    RESThost=spdaq22 RESTport=8080 ./shm_parser.py

Arguments win over the environment; with neither, it raises KeyError. Installed
by gui/Makefile.am, which is why it ships despite having no importer — a
"dead code, zero importers" reading of that is a false positive (AUDIT M23a).
"""
import os
import sys
import json
import httplib2

import errno
from socket import error as socket_error

def cleaning(args):
    lst = []
    args = (str(args)).split("'")
    for i, value in enumerate(args):
        if (i%2):
            lst.append(args[i])
    return lst
            
try:
    try:
        args = cleaning(sys.argv)
        h = args[1]
        p = args[2]
    except Exception:
        h = os.environ['RESThost']
        p = os.environ['RESTport']

    key_address = "http://"+h+":"+p+"/spectcl/shmem/key"    
    key = httplib2.Http().request(key_address)[1]    
    var = json.loads(key.decode())
    size_address = "http://"+h+":"+p+"/spectcl/shmem/size" 
    size = httplib2.Http().request(size_address)[1]    
    var2 = json.loads(size.decode())
    print(var['detail'], var2['detail'])

except socket_error as serr:
    print("Inside exception socket_error")
    if serr.errno != errno.ECONNREFUSED:
        raise serr

