#!/bin/sh
# -*- tcl -*-
# The next line is executed by /bin/sh, but not tcl \
exec tclsh "$0" ${1+"$@"}

#    This software is Copyright by the Board of Trustees of Michigan
#    State University (c) Copyright 2014.
#
#    You may use this software under the terms of the GNU public license
#    (GPL).  The terms of this license are described at:
#
#     http://www.gnu.org/licenses/gpl.txt
#
#    Authors:
#             Ron Fox
#             Giordano Cerriza
#	     NSCL
#	     Michigan State University
#	     East Lansing, MI 48824-1321


##
# @file  mirrorclientscript.tcl
# @brief Script part of mirror client.
# @author Ron Fox <fox@nscl.msu.edu>
#

set updateMultiplier 2000;    # See update - update time multiplier of last update

##
# Entry point:
# - Get the command parameters.
# - If necessary/possible, translate the ports from the port manager.
# - See if there's already a mirror client for this host for the requested SpecTcl
#   (if there is, exit with a message).
# - Set up the mirror
# - Periodically  update it
#@note when we start the updates we'll enter an event loop so that these can be
#      driven via [after].
#@note that evidently connections to the localhost, even when specifying the hostname
#      look like localhost so this needs to be fixed up.
#
package require SpecTclRESTClient

##
# isLocalhost
#  Determines if a host is localhost..  it is if any of the following are true:
#    -  It is explicitly named localhost.
#    - It is equal  to the output of the hostname command.
#    - It is equal to the output of the hostname -f command.
#
# @param host - host to check.
#
proc isLocalhost {host} {
    if {$host eq "localhost" } {
        return 1
    }
    # Matches short name
    
    if {$host eq [exec hostname -s]} {
        return 1
    }
    # Matches any of the long names:
    
    if {$host in [exec hostname --all-fqdns]} {
        return 1
    }
    return 0
}
##
# lookupPort
#   Given an NSCLDAQ service name and a user, returns the port number corresponding to
#   that name.  Note that this requires that a version of NSCLDAQ is setup.
#   If DAQTCLLIB is defined we add it to the auto_path... if a version prior to
#   12.0 is used, the user will need to have defined TCLLIBPATH e.g.
#
# @param host   - host to do the lookup  in.,
# @param service - Service to lookup.
# @param user   - Name of the user running the service.
# @return port number.
#  Errors are thrown if:
#   -   We can't include the port lookup package.
#   -   We can't find a match for the port.
#
 proc lookupPort {host service user} {
    if {[array names ::env DAQTCLLIBS] ne ""} {
        lappend ::auto_path $::env(DAQTCLLIBS)
    }
    set status [catch {package require portAllocator}]
    if {$status} {
        puts stderr  "To use service names, you must either setup NSCLDAQ >= 12.0 or defined TCLLIBPATH to \$DAQROOT/TclLibs"
        exit -1
    }
    set c [portAllocator create %AUTO% -hostname $host]
    set result [$c findServer $service $user]
    $c destroy
    
    if {$result eq ""} {
        puts stderr  "The user $user is not advertising a service named $service in $host"
        exit -1
    }
    return $result
 }
 
 ##
 # makeRestClent
 #   Just a one-liner to create and return a SpecTcl rest client object:
 #
 # @param  host - host runing the REST server
 # @pararm port - numeric port the server listens to for connections.
 # @return string - Rest client base command.
 #
 proc makeRestClient {host port} {
    return [SpecTclRestClient %AUTO% -host $host -port $port]
 }
##
# alreadyMirroring
#   Check to see if the server is already running a mirror.
#
# @param client - REST client.
# @param host   - This will be localhost  if the server is local (e.g. as in
#                 a persistent container mirror outside the container).
#
# @return integer 1 if there's already a mirror to this host.
#
proc alreadyMirroring {client host} {
    set mirrors [$client mirror]
    
    
    # Make a list of hosts:
    
    set hostList [list]
    foreach m $mirrors {
        lappend  hostList [dict get $m host]
    }
    #  Local host:
    
    if {$host eq "localhost"} {
        return [expr {$host in $hostList}]
    }
    # Look for short and long hostnames:
    
    set shortHost [exec hostname -s]
    set longHosts [exec hostname --all-fqdns]
    if {$shortHost in $hostList} {
        return 1
    }
    foreach long $longHosts {
        if {$long in $hostList} {
            return 1
        }
    }
    return 0
    
}
##
# updateMirror
#   Update the mirror, and schedule another update
#    Our update rate is dynamic as follows:
#    minimum update is 1 second from completing the prior update.
#    we time the update.  If 5x the update time is > 1 second we use that instead.
# @note that if the update fails we just exit.
#
proc updateMirror { } {
    set status [catch {
        time {Mirror::mirror update} 1
    } timingInfo]
    if {$status} {
        incr ::forever
        puts stderr "Mirror update failed! exiting"
        Mirror::mirror destroy
        
        return
    }
    set usec [lindex $timingInfo 0]
    set ms [expr {$usec/1000}];    # Milliseconds to do the last update.
    set updateInterval [expr {int(min(1000.0, $ms * $::updateMultiplier))}]
    
    after $updateInterval updateMirror
}

#---------------------------------------------------------------------------
# Entry point.

# If the host is actually local we do a fixup:

set host [Mirror::gethost]
if {[isLocalhost $host]} {
    set host localhost
}
# Get the ports and, if needed, figure them out from service names
set restport [Mirror::getrestport]
set mirrorport [Mirror::getmirrorport]

# If the ports are not numeric try to do the lookup.

if {![string is integer -strict $restport] } {
    set restport [lookupPort $host $restport [Mirror::getuser]]
}

if {![string is integer -strict $mirrorport]} {
    set mirrorport [lookupPort $host $mirrorport [Mirror::getuser]]
}

# See if we've already got a mirror running:

set restClient [makeRestClient $host $restport]
if {[alreadyMirroring $restClient $host]} {
    puts stderr "There is already a mirror running to this host."
    exit -1
}
# Set up the mirror and start updates:

set shmsize [$restClient shmemsize]
set name [Mirror::mirror create $shmsize $host $mirrorport]

updateMirror
$restClient destroy

vwait forever;           # Enter the event loop.
puts stderr "Mirror update failed....exiting"
exit -1
