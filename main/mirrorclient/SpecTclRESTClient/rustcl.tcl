#!/usr/bin/tclsh
#  ^^ for linux.
#
#  This is a simple script to serve as a SpecTcl 
#  Compatible shell for rustogramer (or a SpecTcl running
#  the REST server for that matter).  
# 
#   Accepts two paraemters
#     host - host on which the REST server is running.
#     port - THe port on which the REST server is listening for connections.
#
#  It:
#    *  loads the SpecTclRestCommand (which must be in the TCL Package load)
#       search path.
#    *  Does a SpecTclRestCommand::initialize to set up the command.
#    *  Accepts input on stdin and, submits a command to the interpreter
#       when the input is a complete command.
#

if {[llength $argv] != 2} {
    puts stderr "Incorrect number of command line arguments:"
    puts stderr "Usage:"
    puts stderr "   rustcl.tcl host port"
    exit 1
}
set host [lindex $argv 0]
set port [lindex $argv 1]

package require SpecTclRestCommand
SpecTclRestCommand::initialize $host $port



set prompt1 "% "  ;   # first line of command.
set prompt2 " "   ;   # Second line of command.

set prompt $prompt1
set command ""
while {![eof stdin]} {
    puts -nonewline $prompt 
    flush stdout
    append command [gets stdin]
    set prompt $prompt2

    # Unless its a complete command in which case:

    if {[info complete $command] } {

        set prompt $prompt1
        catch  {eval $command} msg
        puts $msg
        flush stderr
        set command ""
    }

}
exit 0