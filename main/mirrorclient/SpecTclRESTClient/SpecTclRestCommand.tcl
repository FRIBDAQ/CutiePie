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
# @file  SpecTclRestCommand.tcl
# @brief Simulate SpecTcl commands using the REST interface.
# @author Ron Fox <fox@nscl.msu.edu>
#




package provide SpecTclRestCommand 1.0
package require SpecTclRESTClient
package require json::write

##
#  The commands require a client object.  This has to be set with an
#  initialization operations:

namespace eval SpecTclRestCommand {
    variable client ""
    variable traceToken [list]
    variable traceAfterId [list]
    variable varsAfterId  [list]
    
    #  Variables associated with traces.
    
    variable tracePollInterval  2;   # Seconds between polls to get new traces.
    variable parameterTraces [list]
    variable spectrumAddTraces [list]
    variable spectrumDeleteTraces [list]
    variable gateAddTrace [list]
    variable gateDeleteTrace [list]
    variable gateChangedTrace [list]
    variable sbindTraces [list]
    variable unbindTraces [list]
}
#==============================================================================
# Private utilities.

##
# ::SpecTclRestCommand::_pipeDictToList
#   Convert a pipeline dict to the list representation.
#
#  @param d  - pipeline dict deswcription
#  @return list name, processors.
#
proc ::SpecTclRestCommand::_pipeDictToList {d} {
    return [list [dict get $d name] [dict get $d processors]]
}

##
# ::SpecTclRestCommand::_axesDictToAxesList
# Transform a list of axis dicts to a list of axis specifications.
#
# @param axes  - list of axis dicts.
# @return list of lists.
#
proc ::SpecTclRestCommand::_axesDictToAxesList {axes} {
    set result [list]
    foreach axis $axes {
        lappend result [list                                        \
            [dict get $axis low] [dict get $axis high] [dict get $axis bins] \
        ]
    }
    return $result
}

##
# SpecTclRestCommand::_scontentsToJson
#   Converts the list of dicts that represent channel data to Json encoding.
# @param raw list of dicts containing x, optionally y and v.
# @return list of objects with the same attributes.
#
proc SpecTclRestCommand::_scontentsToJson {raw} {
    set objects [list]
    foreach chan $raw {
        lappend objects [json::write object {*}$chan]
    }
    return [json::write array {*}$objects]
}
##
# SpecTclRestCommand::_scontentsToList
#    Takes the list of dicts the scontents REST interface returns as channel
#     data and converts it into the native scontents format.
#
# @param raw - the list of dicts.
# @return list - of two or 3 element sublists.
#
proc SpecTclRestCommand::_scontentsToList {raw} {
    set result [list]
    
    foreach d $raw {
        set sublist [dict get $d x]
        if {[dict exists $d y]} {
            lappend sublist [dict get $d y]
        }
        lappend sublist [dict get $d v]
        lappend result $sublist
    }
    
    return $result
}

##
#SpecTclRestCommand::_computeParameterMetadata
#   Compute the parameter metadata from the remaining stuff.
# @param args - remainder of parameter -new after the id.
# @return dict - Contains the metadata expected by the REST parameterNew method.
#
proc SpecTclRestCommand::_computeParameterMetadata {args} {
    set metadata [dict create]
    
    
    if {[llength $args] == 1} {
        if {[string is integer -strict $args]} {
            dict set metadata resolution $args
        } else {
            dict set metadata units $args
        }
    } elseif {[llength $args] == 2} {
        dict set metadata resolution [lindex $args 0]
        set rangeunits [lindex $args 1]
        dict set metadata low [lindex $rangeunits 0]
        dict set metadata high [lindex $rangeunits 1]
        dict set metadata units [lindex $rangeunits 2]
    } elseif {[llength $args] == 0} {
        # Nothing to do.
    } else {
        error "Parameter metadata specification is invalid $args"
    }
    
    return $metadata
}

##
# SpecTclRestCommand::_dictGetIfExists
#    Returns a dict value or "" if it does not exist.
#
# @param dict - dict variable.
# @param args - Key structure.
#
proc SpecTclRestCommand::_dictGetIfExists {dict args} {
    if {[dict exists $dict {*}$args]} {
        return [dict get $dict {*}$args]
    } else {
        return ""
    }
}

##
# SpecTclRestCommand::_lselect
#   Create a list from one element in each sublist of a list.
# @param l    - list.
# @param index - index to select from the sub lists.
#
proc SpecTclRestCommand::_lselect {l index} {
    set result [list]
    foreach e $l {
        lappend result [lindex $e $index]
    }
    return $result
}
##
# SpecTclRestCommand::_computeWidth
#   Given a tree parameter definition dict returns the bin width
# @param p - tree parameter dict from a list given by treeparamterList
# @return double - bin width.
#
proc SpecTclRestCommand::_computeWidth {p} {
    set low [dict get $p low]
    set high [dict get $p hi]
    set bins [dict get $p bins]
    return [expr {abs($high - $low)/double($bins)}]
}
##
# SpecTclRestCommand::_ptDictsToPts
#   Turn a list of point dictionaries (x,y keys) into a list of x/y pairs.
#
# @param pts - list of point dicts.
# @return list of pairs.
#
proc SpecTclRestCommand::_ptDictsToPts {pts} {
    set result [list]
    
    foreach p $pts {
        lappend result [list [dict get $p x]  [dict get $p y]]
    }
    
    return $result
}
##
# SpecTclRestCommand::_gateDictToDef
#   Turn the dict for a gate that comes back from gateList into
#   a gate definition as SpecTcl native gate -list command might return.
# @param gate  - The gate dict describing the gate.
# @return list - gate definition list.
#
proc SpecTclRestCommand::_gateDictToDef {gate} {
    
    set name [dict get $gate name]
    set gtype [dict get $gate type]
    set result [list $name 0 $gtype];     # Gate id is meaningless..and not returned.
    if {$gtype in [list * + -]} {
        # Compound gates.
        lappend result [dict get $gate gates]
    } elseif {$gtype in [list F T]} {
        lappend result [list]
    } elseif {$gtype in [list gb gc]} {
        lappend result \
            [list [SpecTclRestCommand::_ptDictsToPts [dict get $gate points]] \
            [dict get $gate parameters]]
    } elseif {$gtype in [list b c]} {
        lappend result [list [dict get $gate parameters] \
            {*}[SpecTclRestCommand::_ptDictsToPts [dict get $gate points]]]
    } elseif {$gtype in [list s] } {
        lappend result \
            [list [dict get $gate parameters] \
            [list [dict get $gate low] [dict get $gate high]]]
    } elseif {$gtype in [list gs]} {
        lappend result \
            [list [list [dict get $gate low] [dict get $gate high]] \
            [dict get $gate parameters]]
    } elseif {$gtype in [list am em nm]} {
        lappend result \
            [list [dict get $gate parameters] [format 0x%x [dict get $gate value]]]
    } else {
        error "Unrecognized gate type: $gtype"
    }
    
    return $result  
}
    


##
# SpecTclRestCommand::_getAppliedGateInfo
#   The REST apply interface only provides a gate name.  This
#   code turns that element into the gate definition the apply native
#   command returns.
#
# -  get all gate definitions
# -  Throw them up into an array indexed by gate name.
# -  For each reply from the REST server, convert it into what apply returns.
#
#
# @param info back from the pply command which is a list of dicts containing
#     spectrum - name of a spectrum.
#     gate     - name of the gate.
# @note the -TRUE- gate is special as is -Ungated-
#
proc SpecTclRestCommand::_getAppliedGateInfo {info} {
    set gates [$::SpecTclRestCommand::client gateList]
    array set gateDefs [list]
    set gateDefs(-TRUE-) [list -TRUE- 0 T [list]];   # Special true gate.
    set gateDefs(-Ungated-) [list -Ungated- 0 T [list]]
    foreach gate $gates {
        set name [dict get $gate name]
        set gateDefs($name) [SpecTclRestCommand::_gateDictToDef $gate]
    }
    set result [list]
    foreach ap $info {
        set gateName [dict get $ap gate]
        set spectrumName [dict get $ap spectrum]
        lappend result [list $spectrumName $gateDefs($gateName)]
    }
    return $result
}
#------------------------------------------------------------------------
#  Trace handling.
#

##
# SpecTclRestCommand::_pollTraces
#   Periodically called to poll for new traces and dispatch them to
#   the appropriate trace handlers (if any).
#
proc SpecTclRestCommand::_pollTraces {} {
    
    set traces [$::SpecTclRestCommand::client        \
        traceFetch $SpecTclRestCommand::traceToken   \
    ]
    
    #puts "Trace $traces"
    
    set dstatus [catch {
    foreach trace [dict get $traces parameter] {
        foreach script $SpecTclRestCommand::parameterTraces {
            uplevel #0 [concat $script $trace]
        }
    }
    # While only one gate trace is allowed, doing what we're about to do
    # a. Supports multiple in the future.
    # b. Handles the case where there's no trace set easily.
    
 
    foreach trace [dict get $traces gate] {
        set action [lindex $trace 0]
        set args [lrange $trace 1 end]
        
        if {$action eq "add"} {
            
            foreach script $::SpecTclRestCommand::gateAddTrace {
                uplevel #0 [concat $script $args]
            }
        } elseif {$action eq "delete"} {
        
            foreach script $::SpecTclRestCommand::gateDeleteTrace {
                uplevel #0 [concat $script $args]
            }
            
        } elseif {$action eq "changed"} {

            foreach script $SpecTclRestCommand::gateChangedTrace {
                
                uplevel #0 [concat $script $args]
            }
        } else {
            error "gate trace has an invalid action $action : $trace"
        }
    }
    
    foreach trace [dict get $traces spectrum] {
        set action [lindex $trace 0]
        set args   [lrange $trace 1 end]
        if {$action eq "add"} {
            foreach script $SpecTclRestCommand::spectrumAddTraces {
                uplevel #0  [concat script $args]
            }
        } elseif {$action eq "delete"} {
            foreach script $SpecTclRestCommand::spectrumDeleteTraces {
                uplevel #0 [concat $script $args]
            }
        } else {
            error "spectrum trace action in valid $action : $trace"
        }
    }
     
    foreach trace [dict get $traces binding] {
        set what [lindex $trace 0]
        set params [lrange $trace 1 end]
        
        if {$what eq "add"} {
            foreach tracer $::SpecTclRestCommand::sbindTraces {
                uplevel #0 {*}$tracer {*}$params
            }
        } elseif {$what eq "remove"} {
            foreach tracer $::SpecTclRestCommand::unbindTraces {
                uplevel #0 {*}$tracer {*}$params
            }
        } else {
            
        }
        
    }
    
    } msg]
    if {$dstatus} {
         # puts "trace poll error: $msg"
    }
    set reschedule [expr {$SpecTclRestCommand::tracePollInterval*1000}]
    
    set SpecTclRestCommand::traceAfterId \
        [after  $reschedule SpecTclRestCommand::_pollTraces]
    
    
}
##
# SpecTclRestCommand::_startTraceMonitoring
#    If trace monitoring has not yet been set up, set it up.
#
# @note that SpecTclRestCommand::tracePollInterval dynamically controls
#      the polling interval and that we set the retention time to
#      100*what it is now - and that's not modifiable currently.
#
#
proc SpecTclRestCommand::_startTraceMonitoring { } {
    
    #
    #  If there's already a trace token we don't need to set this up.
    if {$SpecTclRestCommand::traceToken eq ""} {
        set retention [expr {$SpecTclRestCommand::tracePollInterval * 100}]
        set SpecTclRestCommand::traceToken [$SpecTclRestCommand::client \
            traceEstablish $retention                                   \
        ]
        
        SpecTclRestCommand::_pollTraces
        
    }
}
#==============================================================================
# Public entries.
##
# SpecTclRestCommand::initialize
#   Initialize the client object
#
# @param host - host the SpecTcl we control lives in
# @param port - Port the REST server is listening on.
#
proc SpecTclRestCommand::initialize {host port {debug 0}} {
    set SpecTclRestCommand::client [SpecTclRestClient\
        %AUTO% -host $host -port $port -debug $debug         \
    ]
}
##
# SpecTclRestCommand::shutdown
#   Destroy the client object.
#
proc SpecTclRestCommand::shutdown { } {
    if {$SpecTclRestCommand::traceAfterId != [list]} {
        $SpecTclRestCommand::client traceDone $SpecTclRestCommand::traceToken
        after cancel $SpecTclRestCommand::traceAfterId
        set SpecTclRestCommand::traceAfterId [list]
    }
    if {$SpecTclRestCommand::varsAfterId != [list]} {
        after cancel $SpecTclRestCommand::varsAfterId
        set SpecTclRestCommand::varsAfterId [list]
    }
    set ::SpecTclRestCommand::traceToken [list]
    $::SpecTclRestCommand::client destroy
    set ::SpecTclRestCommand::client ""
}

#------------------------------------------------------------------------------
#
# apply
#    Simulate the apply command.  There are two versions:
#    -  apply gate spectrum....
#    -  apply -list ?pattern?
#  We are a bit more tolerant than the native command.
#  If there's junk after the pattern in apply -list we ignore it.
# @parm args - the command parameters.
# @note - yes we know this masks the apply command.
#
proc apply {args} {
    if {[llength $args] < 1} {
        error "'apply' command requires parameters"
    }
    set first [lindex $args 0]
    if {$first eq "-list"} {
        set pattern *
        if {[llength $args] > 1} {
            set pattern [lindex $args 1]
        }
        set info [$SpecTclRestCommand::client applyList $pattern]
        return [SpecTclRestCommand::_getAppliedGateInfo $info]
    } else {
        set gate $first
        set spectra [lrange $args 1 end]
        if {[llength $spectra] == 0} {
            error "'apply' needs at least one spectrum in addition to the gate."
        }
        return [$SpecTclRestCommand::client applyGate $gate $spectra]
    }
}
#------------------------------------------------------------------------------
#
# attach
#   Simulate the attach command in terms of the REST interface.
#
# @param args -parameter arguments.
# @return - what SpecTcl would have - for the most part.
#
#     
proc attach {args} {
    if {[llength $args] ==0} {
        error "'attach' command requires parameters"
    }
    set stype bad
    set size 8192
    set format ring
    
    for {set i 0} {$i < [llength $args]} {incr i} {
        set optname [lindex $args $i]
        if {$optname eq "-pipe"} {
            set stype pipe
        } elseif {$optname eq "-file"} {
            set stype file
        } elseif {$optname eq "-size"} {
            incr i
            set size [lindex $args $i]
        } elseif {$optname eq "-format"} {
            incr i
            set format [lindex $args $i]
        } elseif {$optname eq "-list"} {
            return [$SpecTclRestCommand::client attachList]
        } elseif {[string index $optname 0] eq "-"} {
            error "Unrecognized option $optname"
        } else {
            # This, and the rest are the data source:
            
            return [$SpecTclRestCommand::client attachSource \
                $stype [lrange $args $i end] $size $format         \
            ]       
        }
    }
    # If we get here there's nos ource spec:
    
    error "Missing data source specification."
}
#------------------------------------------------------------------------------
# sbind
#   Simulate the sbind command.  Forms:
#  sbind -all
#  sbind -list
#  sbind -trace
#  sbind -untrace
#  sbind spectrum...
#
proc sbind {args} {
    if {[llength $args] == 0} {
        error "'sbind' command requires parameters"
    }
    set opt [lindex $args 0]
    if {$opt eq "-all"} {
        return [$SpecTclRestCommand::client sbindAll]
    } elseif {$opt eq "-list"} {
        set pattern *
        if {[llength $args] > 1} {
            set pattern [lindex $args 1]
        }
        set rawInfo [$SpecTclRestCommand::client sbindList $pattern]
        set  result [list]
        foreach sb $rawInfo {
            lappend result [list                                        \
                [dict get $sb spectrumid] [dict get $sb name] [dict get $sb binding] \
            ]
        }
        return $result
    } elseif {$opt eq "-trace"} {
        
        if {[llength $args] != 2} {
            error "sbind -trace takes a script and only a script."
        }
        lappend SpecTclRestCommand::sbindTraces [lindex $args 1]
        SpecTclRestCommand::_startTraceMonitoring
    } elseif {$opt eq "-untrace"} {
        if {[llength $args] != 2} {
            error "sbind -untrace takes only a script"
        }
        set script [lindex $args 0]
        set index [lsearch -exact $SpecTclRestCommand::sbindTraces $script]
        if {$index < 0} {
            error  "sbind -untrace no script '$script' is bound"
        } else {
            set SpecTclRestCommand::sbindTraces \
                [lreplace $SpecTclRestCommand::sbindTraces $index $indes]
        }
        
    } else {
        return [$SpecTclRestCommand::client sbindSpectra $args]
    }
}
#----------------------------------------------------------------------------
# Fit is a command ensemble and, therefore implemented as one:
#

namespace eval fit {
    namespace ensemble create
    namespace export create update delete list proc
    
    ##
    # fit::create
    #   Create a new fit.
    # @param name - name of the fit.
    # @param spectrum - spectrum name.
    # @param low, high - fit area of interest limits.
    # @param ftype - fit type.
    #
    proc create {name spectrum low high ftype} {
        return [$::SpecTclRestCommand::client fitCreate $name $spectrum $low $high $ftype]
    }
    ##
    # fit::update
    #
    #   Update some fits.
    # @param pattern - patter of fits to update.
    #
    proc update {{pattern *}} {
        return [$::SpecTclRestCommand::client fitUpdate $pattern]
    }
    ##
    # fit::delete
    #    Delete a fit.
    #
    # @param name name of the fit to delete.
    #
    proc delete {name} {
        return [$::SpecTclRestCommand::client fitDelete $name]
    }
    ##
    # fit::list
    #  List the fit results.
    #
    # @param pattern - pattern of fit names to match
    # @return list as described in the SpecTcl command reference guide.
    #
    proc list {{pattern *}} {
         
        set raw [$::SpecTclRestCommand::client fitList $pattern]
        
        set result [::list]
        
        foreach fit $raw {
        
            set name [dict get $fit name]
            set spectrum [dict get $fit spectrum]
            set type    [dict get $fit type]
            set low     [dict get $fit low]
            set high    [dict get $fit high]
            set params [::list]
            dict for {key value} [dict get $fit parameters] {
                lappend params [::list $key $value]
            }
            lappend result [::list $name $spectrum $type [::list $low $high] $params]
        }
        
        return $result
    }
    ##
    # fit::proc
    #  @param fit name.
    # @return procedure definiton to compute the fit.
    #
    proc proc {name} {
        return [$::SpecTclRestCommand::client fitProc $name]
    }
}
#-------------------------------------------------------------------------------
# Fold command - again a command ensemble.

namespace  eval fold {
    namespace export -apply -list -remove
    namespace ensemble create
    ##
    # -apply
    #   Apply a new fold.
    #
    # @param gate
    # @param args - the spectra to apply the fold to.\
    #
    proc -apply {gate args} {
        return [$::SpecTclRestCommand::client foldApply $gate $args]
    }
    ##
    # -list
    #    List the folds
    #
    # @param pattern - optional spectrum name match pattern.
    #
    proc -list {{pattern *}} {
        set rawInfo [$::SpecTclRestCommand::client foldList $pattern]
        set result [list]
        foreach fold $rawInfo {
            lappend result [list [dict get $fold spectrum ] [dict get $fold gate]]
        }
        return $result
    }
    ##
    # -remove
    #   Remove the fold from a spectrum.
    # @param spectrum
    #
    proc -remove {spectrum} {
        return [$::SpecTclRestCommand::client foldRemove $spectrum]
    }
}
#-------------------------------------------------------------------------------
# Channel command - a namespace ensemble.

namespace eval channel {
    namespace export -get -set
    namespace ensemble create
    
    ##
    # -get
    #    Retrieve the value of a channel.
    # @param name -spectrum name.
    # @param indices - channel indices.
    #
    proc -get {name indices} {
        return [$::SpecTclRestCommand::client channelGet $name {*}$indices]
    }
    ##
    # -set
    #   Change the value of a channel.
    #
    # @param name  - spectrum name.
    # @param indices - channel indices.
    # @param value   - new value.
    #
    proc -set {name indices value} {
        return [$::SpecTclRestCommand::client channelSet $name $value {*}$indices]
    }
}
#-------------------------------------------------------------------------------
#  Simulate clear - not that as of now, clear -id is deprecated.  We'll
#  see if anyone actually uses it because we're not implementing it -- for now.
#  - if we need to later we can get the spectrum list and figure out the names
#    to clear  -- later.

##
# clear
#   Simulate the SpecTcl clear command in terms of the REST API. Forms:
#   -   clear -all   - Clears all spectra.
#   -   clear name1 name2 ... - clears the named spectra.
#    -  clear -id id1 id2... - clears spectra by id - not *UNIMPLEMENTED*
#
proc clear {args} {
    if {[llength $args] == 0} {
        error "'clear' command requires arguments."
    }
    set option [lindex $args 0]
    if {$option eq "-all"} {
        return [$::SpecTclRestCommand::client spectrumClearAll]
    } else {
        foreach name $args {
            $::SpecTclRestCommand::client spectrumClear $name
        }
    }
}
#-------------------------------------------------------------------------------
# Project command simulation.
#

##
# project
#   Simulate the project command.  There are sevral forms of this:
#   - project source new {x | y} ?contour?
#   - project -snapshot source new {x | y} ?contour?
#   - project -nosnapshot source new {x | y} ?contour?
#   
proc project {args} {
    set snap 0;               # Default is no snapshot.
    if {[llength $args] < 3} {
        error "Project ??-no?snapshot? source new direction ?gate?"
    }
    set option [lindex $args 0]
    if {[string index $option 0] eq "-"} {
        
        set args [lrange $args 1 end]
        if {$option eq "-snapshot"} {
            set snap 1
        } elseif {$option eq "-nosnapshot"} {
            set snap 0
        } else {
            error "invalid 'project' option $option"
        }
    }
    #  Args -?no?snapshot should now have been stripped if present.
    
    set source [lindex $args 0]
    set new    [lindex $args 1]
    set direction [lindex $args 2]
    set contour [lindex $args 3] ;    # Empty string if not present
    
    return [$::SpecTclRestCommand::client spectrumProject \
        $source $new $direction $snap $contour]
}
#-----------------------------------------------------------------------------
# specstats
#    Get spectrum statistics.

##
# specstats
#    Simulate the spectrum statistics command.
#
# @param pattern - optional spectrum name pattern.
#
proc specstats { {pattern *} } {
    return [$::SpecTclRestCommand::client spectrumStatistics $pattern]
}
#-----------------------------------------------------------------------------
# treeparameter - can be a command ensemble
#

namespace eval treeparameter {
    namespace export -create -list -listnew -set -setinc -setbins -setunit \
        -setlimits -check -uncheck -version
    namespace ensemble create
    
    ##
    # -create
    #    Create a new tree parameter.
    #
    # @param name -name of the parameter.
    # @param low  -low limit.
    # @param high - high limit.
    # @param bins - axis binning
    # @param units - units of measure.
    #
    proc -create {name low high bins units} {
        return [$::SpecTclRestCommand::client treeparameterCreate \
            $name $low $high $bins $units \
        ]
    }
    ##
    # -list
    #     List the tree parameter properties.
    #
    # @param pattern - name match pattern.
    #
    proc -list {{pattern *}} {
        set raw [$::SpecTclRestCommand::client treeparameterList $pattern]
        
        set result [list]
        foreach p $raw {
            lappend result [list                                        \
                [dict get $p name]  [dict get $p bins] [dict get $p low] \
                [dict get $p hi] [::SpecTclRestCommand::_computeWidth $p] [dict get $p units] \
            ]
        }
        return $result
    }
    ##
    # -listnew
    #    List new tree parameters.
    #
    proc -listnew { } {
        return [$::SpecTclRestCommand::client treeparameterListNew]
    }
    ##
    # -set
    #   Set new characteristics  a tree parameter.
    #H
    # @note we ignore the value of inc as it's presumably a function of low, high
    #       and bins - this behaves differently than the native treeparameter command
    #       which requires inc to be reasonablly correct given those three
    #       characteristics.
    #
    proc -set {name bins low high inc units} {
        return [$::SpecTclRestCommand::client treeparameterSet  \
            $name $bins $low $high $units                      \
        ]
    }
    ##
    # -setinc
    #   Set the increment - which actually computes and sets the bin width.
    # @param name  Tree parameter name
    # @param inc   Floating point bin width.
    #
    proc -setinc {name inc} {
        return [$::SpecTclRestCommand::client treeparameterSetInc $name $inc]
    }
    ##
    # -setbins
    #
    #   Set the number of bins to put between low and high.
    #
    # @param name - treeparamteer name.
    # @param bins  - Desired bins.
    #
    proc -setbins {name bins} {
        return [$::SpecTclRestCommand::client treeparameterSetBins $name $bins]
    }
    ##
    # -setunit
    #   Set the a tree parameter's units of measure.
    #
    # @param name  - tree parameter name
    # @param units - new units of measure.
    proc -setunit {name units} {
        return [$::SpecTclRestCommand::client treeparameterSetUnits $name $units]
    }
    ##
    # -setlimits
    #   Sets new tree parameter low/high limits.
    #
    # @param name - name of the tree parameter.
    # @param low  - new low limit.
    # @param high - new high limit.
    #
    proc -setlimits {name low high} {
        return [$::SpecTclRestCommand::client treeparameterSetLimits \
            $name $low $high \
        ]
    }
    ##
    # -check
    #    Return a tree parameter check flag.
    #
    # @param name - tree parameter name.
    # @return boolean.
    #
    proc -check {name} {
        return [$::SpecTclRestCommand::client treeparameterCheck $name]
    }
    ##
    # -uncheck
    #    Reset a tree parameter's check flag.
    #
    # @param name - tree parameter name.
    #
    proc -uncheck {name} {
        return [$::SpecTclRestCommand::client treeparameterUncheck $name]
    }
    ##
    # -version
    #    Return the version of a tree parameter.
    #
    # @param name - name of the tree parameter.
    #
    proc -version {name} {
        return [$::SpecTclRestCommand::client treeparameterVersion]
    }
}
#-------------------------------------------------------------------------------
# Treevariable - namespace ensemble.
#
#  *UNIMPLEMENTED* - For now we do not implement the ability to set
#                    variables directly via the Tcl set command.
#                    this is theoretically possible using variable traces.
#
namespace eval treevariable {
    namespace export -list -set -check -setchanged -firetraces
    namespace ensemble create
    
    ##
    # -list
    #    Produce list of variables.
    # @param pattern - optional matching pattern.
    # @return list of lists sublists describe a variable with name, value, units, in that
    #         order.
    #
    proc -list {{pattern *}} {
        set raw [$::SpecTclRestCommand::client treevariableList]
        set result [list]
        
        foreach v $raw {
            set name [dict get $v name]
            if {[string match $pattern $name]} {;     #Cient side pattern filter.
                set value [dict get $v value]
                set units [dict get $v units]
                lappend result [list $name $value $units]
            }
        }
        
        return $result
    }
    ##
    # -set
    #   Set new value and units to a tree parameter.
    # @param name
    # @param value
    # @param units
    #
    proc -set {name value units} {
        return [$::SpecTclRestCommand::client treevariableSet $name $value $units]
    }
    ##
    # -check
    #    Return the changed flag.
    #
    # @param name - treevariable name.
    # @return boolean.
    #
    proc -check {name} {
        return [$::SpecTclRestCommand::client treevariableCheck $name]
    }
    ##
    # -setchanged
    #   Set a variables changed flag.
    #
    # @param name -name of the variable.
    #
    proc -setchanged {name} {
        return [$::SpecTclRestCommand::client treevariableSetChanged $name]
    }
    ##
    # -firetraces
    #   Fire the traces on tree variables that match a pattern.
    # @param pattern -  variables name patter on which traces are fired.
    # @note *UNSUPPORTED* getting traces fired in the client. In theory this
    #       _could_ be done with shadowed variables but we'll see how needed it
    #        is
    #
    proc -firetraces {{pattern *}} {
        return [$::SpecTclRestCommand::client treevariableFireTraces $pattern]
    }
}
#------------------------------------------------------------------------------
# filter namespace ensemble.

namespace eval filter {
    namespace export -new -delete -enable -disable -regate -file -list -format
    namespace ensemble create
    
    ##
    # -new
    #    Create a new filter.
    # @param name - filter name.
    # @param gate  - filter gate.
    # @param params - List of parameters
    #
    proc -new {name gate params} {
        return [$::SpecTclRestCommand::client filterCreate $name $gate $params ]
    }
    ##
    # -delete
    #    Delete a filter by name.
    #
    # @param name -the filter to delete.
    #
    proc -delete {name} {
        return [$::SpecTclRestCommand::client filterDelete $name]
    }
    ##
    # -enable
    #    Enable a filter to record data.
    #
    # @param name name of the filte.r
    #
    proc -enable {name} {
        return [$::SpecTclRestCommand::client filterEnable $name]
    }
    ##
    # -disable
    #   Disable a filter by name
    # @param name - filter name.
    #
    proc -disable {name} {
        return [$::SpecTclRestCommand::client filterDisable $name]
    }
    ##
    # -regate
    #    Change the gate that conditionalizes a filter's output.
    # @param name - filtername.
    # @param newgate -new gate name.
    #
    proc -regate {name newgate} {
        return [$::SpecTclRestCommand::client filterRegate $name $newgate]
    }
    ##
    # -file
    #   Set the output file of a filter.
    #
    # @param name - filter name.
    # @param path - file path that will be interpreted within SpecTcl.
    #
    # @note *UNIMPLEMENTED* - the ability to write files interpreted within the
    #    client - this is problematic as the client could be a different usr
    #     than SpecTcl.
    proc -file {name path} {
        return [$::SpecTclRestCommand::client filterFile $name $path]
    }
    ##
    # -list
    #   Provides the information about exsting filters the native filter -list
    #   command provide.s
    #
    # @param pattern - filter name matching glob pattern.#
    # @return list of lists - see filter -list in the command reference.
    #
    proc -list {{pattern *}} {
        set raw [$::SpecTclRestCommand::client filterList $pattern]
        set result [list]
        foreach f $raw {
            lappend result [list                                          \
                [dict get $f name] [dict get $f gate] [dict get $f file] \
                [dict get $f parameters] [dict get $f enabled]           \
                [dict get $f format]                                     \
            ]
        }
        return $result
    }
    ##
    # -format
    #    Set a new output format for a filter.
    #
    # @param name - filter name.
    # @param format - new format name.
    #
    proc -format {name format} {
        return [$::SpecTclRestCommand::client filterFormat $name $format]
    }
}
#-----------------------------------------------------------------------------
# Gate namespace ensemble - we use unknown to trampoline into the
#      create.
# !UNIMPLEMENTED! gate -list -byid
#
namespace eval gate {
    namespace export -new -list -delete -trace
    namespace ensemble create
    
    ##
    # -new
    #   Create a new gate.
    #
    # @param name -name of the new or edited gate.
    # @param gtype - Gate type.
    # @param description - gate description string.
    # !UNIMPLEMENTED T and F gates I think.
    proc -new {name gtype description} {
        # The specific thing we do depends on the gate type:
        
        if {$gtype in [list s gs]} {
            if {$gtype eq "gs"} {
                set parameter [lindex $description 1]
                set limits    [lindex $description 0]
            } else {
                set parameter [lindex $description 0]
                set limits   [lindex $description 1]
            }
            return [$::SpecTclRestCommand::client gateCreateSimple1D  \
                $name $gtype $parameter [lindex $limits 0] [lindex $limits 1] \
            ]
        } elseif {$gtype in [list c b  gc gb]} {
            if {$gtype in [list gc gb]} {
                set points [lindex $description 0]
                set xparameters [lindex $description 1]
                set yparameters [list]
            } else {
                set points [lindex $description 2]
                set xparameters [lindex $description 0]
                set yparameters [lindex $description 1]
            }
           
        
            return [$::SpecTclRestCommand::client gateCreateSimple2D       \
                $name $gtype                                                 \
                $xparameters $yparameters                                    \
                [::SpecTclRestCommand::_lselect $points 0]                  \
                [::SpecTclRestCommand::_lselect $points 1]                  \
            ]
        } elseif {$gtype in [list am em nm]} {
            return [$::SpecTclRestCommand::client gateCreateMask         \
                $name $gtype [lindex $description 0] [lindex $description 1] \
            ]
        } elseif {$gtype in [list + - * c2band]} {
            return [$::SpecTclRestCommand::client gateCreateCompound \
                $name $gtype $description  \
            ]
        } else {
            error "$gtype creation is not implemented."
        }
    }
    ##
    # -list
    #   List gate definitions... we have the handy _gateDictToDef utility to
    #   help us out
    # @param pattern - name matching pattern.
    proc -list {{pattern *}} {
        set raw [$::SpecTclRestCommand::client gateList $pattern]
        set result [list]
        foreach gate $raw {
            lappend result [::SpecTclRestCommand::_gateDictToDef $gate]
        }
        return $result
    }
    ##
    # -delete
    #   Delete the named gate
    #
    # @param name -name of the gate to delete.
    # @note *UNIMPLEMENTED* -id  option.
    #   
    proc -delete {name} {
        return [$::SpecTclRestCommand::client gateDelete $name]
    }
    ##
    # -trace
    #   manipulate gate traces 
    # @param type - trace type: add, delete, change
    # @param script - optional - the new gate trace of that type.
    # @note that we must use args because there's a difference between
    #       supplying no script and supplying an empty one.
    # @return any prior trace of that type.
    proc -trace {args} {
    
        if {[llength $args] == 0} {
            error "gate -trace requires a trace type (add delete change)"
        }
        if {[llength $args] > 2} {
            error "gate -trace too many arguments"
        }
        
        set traceType [lindex $args 0]
        set haveScript 0
        
        set script [list]
        if {[llength $args] == 2} {
            set haveScript 1
            set script [lindex $args 1]
        }
        
        if {$traceType eq "add"} {
            set varname SpecTclRestCommand::gateAddTrace
        } elseif {$traceType eq "delete" } {
            set varname SpecTclRestCommand::gateDeleteTrace
        } elseif {$traceType eq "change" } {
            set varname SpecTclRestCommand::gateChangedTrace
        } else {
            error "gate -trace $traceType is not a valid trace type."
        }
        
        set prior [set $varname];     # Prior trace name.
        if {$haveScript} {
            set $varname [list]
            
            lappend $varname $script;  # Supports future multi trace.
        }
        
        # Set up trace processing if it's not going yet.
        
        SpecTclRestCommand::_startTraceMonitoring
        
        return $prior
    }
}
##
# SpecTclRestCommand::_gateCreate
#    This is registered as an unknown processor for the gate namespace.
#    Doing this supports creation of gates with or without the -new option on
#    the gate command.  All we do is relay to gate::-new.
#
proc SpecTclRestCommand::_gateCreate {ns name args} {
    
    return [list ${ns}::-new $name]
}
namespace ensemble configure gate -unknown SpecTclRestCommand::_gateCreate
#-------------------------------------------------------------------------------
#  Simulate the integrate command.
#

##
# integrate
#    Get SpecTcl to integrate a region of interest (gate or coordinates)
#    And return the values to the user.
#
# @param name  spectrum name.
# @param roi   Area of interest specification which canbe any of:
#       -   A gate name that could be displayed on the spectrum.
#       -   For a 1d spectrum a low/high limit pair.
#       -   For a 2d spectrum a list of points.
# @return - see integrate in the SpecTcl command reference.
#
proc integrate {name roi} {
    set restROI [dict create]
    
    if {[llength $roi] == 1} {
        dict set restROI gate $roi
    } elseif {[llength $roi] == 2} {
        dict set restROI low [lindex $roi 0]
        dict set restROI high [lindex $roi 1]
    } else {
        dict set restROI  xcoord [::SpecTclRestCommand::_lselect $roi 0]
        dict set restROI  ycoord [::SpecTclRestCommand::_lselect $roi 1]
    }
    
    set raw [$::SpecTclRestCommand::client integrate $name $restROI]
    
    return [list                                                             \
        [dict get $raw centroid]  [dict get $raw counts]                     \
        [dict get $raw fwhm]                                                 \
    ]
}
#------------------------------------------------------------------------------
# Simulate parameter command - a namespace ensemble with unknown handler is used.
# !UNIMPLEMENTED! -byid list parameter - note this is supported by the REST interface.
# !UNIMPLEMENTED! -id on delete command. - note this is supported by the REST interface.
#
namespace eval parameter {
    namespace export -new -list -delete -trace -untrace
    namespace ensemble create
    ##
    # -new
    #   Create a new parameter.
    #
    # @param name the new parameter.
    # @param id   The id of the new parameter.
    # @param args This can have four forms:
    #        -   Empty list -no metadata.
    #        -  single element list containing an integer resolution
    #        -  single element list containing units
    #        -   Two element list containing resolution, and low,high,units list.
    # @return string - parameter name on success.
    #
    proc -new {name id args} {
        set metadata [SpecTclRestCommand::_computeParameterMetadata {*}$args]
        $::SpecTclRestCommand::client parameterNew $name $id $metadata
        return $name
    }
    ##
    # -list
    #   List parameters whose names match  a pattern.
    #
    # @param pattern - glob pattern defaults to * matchng everything.
    # @return see SpecTcl command reference for parameter command
    proc -list {{pattern *}} {
        set raw [$::SpecTclRestCommand::client parameterList $pattern]
        set result [list]

        foreach p $raw {
            
            lappend result [list [dict get $p name] [dict get $p id]           \
            [SpecTclRestCommand::_dictGetIfExists $p  resolution]  \
            [list                                                              \
                [SpecTclRestCommand::_dictGetIfExists $p  low]         \
                [SpecTclRestCommand::_dictGetIfExists $p  high]        \
                [SpecTclRestCommand::_dictGetIfExists $p units]   \
            ]]
        }
        
        return $result
    }
    ##
    # -delete
    #    Deletes a parameter.
    #
    # @param name -name of the parameter to delete.
    #
    proc -delete {name} {
        return [$::SpecTclRestCommand::client parameterDelete $name]    
    }
    ##
    # -trace
    #    Just append the script to the list of parameter trace scripts:
    #    SpecTclRestCommand::parameter
    #
    # @param script -the trace script
    proc -trace {script} {
        lappend SpecTclRestCommand::parameterTraces [list $script]
        
        SpecTclRestCommand::_startTraceMonitoring
    }
    ##
    # -untrace
    #    Remove a specific trace from the traces.
    #    -  If there are more than one matching scripts only the oldest one
    #       is removed.
    #    -  If there is no matching script this call silently does nothing.
    #
    # @param script - the script to remove.
    #
    proc -untrace {script} {
        set i 0
        foreach s $SpecTclRestCommand::parameterTraces {
            if {$s eq $script} {
                
                set SpecTclRestCommand::parameterTraces [lreplace \
                    $SpecTclRestCommand::parameterTraces $i $i     \
                ]
                
                break
            }
            incr i
        }
    }

}
##
# SpecTclRestCommand::_parameterCreate
#    This is the unknown handler of the parameter namespace ensemble.
#    It relays to parameter -new in order to allow that option to be
#    omitted when createing a parameter definition.
#
# @param ns - namespace name (::parameter)
# @param name - The subcommand is actually the parameter name.
# @param args - The remaining command words.
#
proc SpecTclRestCommand::_parameterCreate {ns name args} {
    return [list ${ns}::-new $name {*}$args]
}
namespace ensemble configure parameter -unknown SpecTclRestCommand::_parameterCreate

#------------------------------------------------------------------------------
#  pseudo command simulation as namespace ensemble.
#  Note the -new option does not really exist in the SpeTcl command set
#  (I think) but adding it makes using a namespace ensemble possible via a
#  redirecting unknown handler.
#
namespace eval pseudo {
    namespace export -new -list -delete
    namespace ensemble create
    
    ##
    # -new
    #    Create a new pseudo parameter that depends on existing raw parameters
    #
    # @param name - pseudo parameter being created the parameter must already
    #               be defined.
    # @param parameters - the names of existing parameters the pseudo depends on.
    # @param body - Tcl script body that computes the pseudo.
    #
    proc -new {name parameters body} {
        return [$::SpecTclRestCommand::client pseudoCreate $name $parameters $body]
    }
    ##
    # -list
    #    Return a list of pseudo definitions.
    #
    # @param pattern - pattern that specifies which psuedos will be listed.
    # @return list - see SpecTcl command reference page on pseudo.
    #
    proc -list {{pattern *}} {
        set raw [$::SpecTclRestCommand::client pseudoList $pattern]
        set result [list]
        
        foreach p $raw {
            lappend result [list                                      \
                [dict get $p name] [dict get $p parameters]           \
                [dict get $p computation]                             \
            ]
        }
        
        return $result
    }
    ##
    # -delete
    #    Delete each pseudo parameter named in the argument list.
    #
    proc -delete {args} {
        foreach name $args {
            $::SpecTclRestCommand::client pseudoDelete $name
        }
    }
}
proc SpecTclRestCommand::_createPseudo {ns name args} {
    return [list ${ns}::-new $name]
}
namespace ensemble configure pseudo -unknown SpecTclRestCommand::_createPseudo

#-------------------------------------------------------------------------------
# Simulate the sread command via REST.

##
# sread
#   Read a spectrum in from file.
# @param args - the command line parameter. See the SpecTcl
#               command reference to see the possible forms.
# @note the filename is interpreted in the SpecTcl process.
#   -  file names are paths as seen by SpecTcl
#   -   file descriptors are those in SpecTcl and can only be gotten via
#       a call to command opening the file.
#
proc sread {args} {
    if {[llength $args] == 0} {
        error "sread requires parameters"
    }
    set filename [lindex $args end];    # Always last.
    set options [lrange $args 0 end-1]
    
    set optDict [dict create]
    
    # Process options.  -format has a parameter so we need to do as below:
    
    for {set i 0} {$i < [llength $options]} {incr i} {
        set option [lindex $options $i]
        if {$option eq "-format"} {
            incr i
            dict set optDict format [lindex $options $i]
        } elseif {$option eq "-snapshot"} {
            dict set optDict snapshot 1
        } elseif {$option eq "-nosnapshot"} {
            dict set optDict snapshot 0
        } elseif {$option eq "-replace"} {
            dict set optDict replace 1
        } elseif {$option eq "-noreplace"} {
            dict set optDict replace 0
        } elseif {$option eq "-bind"} {
            dict set optDict bind 1
        } elseif {$option eq "-nobind"} {
            dict set optDict bind 0
        } else {
            "$option  is an unrecognized sbind option."
        }
    }
        
    return [$::SpecTclRestCommand::client sread $filename $optDict]
    
    
}
#-----------------------------------------------------------------------------
#  ringformat simulator.

##
# ringformat
#    Simulate the ring format command in terms of the REST API.
#
# @param version major.minor version format.
#
proc ringformat {version} {
    set v [split $version .]
    return [$::SpecTclRestCommand::client ringformat {*}$v]
}  
#----------------------------------------------------------------------------
# scontents simulation.
#

##
# scontents
#    We support scontents spectrum name or scontents -json spectrum name.
#
# @param args - either name of a spectrum or -json name of a spectrum.
#
proc scontents {args} {
    set json 0;            # Assume it's just Tcl format.
    if {[llength $args] == 1} {
        set name $args
    } elseif {[llength $args] == 2} {
        if {[lindex $args 0] ne "-json"}  {
            error "scontents only accepts a -json optional option"
        }
        set json 1
        set name [lindex $args 1]
    } else {
        error "Too many command line parameter for scontents"
    }
    set raw [dict get [$::SpecTclRestCommand::client scontents $name] channels]
    
    
    if {$json} {
        return [::SpecTclRestCommand::_scontentsToJson $raw ]
    }  else {
        return [::SpecTclRestCommand::_scontentsToList $raw]
    }
}
#-------------------------------------------------------------------------------
# shared memory access.

##
# shmkey
#   Return the shared memory key.
# @return string
#
proc shmemkey { } {
    return [$::SpecTclRestCommand::client shmemkey]
}
##
# shmsize
#   @return integer size of shared memory in bytes.
#
proc shmemsize { } {
    return [$::SpecTclRestCommand::client shmemsize]
}
#------------------------------------------------------------------------------
# spectrum command simulation in a namespace ensemble.
#

# !UNIMPLEMENTED! -byid on spectrum -list
# !UNIMPLEMENTED! -list -id
# !UNIMPLEMENTED! -delete -id.
#
namespace eval spectrum {
    namespace export -new -list -delete -trace
    namespace ensemble create
    
    ##
    # -new
    #    Create a new spectrum.
    #
    # @oaram name - new spectrum name.
    # @param type - new spectrum type
    # @param parameters - parameters
    # @param axes axis specifications.
    # @param data type - defaults to long.
    # @return string  - spectrum name.
    proc -new {name type parameters axes {datatype long}} {
        $::SpecTclRestCommand::client spectrumCreate \
            $name $type $parameters $axes [dict create chantype $datatype]
        return $name
    }
    ##
    # -list
    # list spectrum
    #
    # @param pattern - pattern of spectrum names to match
    # @return list of spectrum definitions as described in the SpecTcl
    #         command reference
    # @note - all ids are zero.
    proc -list {args} {
        set pattern *;               # Default pattern.
        set showgates 0
        if {[llength $args] == 1} {
            set pattern [lindex $args 0]
            if {$pattern eq "-showgate"} {
                set showgates 1
                set pattern *
            }
        } elseif {[llength $args] == 2} {
            set option [lindex $args 0]
            set pattern [lindex $args 1]      
            if {$option ne "-showgate"} {
                error "Unrecognized option $option - only -showgate is allowwed"
            }
            set showgates 1
        } elseif {[llength $args] > 2} {
            error "Too many arguments to spectrum -list"
        }
        set raw [$::SpecTclRestCommand::client spectrumList $pattern]
        set result [list]
        foreach s $raw {
            if {[dict exists $s id]} {
                set item [dict get $s id]
            } else {
                set item 0
            }
            
            lappend item                                                        \
                [dict get $s name] [dict get $s type]                           \
                [dict get $s parameters]                                        \
                [::SpecTclRestCommand::_axesDictToAxesList [dict get $s axes]]  \
                [dict get $s chantype]
            if {$showgates} {
                lappend item [dict get $s gate]
            }
            lappend result $item
        }
        return $result
    }
    ##
    # -delete
    #    Delete spectra. This can take either -all or name1...
    #  @param args - remainder of the command line.
    #   
    proc -delete {args} { 
        if {([llength $args] == 1) && ($args eq "-all")} {
            set names [list]
            foreach s [$::SpecTclRestCommand::client spectrumList] {
                lappend names [dict get $s name]
            }
        } elseif {[llength $args] > 0} {
            set names $args
        } else {
            error "spectrum -delete needs parameters"
        }
        
        foreach name $names {
            
            $::SpecTclRestCommand::client spectrumDelete $name
        }
    }
    ##
    # -trace
    #   Implement traces in termsof the trace poll system.
    #   
    #
    # @param what - what to do add or delete.
    # @param script - script to add or remove.
    # @return the prior value of the supplied trace.
    #
    proc -trace {what args} {
        set haveScript 0
        set script  [list]
        if {[llength $args] != 0} {
            set script $args
            set haveScript 1
        }
        
        if {$what eq "add"} {
            set scriptVar SpecTclRestCommand::spectrumAddTraces
            
        } elseif {$what eq "delete"} {
            set scriptVar SpecTclRestCommand::spectrumDeleteTraces
        } else {
            error "spectrum -trace invalid trace type $what"
        }
        set prior [set $scriptVar]
        if {$haveScript} {
            set $scriptVar [list $script]
        }
        SpecTclRestCommand::_startTraceMonitoring
        return $prior
    }
}
##
# SpecTclRestCommand::_createSpectrum
#   Is the unknown subcommand of the spectrum namespace ensemble.
#   It supports spectrum creation by omitting the -new flag.
#
# @param ns - namespace (spectrum).
# @param name - name of spectrum being created.
# @param args  - remaining arguments.
#
proc SpecTclRestCommand::_createSpectrum {ns name args} {
    return [list ${ns}::-new $name]
}
namespace ensemble configure spectrum -unknown SpecTclRestCommand::_createSpectrum
#----------------------------------------------------------------------------
# namespace ensemble that implements the unbind command.
# !UNIMPLEMENTED! passing a list of xids not spetrum names

namespace eval unbind {
    namespace export unbind -id -all -trace -untrace
    namespace ensemble create
    
    ##
    # unbind
    #   Unbind a list of spectra.
    #
    # @param args - the list of names.
    #
    proc unbind {args} {
        return  [$::SpecTclRestCommand::client unbindByName $args]
    }
    ##
    # Unbind by ids:
    # @param args -the ids.
    #
    proc -id {args} {
        return [$::SpecTclRestCommand::client unbindById $args]   
    }
    ##
    # Unbind all spectra.
    #
    proc -all {} {
        return [$::SpecTclRestCommand::client unbindAll]   
    }
    ##
    # -trace
    # Add a trace script
    #
    proc -trace {script} {
        
        lappend SpecTclRestCommand::unbindTraces $script
        SpecTclRestCommand::_startTraceMonitoring
    }
    ##
    # -untrace
    #    Remove a trace script.
    #
    proc -untrace {script} {
        set index [lsearch -exact $SpecTclRestCommand::unbindTraces $script]
        if {$index < 0} {
            error "unbind -trace no trace script '$script' to unbind"
        } else {
            set SpecTclRestCommand::unbindTraces \
                [lreplace $SpecTclRestComand::unbindTraces $index $index]
        }
    }
    
}
##
# SpecTclRestCommand::_unbindByName
#
# Unknown subcommand handler for unbind.
#
# @param ns    - namespace
# @param name1 - first name (subcommand).
# @param args  - remaining parameters (optionally more names).
#
proc SpecTclRestCommand::_unbindByName {ns name1 args} {
    return [list ${ns}::unbind $name1]
}
namespace ensemble configure unbind -unknown SpecTclRestCommand::_unbindByName

#---------------------------------------------------------------------------
# ungate command.

##
# ungate
#    Ungate a list of spectra via REST.
#
# @param args - list of spectra to ungate.
#
proc ungate {args} {
    return [$::SpecTclRestCommand::client ungate $args]
}

#--------------------------------------------------------------------------------
# version command

##
# version
#   Return the SpecTcl Version.
# @return string - the version string.
#
proc version { } {
    set raw [$::SpecTclRestCommand::client version]
    return [dict get $raw major].[dict get $raw minor]-[dict get $raw editlevel]
}
#-----------------------------------------------------------------------------
# swrite simulation
##
# swrite
#   Forms allowed are swrite -format fmt names... and swrite names...
#
# @param args - the arguments.
#
proc swrite {args} {
    set format ascii
    if {[llength $args] > 2} {
        set option [lindex $args 0]
        if {$option eq "-format"} {
            # Require at least 3 parameters.
            
            if {[llength $args < 4]} {
                error "swrite -format must have at least a format, file and spectrum"
            } else {
                set format [lindex $args 1]
                set filename [lindex $args 2]
                set names [lrange $args 3 end]
            }
        } else {
            set filename [lindex $args 0]
            set names    [lrange $ags 1 end]
        }
    } elseif {[llength $args] == 2} {
        set filename [lindex $args 0]
        set names    [lindex $args 1]
    } else {
        error "swrite needs at least a filename and a spectrum."
    }
    return [$::SpecTclRestCommand::client $swrite $filename $names $format]
}
#----------------------------------------------------------------------------
# analysis control commands  (start, stop)

##
# start
#   Start analysis from the source.
#
proc start { } {
    return [$::SpecTclRestCommand::client start]
}
##
# stop
#   Stop analysis from the source.
#
proc stop {} {
    return [$::SpecTclRestCommand::client stop]
}
#-----------------------------------------------------------------------------
#  Roottree command ensemble.

namespace eval roottree {
    namespace export create delete list
    #
    #  The map below avoids conflicts between roottree::list and ::list
    #
    namespace ensemble create -map [dict create list _list create create delete delete]
    
    ##
    # create
    #   Create a new root tree
    # @param name - tree name.
    # @param patterns -list of parameter patterns.
    # @param gate - optional gate that determines which events get written to
    #                  the tree.
    #
    proc create {name patterns {gate {}}} {
        return [$::SpecTclRestCommand::client rootTreeCreate $name $patterns $gate]
    }
    ##
    # delete
    #   Delete the named tree.
    #
    # @param name - name of the tree to remove.
    #
    proc delete {name} {
        return [$::SpecTclRestCommand::client rootTreeDelete $name]
    }
    ##
    # list
    #   List the root trees that match the optional (defaults to *)
    #   pattern.
    #
    # @param pattern - pattern tree must match.
    #
    proc _list { {pattern *}} {
        set raw [$::SpecTclRestCommand::client rootTreeList $pattern]
        set result [list]
        
        foreach tree $raw {
            lappend result [list                                             \
                [dict get $tree tree] [dict get $tree parameters]           \
                [dict get $tree gate]                                       \
            ]
        }
        
        return $result
    }
    
}
#------------------------------------------------------------------------------
# pman command namespace ensemble.
#

namespace eval pman {
    namespace export mk ls current ls-all ls-evp use add rm clear clone
    namespace ensemble create
    
    ##
    # mk
    #   Create a new pipeline.
    # @param name - pipeline name.
    #
    proc mk {name} {
        return [$::SpecTclRestCommand::client pmanCreate $name]
    }
    ##
    # ls
    #   List names of the pipelines.
    #
    proc ls {{pattern *}} {
        return [$::SpecTclRestCommand::client pmanList $pattern]
    }
    ##
    # current
    #   Provide the name and processors in the current pipeline.
    #
    proc current { } {
        set raw [$::SpecTclRestCommand::client pmanCurrent]
        return [::SpecTclRestCommand::_pipeDictToList $raw]
    }
    ##
    # ls-all
    #   List all pipelines and their details.
    #
    proc ls-all {{pattern *}} {
        set raw [$::SpecTclRestCommand::client pmanListAll $pattern]
        set result [list]
        foreach pipe $raw {
            lappend result [::SpecTclRestCommand::_pipeDictToList $pipe]
        }
        return $result
    }
    ##
    # ls-evp
    #   List names of event processors.
    #
    # @param pattern - pattern to match.
    #
    proc ls-evp {{pattern *}} {
        return [$::SpecTclRestCommand::client pmanListEventProcessors]
    }
    ##
    # use
    #   Selects a current event processing pipeline.
    #
    # @param name - name to use.
    #
    proc use {name} {
        return [$::SpecTclRestCommand::client pmanUse $name]
    }
    ##
    # add
    #   Add an event processor to the end of a pipeline.
    # @param pipe  - pipeline name.
    # @Param processor - event processor name.
    #
    proc add {pipe processor} {
        return [$::SpecTclRestCommand::client pmanAdd $pipe $processor]
    }
    ##
    # rm
    #   Remove an event processor from a pipeline.
    #
    # @param pipe - name of the pipeline.
    # @param processor - name of the event processor to remove.
    #
    proc rm {pipe processor} {
        return [$::SpecTclRestCommand::client pmanRemove $pipe $processor]
    }
    ##
    # clear
    #   Empty a pipeline of all processors.
    #
    proc clear {pipe} {
        return [$::SpecTclRestCommand::client pmanClear $pipe]
    }
    ##
    # clone
    #   Make a copy of an event processor pipeline
    #
    # @param existing - name of existing pipe
    # @param new      - name of the new pipe.
    #
    proc clone {existing new} {
        return [$::SpecTclRestCommand::client pmanClone $existing $new]
    }
    
}
#------------------------------------------------------------------------------
# namespace ensemble to simulate the evbunpack command.

namespace eval evbunpack {
    namespace export create addprocessor list
    namespace ensemble create
    
    ##
    # create
    #   Create a new event builder unpacker.
    #
    # @param name - name of the unpacker.
    # @param freq - MHz of the clock.
    # @param base - basename of the diagnostics parameters.
    #
    proc create {name freq base} {
        return [$::SpecTclRestCommand::client evbCreate $name $freq $base]
    }
    ##
    # addprocessor
    #   Assign a pipeline to unpack data from a specific source id.
    # @param name - processor name.
    # @param sid   - source id.
    # @param pipe - pipeline to use.
    #
    proc addprocessor {name sid pipe} {
        return [$::SpecTclRestCommand::client evbAdd $name $sid $pipe]
    }
    ##
    # list
    #   List the names of the evb processors.
    #
    # @param pattern - names must match this pattern.
    #
    proc list {{pattern *}} {
        return [$::SpecTclRestCommand::client evbList $pattern]
    }
}
##
# command
#   Execute a command in the server.
#
# @param cmd - the command.
# @return command's results.
#
proc execCommand {cmd} {
    return [$::SpecTclRestCommand::client command $cmd]
}
##
# updateVariables
#    Update the SpecTcl globals from the server:
#
proc updateVariables { } {
    set varDict [$::SpecTclRestCommand::client getVars]
    dict for {name value} $varDict {
        set ::$name $value
    }
        
    
}
    
##
# maintainVariables
#    Periodically updates the variables
#    Requires an event loop.
# @param seconds - seconds between updates.
#
proc maintainVariables {seconds} {
    set ms [expr {$seconds * 1000}]
    
    updateVariables
    set SpecTclRestCommand::varsAfterId [after $ms maintainVariables $seconds]
}

##
# isRemote
#   @return 1 indicating the script is running as a REST client.
#
proc isRemote { } {
    return 1
}

##
# mirror
#   Gets list of SpecTcl Mirrors.
#
proc mirror {list {pattern *}} {
    if {$list ne "list"} {
        error "Invalid subcommand"
    }
    return [$::SpecTclRestCommand::client mirror $pattern]
}