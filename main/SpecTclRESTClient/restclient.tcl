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
# @file   restclient.tcl
# @brief  Base level client code of SpecTcl REST server.
# @author Ron Fox <fox@nscl.msu.edu>
#

package provide SpecTclRESTClient 1.0
package require snit
package require http
package require json

##
# @class SpecTclRestClient
#
#   Client to the SpecTcl REST server.  The goal is to implement, essentialy,
#   every SpecTcl command as a REST server/client method set.
#
#  OPTIONS:
#    -host  - host SpecTcl is running in (default is localhost)
#    -port  - Port SpecTcl REST server is listening  (defaults to 8080).
#
#  METHODS:
#    applyGate  - Apply gates to spectra.
#    applyList   - List gate applications.
#
#    attachSource - Attach data source (must still start).
#    attachList   - Return attachment strings.
#
#    sbindAll     - sbind -all
#    sbindSpectra - sbind spectrum list.
#    sbindList    - List spectrum bindings.
#
#    fitCreate   - Create a new fit.
#    fitUpdate   - Update a fit.
#    fitDelete   - Delete a fit.
#    fitList     - List fits.
#
#    foldApply   - Apply a gamma gate as a fold to a spectrum.
#    foldList    - List folds.
#    foldRemove  - Unfold a spectrum.
#
#    channelSet  - Set a spectrum channel value.
#    channelGet  - Retrieve a spectrum channel value.
#
#    spectrumClear  - Clear spectrum.
#    spectrumClearAll - Clear all spectra.
#
#    spectrumProject - Project a 2d spectrum.
#
#    spectrumStatistics - Get spectrum oveflow/underflow statistics
#
#    treeparameterCreate
#    treeparameterList
#    treeparameterListNew
#    treeparameterSet
#    treeparameterSetInc
#    treeparameterSetBins
#    treeparameterSetSetUnits
#    treeparameterSetLimits
#    treeparameterCheck
#    treeparameterUncheck
#    treeparameterVersion
#
#    treevariableList
#    treevariableSet
#    treeVariableCheck
#    treeVariableSetChanged
#    treeVariableFireTraces
#
#    filterCreate
#    filterDelete
#    filterEnable
#    filterDisable
#    filterRegate
#    filterFile
#    filterList
#    filterFormat
#
#    gateList
#    gateDelete
#    gateCreateSimple1D
#    gateCreateSimple2D
#    gateCreateMask
#    gateCreateCompound
#
#    integrate
#
#    parameterNew
#    parameterDelete
#    parameterList
#
#    pseudoCreate
#    pseudoList
#    pseudoDelete
#
#   sread
#
#   ringformat
#
#   scontents
#
#   shmemkey
#   shmemsize
#
#   shmupdate_set
#   shmupdate_get
#
#   spectrumList
#   spectrumDelete
#   spectrumCreate
#
#   unbindByName
#   unbindById
#   unbindAll
#
#   ungate
#
#   version
#
#   swrite
#
#   start
#   stop
#
#   rootTreeCreate
#   rootTreeDelete
#   rootTreeList
#
#   pmanCreate
#   pmanList
#   pmanCurrent
#   pmanListAll
#   pmanListEventProcessors
#   pmanUse
#   pmanAdd
#   pmanRemove
#   pmanClear
#   pmanCLone
#
#   evbCreate
#   evbAdd
#   evbList
#
#   command
#
snit::type SpecTclRestClient {
    option -host -default localhost
    option -port -default 8080
    option -debug -default 0
    
    variable domain spectcl
    
    constructor args {
        $self configurelist $args
    }
    #--------------------------------------------------------------------------
    # Private variables
    #
    
    ##
    #  _makeUrl
    #    Create an arbitrary request URL - note that all request are GET so
    #    that's easy.
    # @param subdomain - the subdomain of the URL.
    # @param query     - dict containing the query key/value pairs.
    #                    We'll shimmer that into a list format.
    # @return string - the complete URL.
    #
    method _makeUrl {subdomain {query {}}} {
        set querypart [http::formatQuery {*}$query]
        set url http://$options(-host):$options(-port)/$domain/$subdomain?$querypart
        return $url
    }
    ##
    # _request
    #   Given a URI:
    #   - Starts the request.
    #   - waits for completion.
    #   - Decodes the resulting JSON into a dict.
    #   - If the status field is not OK error with the vaLue of 'detail'
    #   - otherwise, return the full JSON response.
    #
    # @param url - url complete with any query suffixes.
    # @return dict - the full JSON response
    #
    method _request {url} {
        if {$options(-debug)} {
            puts "Request: '$url'"
        }
        set token [http::geturl $url]
        http::wait $token
        set status [http::ncode $token]
        
        # Httpd error:
        
        if {$status != 200} {
            set message "[http::error $token] [http::data $token]"
            http::cleanup token
            error $message
        }
        
        set rawData [http::data $token]
        
        # It's possible this is deflated if e.g. scontents on a 2d spectrum.
        
        upvar #0 $token state
        if {$state(coding) eq "deflate"} {
            set rawData [zlib inflate $rawData]
        }
        
        if {$options(-debug)} {
            puts $rawData;     
        }
        set parseOk [catch {
            set json [json::json2dict $rawData]
        } msg]
        http::cleanup $token
        
        #  Could be errors parsing the JSON or errors flagged in the
        #  json:
        
        if {$parseOk} {
            error "JSON Parse Failed: $msg : $::errorInfo : {$rawData}"
        }
        # Debugging here - if there's no detail then likely we have  a
        # server error - let's throw an error with the rawdData as the message:
        #
        if {![dict exists $json status ]} {
            error "SpecTcl Server error: $rawData"
        }
        
        set status [dict get $json status]
        if {$status ne "OK"} {
            error "Request failed by SpecTcl: $status :  [dict get $json detail]"
        }
        
        
        # Success return what interests the caller.
        
        return $json
    }
    ##
    #  _listToQueryList
    #    Given a query parameter name and a list of items,
    #    returns  list of parameter-name item1 paramter-name item2...
    # @param queryArg - name of the query parameter.
    # @param items    - items to put into the list.
    #
    proc _listToQueryList {queryArg items} {
        set result [list]
        foreach item $items {
            lappend result $queryArg $item
        }
        return $result
    }
    
    #---------------------------------------------------------------------------
    #
    #  Apply command jackets.
    #
    
    ##
    # applyGate
    #   Apply a gate to a list of spectra.  Note each application of a gate
    #   to one spectrum is a transaction.
    #
    # @param gate - name of the gate to apply.
    # @param spectra - list of spectra
    #
    method applyGate {gate spectra} {
        set query [dict create gate $gate]
        
        # The request returns nothing of interest so just loop.
        
        foreach spectrum $spectra {
            dict set query spectrum $spectrum
            $self _request [$self _makeUrl apply/apply $query]
        }
    }
    ##
    # applyList
    #   List gate applications
    # @param pattern - spectrum match pattern optional -defaults to *
    #                  which matches everything.
    # @return list of dicts dict containing spectrum and gate (name).
    #
    method applyList {{pattern *}} {
        set query [dict create pattern $pattern]
        set json [$self _request [$self _makeUrl apply/list $query]]
        return [dict get $json detail]
    }
    #--------------------------------------------------------------------------
    # Attach command jackets.
    
    ##
    # attachSource
    #   Attach a data source.
    # @param stype   - 'pipe' or 'file'
    # @param source - Data source specification (e.g. filename for stype 'file').
    # @param size  - optional blocking factor - defaults to 8192.
    # @param format - Optional data format defaults to ring
    #
    method attachSource {stype source {size 8192} {format ring}} {
        set query [dict create type $stype source $source size $size format $format]
        $self _request [$self _makeUrl attach/attach $query]
    }
    ##
    # attachList
    # @return string that describes the current attachment:
    #
    method attachList {} {
        set result [$self _request [$self _makeUrl attach/list [dict create]]]
        return [dict get $result detail]
    }
    #-------------------------------------------------------------------------
    #  Sbind command jackets.
    
    ##
    # sbindAll
    #   Sbind all spectra.
    #
    method sbindAll {} {
        $self _request [$self _makeUrl sbind/all [dict create]]
    }
    
    ##
    # sbindSpectra
    #
    #  Sbind a list of spectra.
    #  
    # @param spectra - the spectra to bind.
    #
    method sbindSpectra {spectra} {
        set qparams [_listToQueryList  spectrum $spectra]
        
        $self _request [$self _makeUrl sbind/sbind $qparams]
    }
    ##
    # sbindList
    #   Return a list of the bindings.
    #
    # @param pattern - glob pattern to limit the spectra looked at.
    # @return list of dicts containing spectrumid - the id of the spectrum,
    #                 name - name of the spectrum and -binding -the binding id.
    #
    method sbindList {{pattern *}} {
        set result [$self _request [$self _makeUrl sbind/list [dict create pattern $pattern]]]
        
        return [dict get $result detail]
    }
    #-------------------------------------------------------------------------
    # fit command jackets.
    
    ##
    # fitCreate
    #    Create a new fit object.
    #
    # @param name = fit name.
    # @param spectrum - fit spectrum.
    # @param low, high - Channel fit limits.
    # @param ftype - fit type.
    #
    method fitCreate {name spectrum low high ftype} {
        $self _request [$self _makeUrl fit/create [dict create             \
            name $name spectrum $spectrum low $low high $high              \
            type $ftype                                                     \
        ]]
    }
    ##
    # fitUpdate
    #    Update the paramters of fits that match the pattern.
    #
    # @param pattern - glob pattern to match the fit names update.
    #
    method fitUpdate {{pattern *}} {
        $self _request [$self _makeUrl fit/update [dict create pattern $pattern]]
    }
    ##
    #  fitDelete
    #   Delete a single fit
    # @param name  - Name of the fit to delete.
    #
    method fitDelete {name} {
        $self _request [$self _makeUrl fit/delete [dict create name $name]]
    }
    ##
    # fitList
    #    List information about fits.
    #    list of dicts containing name - fit name, spectrum -spectrum name.
    #    type -fit type, low, high -fit limits and parameter - a dict specific
    #    to the fit type that contains the fit parameters.
    #
    # @param pattern - glob pattern. Fits with names that match this are in the
    #         result.
    #
    method fitList {{pattern *}} {
        set json [$self _request [$self _makeUrl fit/list [dict create pattern $pattern]]]
        
        return [dict get $json detail]
        
    }
    ##
    # fitProc
    #   Return a proc that can evaluate a fit.
    #
    # @param name - name of the fit.
    #
    # @return    - String definition of the proc
    #
    method fitProc {name} {
        set result [$self _request [$self _makeUrl fit/proc [dict create name $name]]]
        return [dict get $result detail]
    }
        
    #--------------------------------------------------------------------------
    # fold command jackets.
    #
    
    ##
    # foldApply
    #    Apply a fold to one or more spectra.
    # @param gate  - name of the gate to apply the fold to.
    # @param spectra - list of spectra to fold with that gate.
    #
    method foldApply {gate spectra} {
        set queryList [list gate $gate {*}[_listToQueryList spectrum $spectra]]
        $self _request [$self _makeUrl fold/apply $queryList]
    }
    ##
    # foldList
    #    List folds  on spectra that match the pattern
    # @param pattern - only spectra that match this optional glob pattern are
    #                  listed.  Defaults to * which matches all spectra.
    # @return list of dicts which contain spectrum (a spectrum name) and
    #         gate the folding gate.
    #
    method foldList {{pattern *}} {
        set info [$self _request [$self _makeUrl fold/list [dict create pattern $pattern]]]
        return [dict get $info detail]
    }
    ##
    # foldRemove
    #   Remove a fold from a spectrum.
    #
    # @param spectrum  - name of spectrum to unfold.
    #
    method foldRemove {spectrum} {
        $self _request [$self _makeUrl fold/remove [dict create spectrum $spectrum]]
    }
    #-------------------------------------------------------------------------
    # jackets channel set/get.
    #
    #  Note we freely use that -1 is an invalid channel coordinate.
    #
    
    ##
    # channelGet
    #  Return the value of a channel
    #
    # @param spectrum  - name of a spectrum.
    # @param xchannel  - x channel value
    # @param ychannel  - optional y channel value (only for 2d spectra).
    #
    method channelGet {spectrum xchannel {ychannel -1}} {
        
        set qparams [dict create spectrum $spectrum xchannel $xchannel]
        if {$ychannel != -1} {
            dict set qparams ychannel $ychannel
        }
        
        set info [$self _request [$self _makeUrl channel/get $qparams]]
        return [dict get $info detail]
    }
    ##
    # channelSet
    #   Set the value of a channel
    #
    # @param spectrum  - name of spectrum.
    # @param value     - value to set.
    # @param xchannel  - xchannel to set.
    # @param ychannel  - optional (only needed if spectrum is 2d) y channel.
    #
    method channelSet {spectrum value xchannel {ychannel -1}} {
        set qparams \
            [dict create spectrum $spectrum value $value xchannel $xchannel]
        if {$ychannel != -1} {
            dict set qparams ychannel $ychannel
        }
        
        $self _request [$self _makeUrl channel/set $qparams]
    }
    #--------------------------------------------------------------------------
    # clear jackets.. these spectcl/spectrum/zero
    
    ##
    # clearSpectrum
    #    clear the contents of one spectrum (actually the name can be a glob pattern)
    #
    # @param name  - name of the spectrum to clear.
    #
    method spectrumClear {name} {
        $self _request [$self _makeUrl spectrum/zero [dict create pattern $name]]
    }
    ##
    # spectrumClearAll
    #
    method spectrumClearAll {} {
        return [$self spectrumClear *]
    }
    #-------------------------------------------------------------------------
    # project command jacket.
    
    ##
    # spectrumProject
    #
    # @param old - old spectrum name.
    # @param new  - new spectrum name.
    # @param direction - Projection direction x|y.
    # @param snapshot - Boolean snapshot flag.
    # @param contour - If provided only the part within the contour is projected.
    #
    method spectrumProject {old new direction snapshot {contour ""} } {
        set qdict [dict create \
            snapshot $snapshot source $old newname $new direction $direction
        ]
        if {$contour ne ""} {
            dict set qdict contour $contour
        }
        $self _request [$self _makeUrl project $qdict]
    }
    #--------------------------------------------------------------------------
    # specstats jacket.
    #
    
    ##
    # spectrumStatistics
    #   @param pattern  - the glob pattern matching spectrum names we want
    #                     stats for.
    #   @return list of dicts containing name, underflows and overflows keys.
    #
    method spectrumStatistics {{pattern  *}} {
        set info [$self _request [$self _makeUrl specstats [dict \
            create pattern $pattern]]]
        return [dict get $info detail]
    }
    #-------------------------------------------------------------------------
    # treeparameter jackets
    
    ##
    # treeparameterCreate
    #   Make a new tree parameter.
    # @param name - name of the parameter.
    # @param low  - Low limit
    # @param high  - High limit.
    # @param bins  - number of recommended bins.
    # @param units - optional units (defaults to "").
    # No useful return value.
    #
    method treeparameterCreate {name low high bins {units ""}} {
        set qparams [dict create \
            name $name low $low high $high bins $bins units $units \
        ]
        $self _request  [$self _makeUrl /parameter/create $qparams]
    }
    ##
    #    treeparameterList
    #   List the tree parameters.
    # @param filter - GLOB filter on the names of the tree parameters listed.
    #    note that this defaults to * which lists all of them.
    #
    # @return list of dicts each dict descrbing a tree parameter with the keys:
    #    - name   - name of a parameter.
    #    - bins   - number of bins.
    #    - low    - Low limit.
    #    - hi     - high limt.
    #    - units  - unts.
    #
    # @note the REST interface returns all parameter not just the tree parameters.
    #      we filter the non-tree parameters out.
    #
    method treeparameterList {{filter *}} {
        set raw [$self _request [$self _makeUrl                   \
            /parameter/list [dict create filter $filter]             \
        ]]
        set result [list]
        foreach param [dict get $raw detail] {
            if {[dict exists $param bins]} {
                lappend result $param;              # it's a tree parameter.
            }
        }
        
        return $result
        
    }
    ##
    #    treeparameterListNew
    #   Returns the list of tree parameters that were created via
    #    treeparameter -create by any means.
    # @return list of strings.
    #
    method treeparameterListNew {} {
        set raw [$self _request [$self _makeUrl /parameter/listnew [dict create]]]
        return [dict get $raw detail]
    }
    ##
    #    treeparameterSet
    #  Sets new values for all aspects of a tree parameter (treeparameter -set).
    #
    # @param name - name of the parameter.
    # @param bins - Suggested binning.
    # @param low  - Suggested low limit.
    # @param high - Suggested high limit.
    # @param units - optional units (defaults to "").
    #
    #  Nothing useful is returned.
    #
    method treeparameterSet {name bins low high {units ""} } {
        set pdict [dict create\
            name $name bins $bins low $low high $high units $units \
        ]
        $self _request [$self _makeUrl /parameter/edit $pdict]
    }   
    ##
    #    treeparameterSetInc
    #  Modify the increment of a parameter.  This requires some fancy footwork
    #  as the inc is derived from low, high and bins and is
    #   (high-low)/bins  We're going to set only bins  so bins = |(high-low)/inc|
    # @param name  - name of the parameter.
    # @param newinc  - New channel increment.
    #
    method treeparameterSetInc {name newinc} {
        set current [$self treeparameterList $name]
        if {[llength $current] ==1 } {
            set info [lindex $current 0]
            set low [dict get $info low]
            set high [dict get $info hi]
            set bins [dict get $info bins]
            set newbins [expr int(abs(($high-$low)/$newinc) + 0.5)]; #rounded.
            $self _request [$self _makeUrl \
                /parameter/edit [dict create name $name bins $newbins]  \
            ]
        } else {
            error "$name is not an unambiguous tree parameter name."
        }
    }
    ##
    #    treeparameterSetBins
    # Set the number of bins in a tree parameter.
    #
    # @param name - name of the parameter.
    # @param newbins - new bins value.
    #
    method treeparameterSetBins {name newbins} {
        set qdict [dict create name $name bins $newbins]
        $self _request [$self _makeUrl /parameter/edit $qdict]
    }
    ##
    #    treeparameterSetUnits
    #     Change the units of a tree parameter.  Note that currently there's
    #     no way to remove units (set them to an empty string), since an empty
    #     units string will just not change anything.
    #
    # @param name - name of the parameter.
    # @param units -Units name
    #
    method treeparameterSetUnits {name units} {
        if {$units eq ""} {
            error "Setting units name to '' is not supported"
        }
        set qdict [dict create name $name units $units]
        $self _request [$self _makeUrl /parameter/edit $qdict]
    }
    ##
    #    treeparameterSetLimits
    #    set the limit meta data on a tree parameter.
    #
    # @param name - name of the parameter.
    # @param low  - low limit.
    # @param high  - high limit.
    #
    method treeparameterSetLimits {name low high} {
        set qdict [dict create name $name low $low high $high]
        $self _request [$self _makeUrl /parameter/edit $qdict]
    }
    ##
    #    treeparameterCheck
    #  Return the tree parameter check flag for a single tree parameter.
    #  this is set if anybody changed the tree parameter definition since
    #  the program started.
    #
    # @param name - name to check
    #
    method treeparameterCheck {name} {
        set data [$self _request [$self _makeUrl /parameter/check \
                [dict create name $name]]]
        return [dict get $data detail]
    }
    ##
    #    treeparameterUncheck
    #  uncheck a tree paramter modification flag
    #
    # @param name - name to uncheck.
    #
    method treeparameterUncheck {name} {
        $self _request [$self _makeUrl /parameter/uncheck [dict create name $name]]
    }
    #    treeparameterVersion
    #
    #  Return the tree parameter version string.
    #
    # @return string.
    #
    method treeparameterVersion {} {
        set data [$self _request [$self _makeUrl /parameter/version [dict create]]]
        return [dict get $data detail]
    }
    #-------------------------------------------------------------------------
    #  treevariable command jackets.
    
    ##
    #    treevariableList
    #  Returns a list of dicts where each dict describes a tree variable.
    #  the dicts have the keys name, value and units with obvious meanings.
    #
    method treevariableList {} {
        set info [$self _request [$self _makeUrl /treevariable/list [dict create]]]
    
        return [dict get $info detail]
    }
    ##
    #    treevariableSet
    #      Set value and units of a tree variable.
    #
    # @param name - name of the variable.
    # @param value - new value.
    # @param units - new units.
    #
    method treevariableSet {name value units} {
        set qdict [dict create                                            \
            name $name value $value units $units                         \
        ]
        $self _request [$self _makeUrl treevariable/set $qdict]
    }
    ##
    #    treevariableCheck
    #  Returned the changed flag
    # @param name -name of the variable.
    #
    method treevariableCheck {name} {
        set info [
            $self _request [$self _makeUrl treevariable/check [dict create name $name]] \
        ] 
        return [dict get $info detail]
    }
    ##
    #    treevariableSetChanged
    #   Set the changed flag of a variable.
    # @param name - treevariable name.
    #
    method treevariableSetChanged {name} {
        $self _request [$self _makeUrl treevariable/setchanged [dict create name $name]]
    }
        
    
    #    treeVariableFireTraces
    #
    # Fire any traces associated with a tree variable.  Note the traces are
    # fired in the server not the client.  This can cause some interesting
    # issues with porting UIs to work remotely that must be handled by the
    # next layer of software.
    #
    # @param pattern - name to match that filters the set of traces fired.
    #
    # 
    method treevariableFireTraces {{pattern *}} {
        $self _request [$self _makeUrl treevariable/firetraces [dict create]]
    }
    #------------------------------------------------------------------------
    #  Filter command jackets
    #

    ##
    #    filterCreate
    #  Create a new filter.
    # @param name - filter name.
    # @param gate - filter gate.
    # @param parameters - List of paramete names to write in events in the
    #              output file.
    # @return - nothing useful
    #
    method filterCreate {name gate parameters} {
        dict set qDict name $name
        dict set qDict gate $gate
        lappend qDict {*}[_listToQueryList parameter $parameters]
        
        
        
        $self _request [$self _makeUrl filter/new $qDict]
    }
    ##
    #    filterDelete
    #  Deletes an existin filter.
    #
    # @param name -filter name.
    #
    method filterDelete {name} {
        $self _request [$self _makeUrl filter/delete [dict create name $name]]
    }
    ##
    #    filterEnable
    #   Enable an existing filter.
    # @param name -the filter name.
    #
    method filterEnable {name} {
        $self _request [$self _makeUrl  filter/enable [dict create name $name]]
    }
    ##
    #    filterDisable
    #  Disable an existing filter.
    # @param name - name of existing filter,.
    #
    method filterDisable {name} {
        $self _request [$self _makeUrl filter/disable [dict create name $name]]
    }
    ##
    #    filterRegate
    # Change the gate that's applied to a filter.
    #
    # @param name - filter name.
    # @param gate - Filter gate.
    #
    method filterRegate {name gate} {
        $self _request [$self _makeUrl \
            filter/regate [dict create name $name gate $gate] \
        ]
    }
    ##
    #    filterFile
    # Set the output file of a filter.
    #
    # @param name  - filter name.
    # @param path  - Filename path. Note that this is interpreted in the
    #               context of this server and must be writable by this process,
    #               not the client.  As such it is recommended to use full path
    #               names.
    #
    method filterFile {name path} {
        $self _request [$self _makeUrl filter/file [dict create                     \
            name $name file $path                                       \
        ]]
    }
    ##
    #    filterList
    #  List the filters.
    #
    # @param pattern - optional GLOB pattern  that restricts the set of
    #      filters listed to those that match it. Defaults to '*' if not
    #      provided, which matches everything.
    # @return list of dicts - each dict describes a filter and contains the
    #         following keys:
    #         - name -the filter name.
    #         - gate - the filter gate name.
    #         - file - The filter output file (empty string if there isn't one).
    #         - parameters - list of parameters written for each event.
    #         - enabled - "enabled" or "disabled" depending on the enable status.
    #         - format - format string (normally 'xdr').
    #
    method filterList {{pattern *}} {
        set info [$self _request [$self _makeUrl  \
            filter/list [dict create pattern $pattern]] \
        ]
        return [dict get $info detail]
    }
        
    
    #    filterFormat
    #  Set the filter file format.
    #
    # @param name - filter name.
    # @param format - format string.
    #
    method filterFormat {name format} {
        set qparams [dict create name $name format $format]
        $self _request [$self _makeUrl filter/format $qparams]
    }
    #---------------------------------------------------------------------------
    # gates Wrappers.
    #

    ##    
    #    gateList
    #  Provide a list of all the gates whose names match a glob pattern.
    #
    #  @param pattern - the glob pattern (optional defaults to *)
    #  @return list of dicts.  The dicts have some fixed (ominpresent) keys and
    #          others that depend on the gate gatetype:
    #          -  name - gate name.
    #          -  type - SpecTcl Gate gatetype.
    #          -  gates dependent gate names if compound gate gatetype (*, +, -)
    #          -  parameters - Parameters needed by the gate if gatetype
    #             s,c,b,c2band, gs, gb, gc, em am nm
    #          -  points list of points, each point is an x/y pair. Present for
    #             2-d simple gates of gatetype: c, b, c2band, gb, gc
    #          -  low  - Low limit in 1d gates like (s, gs).
    #          -  high - high limit of 1d gates (s,gs)
    #           - value - mask value of mask gates (em, am, nm).
    #
    method gateList {{pattern *}} {
        set info [$self _request [$self _makeUrl \
            gate/list [dict create pattern $pattern] \
        ]]
        return [dict get $info detail]
    }
    ##
    #    gateDelete
    #  Deletes a gate. The gate isn't actually deleted because we don't know how
    #  to handle dependencies.  What actually happens is that the gate is turned
    #  into a false gate.
    #    Potential depenencies include membership in the gate lst of a compound
    #    gate and application to spectra.
    #
    # @param name - name of the gate.
    #
    method gateDelete {name} {
        $self _request [$self _makeUrl gate/delete [dict create name $name]]
    }
    ##
    #    gateCreateSimple1D
    #  Create a simple 1d gate.  This must be of gatetype s, or gs.   It can have
    #  one (s) or more parameters.
    #
    # @param name -name of the gate - if the gate already exists this gate definition
    #     will replace it.
    # @param gatetype - gate gatetype must be gs, or s, or an error will be raised.
    # @param parameters - list of gate parameters must only be one if $gatetype eq "s"
    # @param low  low limit of gate
    # @param high high limit of gate.
    #
    method gateCreateSimple1D {name gatetype parameters low high} {
        if {$gatetype ni [list s gs]} {
            error "The only gate gatetypes gateCreateSimple allows are 's' and 'gs'"
        }
        if {($gatetype eq "s") && ([llength $parameters] != 1)} {
            error "gateCreateSimple - slice gates must have only one parameter"
        }
        # Build the query parameter dict.
        
        set qparams [dict create                                   \
            name $name type $gatetype low $low high $high              \
        ]
        lappend qparams {*}[_listToQueryList parameter $parameters]
        
        $self _request [$self _makeUrl gate/edit $qparams]
        
    }
    ##
    #    gateCreateSimple2D
    #  Create a simple 2d gate.
    #
    # @param name -gate name.  If existing the existing gate will be replaced.
    # @param gatetype - Gate gatetype must be: b,c, gb, gc
    # @param xparameters - list of parameters on the x axis (1 if b,c).
    # @param yparameters - list of parameters on the y axis (1 if b,c).
    # @param xcoords      - Gate x-coordinates.
    # @param ycoords      - gate y-coordinates.
    # @note for gc, gb, the parameters must all be in xparameters and the yparameters
    #                       are ignored.
    #
    method gateCreateSimple2D {name gatetype xparameters yparameters xcoords ycoords} {
        
        #  Error checking:
        
        if {$gatetype ni [list b c gb gc]} {
            error "gateCreateSimple2D requires a gatetype that is 'b', 'c', 'gb' or 'gc'"
        }
        
        if {($gatetype in [list b c]) &&
            (([llength $xparameters] != 1) || ([llength $yparameters] != 1))} {
            error "gateCreasteSimple2D gate gatetype $gatetype only allows one x and y parameter"
        }
        #  Now build the query dict:
        
        set qparams [dict create name $name type $gatetype]
        if {$gatetype in "b c"} {
            lappend qparams {*}[_listToQueryList xparameter $xparameters]
            lappend qparams {*}[_listToQueryList yparameter $yparameters]    
        } else {
            lappend qparams {*}[_listToQueryList parameter $xparameters]
            lappend qparams {*}[_listToQueryList parameter $yparameters]; # just in case
        }
        lappend qparams {*}[_listToQueryList xcoord $xcoords]
        lappend qparams {*}[_listToQueryList ycoord $ycoords]
        
        
        $self _request [$self _makeUrl gate/edit $qparams]
    }
    ##
    #    gateCreateMask
    #  Create a mask gate.
    #
    # @param name -name of the gate.
    # @param gatetype -type of gate, must be one of em, am, nm.
    # @param parameter - the gate parameter.
    # @param mask      - The mask value.
    #
    method gateCreateMask {name gatetype parameter mask} {
        if {$gatetype ni [list em am nm]} {
            error "gateCreateMask gatetype must be 'em', 'am' or 'nm'"
        }
        $self _request [$self _makeUrl                               \
            gate/edit                                                \
            [dict create name $name type $gatetype parameter $parameter \
            value $mask]                                             \
        ]
    }
    ##
    #    gateCreateCompound
    # Create a compound gate, that is, a gate that depends on other gates.
    #
    # @param name  - Name of the gate.
    # @param gatetype  - gate gatetype, one of + - * c2band
    # @param gates - The gates the compound gate depends on.
    #
    method gateCreateCompound {name gatetype gates} {
        # Error Checking:
        
        if {$gatetype ni [list + - * c2band]} {
            error "gateCreateCompound gatetype must be one of '+', '-', '*', 'c2band'"
        }
        if {($gatetype eq "c2band") && ([llength $gates] != 2)} {
            error "gateCreateCompound 'c2band' must only have 2 gates."
        }
        if {($gatetype eq "-") && ([llength $gates] != 1)} {
            error "gateCreateCompound '-' must only have one gate"
        }
        set qdict [dict create name $name type $gatetype]
        lappend qdict {*}[_listToQueryList gate $gates]
        
        $self _request [$self _makeUrl gate/edit $qdict]
    }
    #----------------------------------------------------------------------------
    # integrate REST command jacket.
    
    ##
    # integrate
    #   Because of the flexibility of the integrate REST/command, we're a bit
    #   creative in how we do this:
    #
    # @param name - spectrum name.
    # @param roi  - Region of interest.  This must be expressed either as:
    #          - gate - gate name.
    #          - low, high - low/high limits of a 1d spectrum.
    #          - xcoord/ycoord - point list of 2-d ROI.
    #
    # @return dict containing:
    #    -  centroid - one or two elements centroid coordinates.
    #    -  fwhm     - one or two elements FWHM extents.
    #    -  counts   - number of counts in the ROI
    #
    method integrate {name roi} {
        # how we handle the ROI depends on whether or not it has x/y coords:
        
        
        if {[dict exists $roi xcoord]} {
            set querydict [dict create spectrum $name]
            if {![dict exists $roi ycoord]} {
                error "integrate - xcoords require ycoords"
            }
            lappend querydict {*}[_listToQueryList xcoord [dict get $roi xcoord]]
            lappend querydict {*}[_listToQueryList ycoord [dict get $roi ycoord]]
        } else {
            set querydict [dict merge [dict create spectrum $name] $roi]
        }
    
        set info [$self _request [$self _makeUrl integrate $querydict]]
        return [dict get $info detail]
    }
    ##
    #    parameterNew
    # Create a new parameter raw parameter (using the parameter command):
    #  @param name -name of the parameter.
    #  @param id - parameter number.
    #  @param metadata - Dict containing parameter metadata:
    #     - resolution - # bits of resolution.
    #     - low        - Parameter low limit (requires high and units).
    #     - high       - Parameter high limit (requires low and units).
    #     - units     - Units of measure string.
    #
    method parameterNew {name number metadata} {
        set pdict [dict merge [dict create name $name number $number] $metadata]
        $self _request [$self _makeUrl rawparameter/new $pdict]
    }     
    #
    #     parameterDelete
    #  Delete a parameter.
    #   @param name - optional parameter name make it an empty string if you
    #                 want to specify the id.
    #   @param id   - optional parameter id.
    #
    method parameterDelete {{name {}} {id {}}} {
        if {$name ne ""} {
            set pdict [dict create name $name]
        } elseif {$id ne ""} {
            set pdict [dict create id $id]
        } else {
            error "parameterDelete needs either a name or an id."
        }
        $self _request [$self _makeUrl rawparameter/delete $pdict]
    }
    ##
    #    parameterList
    #  List the parameters either by providing a pattern to match the
    #  names  or an id
    #
    # @param pattern - match this pattern to names. set to "" if you want to specify ids.
    # @param id      - id to request.
    #
    method parameterList {{pattern *} {id {}}} {
        if {$pattern ne ""} {
            set pdict [dict create pattern $pattern]
        } elseif {$id ne ""} {
            set pdict [dict create id $id]
        } else {
            error "Either a pattern or id is required"
        }
        set info [$self _request [$self _makeUrl rawparameter/list $pdict]]
        return [dict get $info detail]
    }
    #---------------------------------------------------------------------------
    # pseudo command jackets.
    
    ##
    #    pseudoCreate
    # Create a new pseudo parameter
    #  @param  name -name of the pseudo.
    #  @param  parameters - parameters the pseudo depends on.
    #  @param  body       - Computational body of the proc that computes the pseudo.
    #
    method pseudoCreate {name parameters body} {
        set qdict [dict create pseudo $name computation $body]
        lappend qdict {*}[_listToQueryList parameter $parameters]
        $self _request [$self _makeUrl pseudo/create $qdict]
    }
    ##
    #    pseudoList
    #  Produce a list of pseudo parameter definitions.
    # @param pattern - optional name match glob pattern which defaults to *
    # @return list of dicts. Each dict describes a pseudo and has the keys:
    #    -  name  -name of the pseudo.
    #    -  parameters -  parameters the pseudo depends on.
    #    -  computation - the body of the proc that computes the pseudo.
    #
    method pseudoList {{pattern *}} {
        set info [$self _request [$self _makeUrl \
            pseudo/list [dict create pattern $pattern]  \
        ]]
        return [dict get $info detail]
    }
    ##
    #    pseudoDelete
    #  deletes a pseudo parameter definition.
    # @param name - name of the pseudo to delete.
    #
    method pseudoDelete {name} {
        $self _request [$self _makeUrl                       \
            pseudo/delete [dict create name $name]           \
        ]
    }
    #---------------------------------------------------------------------------
    # jacket around the sread command.
    
    ##
    # sread
    #    Read a spectrum from file.
    # @param filename - file to read from.
    # @param opts  - dict with any or none of the following option keys:
    #       -   format - file format (defaults to ascii).
    #       -   snapshot - Flag, true to make a snapshot spectrum (1 by default).
    #       -   replace  - flag true to replace any existing spectrum (0 by default).
    #       -   bind     - Bind flag, true by default.
    #
    method sread {filename opts} {
        dict set opts filename $filename
        
        $self _request [$self _makeUrl sread $opts]
    }
    #--------------------------------------------------------------------------0
    # jacket ringformat command.
    
    ##
    # ringformat
    #    Set the ringbuffer format.
    # @param major - major version.
    # @param minor - minor version (defaults to 0).
    #
    method ringformat {major {minor 0}} {
        $self _request [$self _makeUrl /ringformat                        \
            [dict create major $major minor $minor]                       \
        ]
    }
    #--------------------------------------------------------------------------
    #  scontents jacket.
    #
    
    ##
    # scontents
    #    return spectrum contents.
    # @param name -name of the spectrum.
    # @return dict containing spectrum contents and metadata.
    #     The dict has the following keys:
    #      - xoverflow   - number of overflows on x axis  (or 1d)
    #      - xunderflow  - number of underflows on x axis (or 1d)
    #      - yoverflow   - Number of overflows on y axis (2d only)
    #      - yunderflow   - number of undeflows on y axis.
    #      - channels    - array of dicts containing nonzero channel data. each dict has:
    #                 *  x - X channel number (or channel if 1d).
    #                 *  y - Y channel number (2d)
    #                 *  v - Value of that channel (will never be zero).
    # @note  - sadly it seems that tclhttp is not able to handle the deflated data
    #          raising a 'data error'  Therefore we'll turn off compression.
    #          which, in large 2d spectra is a performance hit.
    method scontents {name} {
        set info [$self _request [$self _makeUrl          \
            spectrum/contents [dict create name $name compress 0]    \
        ]]
        return [dict get $info detail]
    }
    #------------------------------------------------------------------------
    # shared memory info jackets.
    
    ##
    # shmemkey
    #    Returns the shared memory key.  This can be used in a system local
    #    to SpecTcl to locate the shared memory region it uses as display
    #    data.
    #
    # @return The SYS-V shared memory key used by spectcl.
    #
    method shmemkey {} {
        set info [$self _request [$self _makeUrl shmem/key [dict create]]]
        return [dict get $info detail]
    }
    ##
    # shmemsize
    #    Returns the number of megabytes specified as the
    #    DisplayMemory size.  This can be  used, in conjunction with
    #    shmemkey to obtain a mapping to the SYS-V shared memory region
    #    spectcl uses for displayed spectra.
    #
    # @return The integer size in bytes.
    #
    method shmemsize {} {
        set info [$self _request [$self _makeUrl shmem/size [dict create]]]
        return [dict get $info detail]
                  
    }
    ##
    #  shmupdate_set
    #     Set the update rate for the shared memory.  This is rustogramer only.
    #
    #  @param seconds - number of seconds between updates.

    method shmupdate_set seconds {
        set info [$self _request [$self _makeUrl sbind/set_update [dict create seconds $seconds]]]
        
    }
    ##
    #  shmupdate_get
    #    Retrieve the number of seconds between shared memory updates:
    #
    method shmupdate_get {} {
        set info [$self _request [$self _makeUrl sbind/get_update [dict create]]]
        return [dict get $info detail]
    }
    #---------------------------------------------------------------------------
    # spectrum command jackets.
    
    ##
    #   spectrumList
    #  List spectra whose names match a pattern.
    #
    # @param pattern - The pattern (which can include Glob wildcard characters),
    #                  this defaults to "*" which matches all spectra.
    # @return List of dicts that contain:
    #        -   id - id of the spectrum (only present if REST defines it).
    #        -   name - spectrum name.
    #        -   type - spectrum type.
    #        -   parameter - list of parameters needed by the spectrum.
    #        -   axes - List of axis definitions (dict with low, high, bins keys).
    #        -   chantype - data type of the channel.
    #        -   gate  - Name of the gate that's applied to the spectrum.
    #
    method spectrumList {{pattern *}} {
        set info [$self _request [$self _makeUrl               \
            spectrum/list [dict create filter $pattern]        \
        ]]
        return [dict get $info detail]
    }
    ##
    #   spectrumDelete
    #   Delete a spectrum that's already defined.
    #
    #   @param name -  name of the spectrum to delete.
    #
    method spectrumDelete {name} {
        $self _request [$self _makeUrl spectrum/delete [dict create name $name]]
    }
    ##
    #   spectrumCreate
    #
    #  Create a new spectrum.  Some parameters are required but projection
    #  spectrum (2dmproj) has additional parameter that are in the opts
    #
    # @param name - new, unique spectrum name.
    # @param type - Spectrum type.
    # @param parameters - list of parameters the spectrum depends on.
    # @param axes - List of axes
    # @param opts - Options  These can be:
    #          -   chantype - the channel data type (defaults to long)
    #          -   direction - only used in 2dmproj where this is the
    #                         projection direction.
    #          -   roigate   - Only used inthe 2dmproj type where this is the
    #                          gate that defines the region of interest.
    #
    method spectrumCreate {name stype parameters axes {opts {}}} {
        set qparams [dict create \
            name $name type $stype parameters $parameters axes $axes \
        ]
        
        set qparams [dict merge $qparams $opts]
        
        $self _request [$self _makeUrl spectrum/create $qparams]
    }
    #--------------------------------------------------------------------------
    #  Jackets for the unbind operations.
    
    ##
    #   unbindByName
    #  Unbind spectra given a list of names.
    # 
    # @param names -names of the spectra to unbind.
    #
    method unbindByName {names} {
        $self _request [$self _makeUrl unbind/byname \
            [_listToQueryList name $names]       \
        ]
    }
    ##
    #   unbindById
    #  Unbind a list of spectrum ids.
    #
    # @param ids - list of ids to unbind.
    #
    method unbindById {ids} {
        $self _request [$self _makeUrl unbind/byid     \
            [_listToQueryList id $ids]
        ]
    }
    ##
    #   unbindAll
    # unbind all spectra.
    #
    method unbindAll {} {
        $self _request [$self _makeUrl unbind/all [dict create]]
    }
    #-------------------------------------------------------------------------
    # jackets to the ungate operation.
    
    ##
    # ungate
    #    Ungate one or more spectra.
    # @param names - list of spectrum names to ungate.
    method ungate {names} {
        $self _request [$self _makeUrl ungate [_listToQueryList name $names]]
    }
    #--------------------------------------------------------------------------
    # version command jacket.
    
    ##
    # version
    #   @return dict with the keys: major, minor and editlevel
    #
    method version {} {
        set vsn [$self _request [$self _makeUrl version [dict create]]]
        return [dict get $vsn detail]
    }
    #-------------------------------------------------------------------------
    # swrite command jacket.
    
    ##
    # swrite
    #   Write spectra to file.
    #
    # @param filename - name of the file to write.
    # @param spectra - names of spectra to write.
    # @param format  - optional format that defaults to ascii.
    #
    method swrite {filename spectra {format ascii}} {
        set qdict [dict create file $filename format $format]
        lappend qdict {*}[_listToQueryList spectrum $spectra]
        
        $self _request [$self _makeUrl swrite $qdict]
    }
    #---------------------------------------------------------------------------
    # analysis control jackets.

    ##    
    #   start
    #  Start analyzing from the data source.
    #
    method start {} {
        $self _request [$self _makeUrl analyze/start ]
    }
    ##
    #   stop
    # stop analysis.
    #
    method stop {} {
        $self _request [$self _makeUrl analyze/stop]
    }
    #---------------------------------------------------------------------------
    #  roottree command jackets.
    #
    
    ##
    #   rootTreeCreate
    #  Create a new root tree.
    # @param name  - Name of the tree.
    # @param parameterPatterns - list of parameter patterns
    # @param gate  - Optional gate that conditionalizes events booked into the
    #                tree.
    #
    method rootTreeCreate {name parameterPatterns {gate ""}} {
        set qdict [dict create tree $name]
        lappend qdict {*}[_listToQueryList parameter $parameterPatterns]
        if {$gate ne ""} {
            lappend qdict gate $gate
        }
        $self _request [$self _makeUrl roottree/create $qdict]
    }
    ##
    #   rootTreeDelete
    #  Delete an existing root tree.
    #
    # @param name - name of the tree to delete.
    #
    method rootTreeDelete {name} {
        $self _request [$self _makeUrl                            \
            roottree/delete [dict create tree $name]             \
        ]
    }
    ##
    #   rootTreeList
    # List the properties of all of the defined root trees.
    #
    # @param pattern - optional pattern matching the names of the trees to list
    #                  defaults to * which matches all trees.
    # @return list of dicts that define the trees. Each dict has the keys:
    #     -   tree - name of the tree.
    #     -   parameters - list of parameter patterns that determine what's
    #     -   gate     - The name of the gate that conditionalizes the
    #                    booking of events into the tree.
    #
    method rootTreeList  {{pattern *}} {
        set info [$self _request [$self _makeUrl          \
            roottree/list [dict create pattern $pattern]] \
        ]
        return [dict get $info detail]
    }
    #----------------------------------------------------------------------------
    # pman command jackets:

    ##        
    #   pmanCreate
    #  Create a new, empty pipeline.
    #
    # @param name - name of the new pipeline
    #
    method pmanCreate {name} {
        $self _request [$self _makeUrl pman/create [dict create name $name]]
    }
    ##
    #   pmanList
    #  List the pipelines that match the GLOB pattern supplied.
    # @param pattern - optional matching pattern, defaults to * which matches
    #                  everything.
    #
    method pmanList {{pattern *}} {
        set info [$self _request [$self _makeUrl \
            pman/ls [dict create pattern $pattern] \
        ]]
        return [dict get $info detail]
    }
    ##
    #   pmanCurrent
    # Provide information about the current pipeline.
    # returns a dict containng name - the name of the pipeline and
    #   processors the ordered list of event processors in the pipeline.
    #
    method pmanCurrent {} {
        set info [$self _request [$self _makeUrl pman/current [dict create]]]
        return [dict get $info detail]
    }
    ##
    #   pmanListAll
    #  List all the event processors that match the pattern.
    #
    # @param pattern - pattern to match, defaults to *
    # @return list of pipeline definitions as described in pmanCurrent.
    #
    method pmanListAll {{pattern *}} {
        set info [$self _request [$self _makeUrl \
            pman/lsall [dict create pattern $pattern]] \
        ]
        return [dict get $info detail]
    }
    ##
    #   pmanListEventProcessors
    # List the names of matching event processors.
    #
    # @param pattern - pattern to match.
    # @return list of event processor names.
    #
    method pmanListEventProcessors {{pattern *}} {
        set info [$self _request [$self _makeUrl \
            pman/lsevp [dict create pattern $pattern] \
        ]]
        return [dict get $info detail]
    }
    ##
    #   pmanUse
    #  Choose an event procesing pipeline.
    # @param pipeline - name of the pipeline.
    #
    method pmanUse {pipeline} {
        $self _request [$self _makeUrl pman/use [dict create name $pipeline]]
    }
    ##
    #   pmanAdd
    #  add an event processor to the end of a processing pipeline.
    #
    # @param pipeline  - name of the pipeline.
    # @param processor -name of the processor.
    #
    method pmanAdd {pipeline processor} {
        set qdict [dict create pipeline $pipeline processor $processor]
        $self _request [$self _makeUrl pman/add $qdict]
    }
    ##
    #   pmanRemove
    #  Remove an event processor from a pipeline.
    #
    # @param pipeline - name of the pipeline.
    # @param processorr - event processor to remove.
    #
    method pmanRemove {pipeline processor} {
        set qdict [dict create pipeline $pipeline processor $processor]
        $self _request [$self _makeUrl pman/rm $qdict]
    }
    ##
    #   pmanClear
    # Remove all event processors from a pipeline
    #
    # @param pipeline - name of the pipeline
    #
    method pmanClear {pipeline} {
        $self _request [$self _makeUrl \
            pman/clear [dict create pipeline $pipeline] \
        ]
    }
    ##
    #   pmanCLone
    # Duplicate an event processor.
    #
    # @param source - processor to duplicate.
    # @param new    - new event processor to make.
    #
    method pmanClone {source new} {
        set qdict [dict create source $source new $new]
        $self _request [$self _makeUrl pman/clone $qdict]
    }
    #-------------------------------------------------------------------------
    #  evbunpack command jacket.
    #
    
    ##
    #   evbCreate
    #  Create an event builder unpacker.
    # @param name - name for the unpacker.
    # @param frequency - timestamp clock in MHz.
    # @param basename - Base name of the tree parameters created for diagnostics.
    #
    method evbCreate {name frequency basename} {
        set qparams [dict create                                \
            name $name frequency $frequency basename $basename   \
        ]
        $self _request [$self _makeUrl evbunpack/create $qparams]
    }
    ##
    #   evbAdd
    #   Add an event processing pipeline to process a source.
    #
    # @param name    - name of the event built event processor.
    # @param source  - Id of the source it processes.
    # @param pipe    - Name of the pipe to process data from that source.
    #
    method evbAdd {name source pipe} {
        set qdict [dict create name $name source $source pipe $pipe]
        $self _request [$self _makeUrl evbunpack/add $qdict]
    }
    ##
    #   evbList
    #  List the currently defined event builder event processor.s
    #
    # @param pattern - Optional glob pattern limiting the responses.
    # @return list of event processor names for event built data.
    #
    method evbList {{pattern *}} {
        set result [$self _request [$self _makeUrl \
            evbunpack/list [dict create pattern $pattern]] \
        ]
        return [dict get $result detail]
    }
    #-------------------------------------------------------------------------
    
    ##
    # command
    #   execute an arbitrary commnd.
    # @param cmd - command test
    # @return - what the command returns.
    #
    method command {cmd} {
        set info [$self _request [$self _makeUrl script [dict create command $cmd]]]
        
        return [dict get $info detail]
    }
    ##
    # getVars
    #   Return the SpecTcl variables dict.
    #
    method getVars { } {
        return [dict get [$self _request [$self _makeUrl shmem/variables]] detail]
    }
    #-------------------------------------------------------------------------
    # traces are handled a bit strangely because normally they are
    # server side events.. We establish an interest in traces,
    # poll for them and then end our interest.
    #
    
    ##
    # traceEstablish
    #   Establish an interest in traces.
    #
    # @param retention - maxiumum # seconds traces are kept in our trace buffer.
    # @return integer  - Token to use in done and fetch.
    #
    method traceEstablish  {retention} {
        return [dict get [$self _request [$self _makeUrl          \
            trace/establish [dict create retention $retention]]]  \
        detail]
    }
    ##
    # traceDone
    #   Stop buffering of traces on our behalf by the server.
    #
    # @param token - Token that identifies us - return value from traceEstablish.
    #
    method traceDone {token} {
        $self _request [$self _makeUrl trace/done [dict create token $token]]
    }
    ##
    # traceFetch
    #    Fetch traces that fired since the last fetch or since we established
    #    our interest in traces.
    #
    # @param token - the client token returned from traceEstablish
    # @return dict with the keys parameter, spectrum, gate.  Each of these
    #         is a list of strings.  The strings are the parameters of the
    #         trace.
    #
    method traceFetch {token} {
        return [dict get [$self _request [$self _makeUrl             \
            trace/fetch [dict create token $token ]]] \
            detail                                                   \
        ]
    }
    #--------------------------------------------------------------------------
    # mirror
    
    method mirror {{pattern *}} {
        return [dict get [$self _request [$self _makeUrl                  \
            mirror [dict create pattern $pattern]                         \
        ]] detail]                                                        \
    }
}
