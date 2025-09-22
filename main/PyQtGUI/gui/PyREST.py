#!/usr/bin/env python
import io
import sys, os, json, httplib2, urllib
sys.path.append(os.getcwd())

# Python class to interface SpecTcl REST plugin

class PyREST:
    def __init__(self, loggerMain, server, rest):
        self.server = server
        self.rest = rest
        self.logger = loggerMain
    

    #### Bashir added so the REST client can switch to whatever they type in the Connect window #################
    def reconfigure(self, server, rest):
        self.server = str(server).strip()
        self.rest = str(rest).strip()
        # self.logger.info("PyREST reconfigured to http://%s:%s", self.server, self.rest)
    ##################################################################################

    ########################################
    ## Parameter requests
    ########################################    
    
    # get list of parameters in a dictionary form. List the SpecTcl parameters and their properties. If the parameter
    # is also defined as a tree parameter, the tree parameter properties are supplied. It is of the form
    # Each of the objects has the following fields:
    #  name
    #  id
    #  bins
    #  low
    #  high
    #  units
    def listParameter(self, pattern="*"):
        url = "http://"+self.server+":"+self.rest+"/spectcl/parameter/list?filter="+str(pattern)
        response = self.sendRequest(url)
        if response is None :
            return None 
        param_dict = json.loads(response.decode())
        return param_dict["detail"]   


    # edit parameter. Modifies the properties of a parameter. It takes as arguments:
    # name (mandatory - must be the name of a currently defined tree parameter)
    # bins
    # low
    # high
    # units
    # To be used in Python as self.editParameter("h", bins=10, low=10, high=20)
    def editParameter(self, name, **kwargs):
        url = "http://"+self.server+":"+self.rest+"/spectcl/parameter/edit?name="+str(name)+"&"
        for key, value in kwargs.items():
            url += key+"="+value+"&"
        self.sendRequest(url)
        

    # promote parameter. Promotes a simple parameter to a tree parameter. The difference between a parameter
    # and a tree parameter from the point of view of the rest interface is that tree parameters have additional
    # properties that can assist in choosing good axis limits and binning when create spectra
    # It takes as argument:
    # name (mandatory)
    # bins (mandatory)
    # low (mandatory)
    # high (mandatory)
    # units (optional)
    def promoteParameter(self, name, bins, low, high, units=""):
        url = "http://"+self.server+":"+self.rest+"/spectcl/parameter/promote?name="+str(name)+"&bins="+str(bins)+"&low="+str(low)+"&high="+str(high)+"&units="+str(units)
        self.sendRequest(url)


    # create parameters. Provides direct access to the SpecTcl treeparameter -create command. This command can create
    # a new parameter if there is no existing parameter. It accepts the arguments:
    # name (mandatory)
    # bins (mandatory)
    # low (mandatory)
    # high (mandatory)
    # units (optional)
    def createParameter(self, name, bins, low, high, units=""):
        url = "http://"+self.server+":"+self.rest+"/spectcl/parameter/create?name="+str(name)+"&bins="+str(bins)+"&low="+str(low)+"&high="+str(high)+"&units="+str(units)
        self.sendRequest(url)    
        

    # lists only the tree parameters that have been created by treeparameter -create command
    def listnewParameter(self):
        url = "http://"+self.server+":"+self.rest+"/spectcl/parameter/listnew"
        response = self.sendRequest(url)
        if response is None :
            return None 
        param_dict = json.loads(response.decode())
        return param_dict["detail"] 
       

    # returns the state of a tree parameter check flag. The required query parameter name is the name of the tree
    # parameter to operate on. The check flag indicates if a tree parameter has been created or modified since SpecTcl
    # started or since the flag was cleared.
    def checkParameter(self, name):
        url = "http://"+self.server+":"+self.rest+"/spectcl/parameter/check?name="+str(name) 
        response = self.sendRequest(url)
        if response is None :
            return None 
        param_dict = json.loads(response.decode())
        return param_dict["detail"] 


    # clears the check flag 
    def uncheckParameter(self, name):
        url = "http://"+self.server+":"+self.rest+"/spectcl/parameter/uncheck?name="+str(name)
        response = self.sendRequest(url)
        if response is None :
            return None 
        param_dict = json.loads(response.decode())
        return param_dict["detail"] 


    # Tree parameter implementation version. Returns a string
    def versionParameter(self, name):
        url = "http://"+self.server+":"+self.rest+"/spectcl/parameter/version?name="+str(name)  
        response = self.sendRequest(url)
        if response is None :
            return None 
        param_dict = json.loads(response.decode())
        return param_dict["detail"]   


    ########################################
    ## Spectrum requests
    ########################################    

    # get list of spectra in a dictionary form. Produce information about the spectra whose names match a pattern
    # with glob wildcards characters. Each of the objects has the following fields:
    #  name - spectrum name
    #  type - spectrum type code
    #  params - parameter definitions provided as an array of strings
    #  axes - array of objects that describe the SpecTcl axes, each object has the attributes low, high, bins
    #  chantype - channel type code (i.e. long)
    #  gate - gate applied to the spectrum
    def listSpectrum(self, pattern="*"):
        url = "http://"+self.server+":"+self.rest+"/spectcl/spectrum/list?filter="+str(pattern)
        response = self.sendRequest(url)
        if response is None :
            return None 
        spectrum_dict = json.loads(response.decode())
        return spectrum_dict["detail"]   


    # delete spectrum. the name parameter provides the name of the spectrum to delete
    def deleteSpectrum(self, name):
        url = "http://"+self.server+":"+self.rest+"/spectcl/spectrum/delete?name="+str(name)
        self.sendRequest(url)


    # create new spectrum. params and axes (list of lists) are lists. It is defined by the parameters:
    #  name (mandatory)
    #  type (mandatory) spectrum type code i.e. 1 for 1-d spectrum, 2 for 2-d spectrum
    #  parameters (mandatory) parameter expressed as a space separated list
    #  axes (mandatory) space separated list of SpecTcl axis i.e. {0 1023 1024} {0 511 512}
    #  chantype - channel type code. defaults to long
    def createSpectrum(self, name, types, params, axes):
        url = "http://"+self.server+":"+self.rest+"/spectcl/spectrum/create?"+str(name)+"&type="+str(types)+"&parameters="
        if int(types) == 2:
            url += "{"+str(params[0])+"} {"+str(params[1])+"}&axes={"+(axes[0])[0]+" "+(axes[0])[1]+" "+(axes[0])[2]+"} {"+(axes[1])[0]+" "+(axes[1])[1]+" "+(axes[1])[2]+"}"
        else:
            url += "{"+params[0]+"}&axes={"+(axes[0])[0]+" "+(axes[0])[1]+" "+(axes[0])[2]+"}"
        self.sendRequest(url)


    # clear spectrum. Clears the counts in a set of spectra
    def clearSpectrum(self, pattern=""):
        url = "http://"+self.server+":"+self.rest+"/spectcl/spectrum/clear?filter="+str(pattern)
        self.sendRequest(url)


    # contents spectrum. Returns the content of a spectrum. This is sufficient to reconstruct the spectrum channels and overflow statistics.
    # The only parameter is the mandatory name parameter which is the name of the spectrum to return. There are differences depending on the
    # spectrum type and the version of the SpecTcl REST server. The output dictionary contains:
    # statistics:
    #     xunderflow - the number of times and X parameter value was below the X axis low limit
    #     yunderflow - the number of times and Y parameter value was below the Y axis low limit
    #     xoverflow - the number of times and X parameter value was above the X axis high limit
    #     yoverflow - the number of times and X parameter value was above the Y axis high limit
    # channels:
    #     x - the X channel number
    #     y - (2-d spectra only) the Y channel number
    #     v - the number of counts for that channel
    def contentSpectrum(self, name):
        url = "http://"+self.server+":"+self.rest+"/spectcl/spectrum/contents?name="+str(name)
        response = self.sendRequest(url)
        if response is None :
            return None 
        spectrum_dict = json.loads(response.decode())
        return spectrum_dict["detail"]   


    ########################################
    ## Gate requests
    ########################################

    # get list of gates in a dictionary form. List the definitions of gates whose names match a pattern with glob wildcards.
    # The detail field is a dictionary of objects. Each object describes a gate and the actual shape can vary depending on the gate type.
    # Each of the objects has the following fields:
    #  name
    #  type
    #  parameters
    #       0
    #       low
    #       high
    #  or
    #  parameters
    #       0
    #       1
    #  points
    #       0
    #          x
    #          y
    #       1
    #          ...
    def listGate(self, pattern="*"):
        url = "http://"+self.server+":"+self.rest+"/spectcl/gate/list?filter="+str(pattern)
        response = self.sendRequest(url)
        if response is None :
            return None 
        gate_dict = json.loads(response.decode())
        return gate_dict["detail"]     


    # delete gate. the name parameter provides the name of the gate to delete
    def deleteGate(self, name):
        url = "http://"+self.server+":"+self.rest+"/spectcl/gate/delete?name="+str(name)
        self.sendRequest(url)


    # edit/create gate. It creates or redefines an existing gate. The name parameter specifies the name of the gate.
    # parameters can be a value of a list, boundaries can be a list or a list of lists.
    # If the gate doesn't exist it will be created. Gate can not only be edited to change their points, any aspect of a gate
    # can be modified (except for the name)
    # A band must have a minimum of two points, while a contour requires at least three points
    def createGate(self, name, types, parameters, boundaries, maskval="*"):
        url = "http://"+self.server+":"+self.rest+"/spectcl/gate/edit?name="+str(name)+"&type="+str(types)
        if str(types) == "s": # slice
            url += "&parameter="+str(parameters[0])+"&low="+str(boundaries[0])+"&high="+str(boundaries[1])
        elif str(types) == "gs": # gamma slice
            for i in parameters:
                url += "&parameter="+str(i)
            url += "&low="+str(boundaries[0])+"&high="+str(boundaries[1])
        elif (str(types) == "c" or str(types) == "b"):  # contour or band
            url += "&xparameter="+str(parameters[0])+"&yparameter="+str(parameters[1])
            for point in boundaries:
                url += "&xcoord="+str(point["x"])+"&ycoord="+str(point["y"])
        elif (str(types) == "gc" or str(types) == "gb"):  # gamma contour or band
            for i in parameters:
                url +="&parameter="+str(i)
            for point in boundaries:
                url += "&xcoord="+str(point["x"])+"&ycoord="+str(point["y"])
        elif (str(types) == "em" or str(types) == "am" or str(types) == "nm"):  # bit mask
            url += "&parameter="+str(parameters)+"&value="+str(maskval)
        #want + gate here for c and b gates on m2 spectrum (for cutiepie)
        elif str(types) == "+":
            url = "http://"+self.server+":"+self.rest+"/spectcl/gate/edit?name="+str(name)+"&type=%2B"
            for i in parameters:
                url +="&gate="+str(i)
        self.sendRequest(url)


    # Creates a simple 1d gate. This must be of type s or gs. It can have one or more parameters. The query parameters are:
    # name - gate name; if the gate already exists this gate definition will replace it
    # gatetype - type of gate; it must be s or gs or an error will be raised
    # parameters - list of gate parameters (only one if type is s)
    # low, high - low and high limit of the gate
    def create1DGate(self, name, types, parameters, low, high):
        url = "http://"+self.server+":"+self.rest+"/spectcl/gate/edit?name="+str(name)+"&type="+str(types)
        if str(types) == "s": # slice
            url += "&parameter="+str(parameters)+"&low="+str(boundaries[0])+"&high="+str(boundaries[1])
        elif str(types) == "gs": # gamma slice
            for i in parameters:
                url += "&parameter="+str(i)
            url += "&low="+str(boundaries[0])+"&high="+str(boundaries[1])
        else:
            raise Exception("Only s and gs types are allowed")
            
        self.sendRequest(url)


    # Creates a simple 2d gate. This must be of type c/b or gc/gb. The query parameters are:
    # name - gate name; if the gate already exists this gate definition will replace it
    # gatetype - type of gate; it must be s or gs or an error will be raised
    # parameters - list of gate parameters 
    # boundaries - gate points in coordinates (x,y)
    def create2DGate(self, name, types, parameters, boundaries):
        url = "http://"+self.server+":"+self.rest+"/spectcl/gate/edit?name="+str(name)+"&type="+str(types)
        if (str(types) == "c" or str(types) == "b"):  # contour or band
            url += "&xparameter="+str(parameters[0])+"&yparameter="+str(parameters[1])
            for point in boundaries:
                url += "&xcoord="+str(point[0])+"&ycoord="+str(point[1])
        elif (str(types) == "gc" or str(types) == "gb"):  # gamma contour or band
            for i in parameters:
                url +="&parameter="+str(i)
            for point in boundaries:
                url += "&xcoord="+str(point[0])+"&ycoord="+str(point[1])                
        else:
            raise Exception("Only c/b and gc/gb types are allowed")
        self.sendRequest(url)
        

    # Create mask gate. The query parameters are:
    # name - gate name; if the gate already exists this gate definition will replace it
    # gatetype - type of gate; it must be s or gs or an error will be raised
    # parameters - list of gate parameters 
    # mask - mask value
    def createMaskGate(self, name, types, parameters, mask):
        if (str(types) == "em" or str(types) == "am" or str(types) == "nm"):  # bit mask
            url += "&parameter="+str(parameters)+"&value="+str(maskval)
        else:
            raise Exception("Only em, am, nm types are allowed")
        self.sendRequest(url)


    # list the gates applied to a spectrum. The result is an object with attributes status and detail.
    # Each element describes a single gate application for a matching spectrum. The attributes of these objects are:
    # spectrum - name of the spectrum
    # gate - name of the gate applied. If the value is -TRUE-, the spectrum has not gate applied
    def applylistgate(self, pattern="*"):
        url = "http://"+self.server+":"+self.rest+"/spectcl/apply/list?pattern="+str(pattern)
        response = self.sendRequest(url)
        if response is None :
            return None 
        gate_dict = json.loads(response.decode())
        return gate_dict["detail"]

        
    # gate application. Applies the gate to a spectrum. It takes as argument the gate name and the spectrum name
    def applyGate(self, gate, spectrum):
        url = "http://"+self.server+":"+self.rest+"/spectcl/apply/apply?gate="+str(gate)+"&spectrum="+str(spectrum)
        self.sendRequest(url)
        

    ############################################################
    # Attaching data sources
    ############################################################    

    # The SpecTcl attach command can be accessed using the REST plugin. The query parameters are:
    # type - options are file (to read data from a file) or pipe (to read data from a program on the other end of a pipe)
    # source - the source string expected to attach the source type. For "file" data source, this is the FULL path to the file to read
    #          for "pipe" full command string
    # size (optional) - sets the blocking factor for reads from the data source (default 8192)
    # format (optional) - sets the data format. Acceptable values are ring (default), nscl (NSCLDAQ before v10),
    #                     jumbo (fixed length buffers longer than 128K bytes from NSCLDAQ before v10), filter (XDR filter data)
    def attachSource(self, types, source, size="8192", formats="ring"):
        url = "http://"+self.server+":"+self.rest+"/spectcl/attach/attach?type="+str(types)+"&source="+str(source)+"&size="+str(size)+"&format="+str(formats)
        self.sendRequest(url)


    # starts the data analysis
    def startSource(self):
        url = "http://"+self.server+":"+self.rest+"/spectcl/attach/start"
        self.sendRequest(url)


    # information on attached sources
    def listSource(url):
        url = "http://"+self.server+":"+self.rest+"/spectcl/attach/list"
        self.sendRequest(url)        
        

    ############################################################
    # Binding spectra to display memory
    ############################################################    

    # bind all spectra to display memory
    def sbindall(self):
        url = "http://"+self.server+":"+self.rest+"/spectcl/sbind/all"
        self.sendRequest(url)


    # bind all spectra named in all instances of the spectrum query parameter to display memory.
    # spectra is a list
    def sbindSpectrum(self, spectra):
        url = "http://"+self.server+":"+self.rest+"/spectcl/sbind/sbind?"
        for spectrum in spectra:
            url += "spectrum="+str(spectrum)+"&"
        self.sendRequest(url)    


    # return information about the bindings of spectra that match the pattern query parameter interpreted as a glob spectrum name match string.
    # Each of the objects has the following fields:
    #  spectrumid
    #  name
    #  binding
    def listsbind(self, pattern=""):
        url = "http://"+self.server+":"+self.rest+"/spectcl/sbind/list?filter="+str(pattern)
        response = self.sendRequest(url)
        if response is None :
            return None 
        sbind_dict = json.loads(response.decode())
        return sbind_dict["detail"]


    ############################################################
    # Fit command
    ############################################################

    # Provides access to the SpecTcl fit command. fit provides you with the ability to fit spectra to arbitrary functions.
    # The built-in linear and gaussian fit types are provided and the programming manual describes how to extend the set of fits that
    # are provided by this command

    # creates and fits a function to a spectrum. The query parameters are:
    # name - name associated to the fit
    # spectrum - spectrum on which the fit is computed
    # low, high - the channel coordinates over which the fit is to be computed
    # type - type of fit to perform. The built in fit types are linear and gaussian
    def createFit(self, fitname, spectrum, low, high, fittype):
        url = "http://"+self.server+":"+self.rest+"/spectcl/fit/create?name="+str(fitname)+"&spectrum="+str(spectrum)+"&low="+str(low)+"&high="+str(high)+"&type="+str(fittype)
        self.sendRequest(url)                          


    # As spectra accumulate, fit data will be outdated. This allows the fit information to be recomputed to match current data.
    # It updates the set of fits whose names match the glob pattern in the query parameter pattern.
    def updateFit(self, pattern="*"):
        url = "http://"+self.server+":"+self.rest+"/spectcl/fit/update?filter="+str(pattern)
        response = self.sendRequest(url)
        if response is None :
            return None 
        fit_dict = json.loads(response.decode())
        return fit_dict["detail"]


    # deletes a fit
    def deleteFit(self, name):
        url = "http://"+self.server+":"+self.rest+"/spectcl/fit/delete?name="+str(name)
        self.sendRequest(url)


    # list of all created fits. The result is a detail attribute that is an array. Each element of the array is an object that describes
    # a matching fit. The attributes of these objects are:
    # name - name of the fit
    # spectrum - name of the spectrum the fit is computed on
    # type - type of the fit being computed
    # low, high - fit limits in spectrum channel coordinates
    # parameters - an object that contains the fit parameters. Likely there will be a chi square attribute that describes the goodness of the fit
    def listFit(self, pattern=""):
        url = "http://"+self.server+":"+self.rest+"/spectcl/fit/list?filter="+str(pattern)  
        response = self.sendRequest(url)
        if response is None :
            return None 
        fit_dict = json.loads(response.decode())
        return fit_dict["detail"]              

                  
    ############################################################
    # Fold command
    ############################################################    

    # Fold command has three operations 1) fold a list of gamma spectra on a gamma gate 2) remove a fold from a gamma spectrum
    # 3) list the folded spectra

    # List folded spectra that match the pattern query parameter (treated as a glob pattern)
    # Each object has the following attributes:
    # spectrum - name of a folded spectrum
    # gate - name of the gate used to fold the spectrum 
    def listFold(self, pattern="*"):
        url = "http://"+self.server+":"+self.rest+"/spectcl/fold/list?filter="+str(pattern)
        response = self.sendRequest(url)
        if response is None :
            return None 
        fold_dict = json.loads(response.decode())
        return fold_dict["detail"] 


    # apply gamma gate to a list of spectra
    def applyFold(self, gate, spectra):
        url = "http://"+self.server+":"+self.rest+"/spectcl/fold/apply?gate="+str(gate)
        for spectrum in spectra:
            url += "&spectrum="+str(spectrum)+"&"
        self.sendRequest(url)


    # unfolds a spectrum
    def removeFold(self, spectrum):
        url = "http://"+self.server+":"+self.rest+"/spectcl/fold/remove?spectrum="+str(spectrum)
        self.sendRequest(url)


    ############################################################
    # Access channel command
    ############################################################        

    # Spectrum channel values can be inspected. To inspect, the arguments are:
    # spectrum (mandatory) - spectrum name
    # xchannel (mandatory) - X channel coordinate (NOT real coordinate)
    # ychannel (mandatory if 2-d) - Y channel coordinate (NOT real coordinate)    
    # To be used in Python as self.getChannelContent("h", xchannel=10, ychannel=10)
    def getChannelContent(self, name, **kwargs):
        url = "http://"+self.server+":"+self.rest+"/spectcl/channel/get?spectrum="+str(name)+"&"
        for key, value in kwargs.items():
            url += key+"="+str(value)+"&"
        response = self.sendRequest(url)
        if response is None :
            return None 
        get_dict = json.loads(response.decode())
        return get_dict["detail"] 


    # Spectrum channel values to be set. The arguments are:
    # spectrum (mandatory) - spectrum name
    # xchannel (mandatory) - X channel coordinate (NOT real coordinate)
    # value (mandatory) - value to add in the X channel coordinate
    # ychannel (mandatory if 2-d) - Y channel coordinate (NOT real coordinate)    
    # value (mandatory if 2-d) - value to add in the Y channel coordinate
    # To be used in Python as self.setChannelContent("h", xchannel=10, xvalue=100, ychannel=10, yvalue=100)
    def setChannelContent(self, name, **kwargs):
        url = "http://"+self.server+":"+self.rest+"/spectcl/channel/set?spectrum="+str(name)+"&"
        for key, value in kwargs.items():
            if (key == "xvalue" or key == "yvalue"):
                key = "value"
            url += key+"="+value+"&"            
        self.sendRequest(url)


    ############################################################
    # Clear spectra
    ############################################################        
    
    def spectrumClear(self, pattern="*"):
        url = "http://"+self.server+":"+self.rest+"/spectcl/spectrum/zero?filter="+str(pattern)
        self.sendRequest(url)
        

    def spectrumAllClear(self):
        self.spectrumClear("*")
        

    ############################################################
    # Projecting spectra
    ############################################################        

    # Project an existing spectrum onto one of the axes creating a new spectrum. The query parameters are:
    # snapshot - boolean value. If nonzero, the projected spectrum will not increment: it's a snapshot of the projection in time.
    # source - specifies the name of the spectrum to project
    # newname - specifies the name of the new projected spectrum
    # direction - either x or y, identifying the axis onto which the spectrum is projected
    # contour (optional) - if specified, this is a contour that must have been displayable on the source spectrum. The projected spectrum
    #                      The projected spectrum is initially populated only with counts that are within that contour. Furthermore, if the
    #                      projected spectrum is not a snapshot spectrum, it is gated on that contour so that the projection remains faithful
    #                      as new data arrive.
    def createProjection(self, snapshot, source, newname, direction, contour=""):
        url = "http://"+self.server+":"+self.rest+"/spectcl/project?snapshot="+str(snapshot)+"&source="+str(source)+"&newname="+str(newname)+"&direction="+str(direction)+"&contour="+str(contour)
        self.sendRequest(url)

    ############################################################
    # Spectrum underflow and overflow statistics
    ############################################################        

    # Returns the underflow and overflow statistics for the spectra whose names match the optional pattern query.
    # Each object has the attributes:
    # name
    # underflows
    # overflows
    def getSpectrumStats(self, name, pattern="*"):
        url = "http://"+self.server+":"+self.rest+"/spectcl/specstats?name="+str(name)+"&filter="+str(pattern)
        response = self.sendRequest(url)
        if response is None :
            return None 
        stats_dict = json.loads(response.decode())
        return stats_dict["detail"] 


    ############################################################
    # Accessing tree variable command
    ############################################################

    # variable list. The detail attribute is an array of objects. Each object describes a tree variable and has the
    # following attributes:
    # name - name of the variable
    # value - current value of the treevariable
    # units - units of measure of the tree variable
    def listVariable(self):
        url = "http://"+self.server+":"+self.rest+"/spectcl/treevariable/list"
        response = self.sendRequest(url)
        if response is None :
            return None 
        var_dict = json.loads(response.decode())
        return var_dict["detail"] 


    # Tree variable values can be changed. Note that the units are NOT optional
    def setVariable(self, name, value, units):
        url = "http://"+self.server+":"+self.rest+"/spectcl/treevariable/set?name="+str(name)+"&value="+str(value)+"&units="+str(units)
        self.sendRequest(url)


    # tree variables have a flag that indicates if they have been modified in the life of the SpecTcl run. This flag is normally
    # used to limit the amount of information that must be saved in files that capture the SpecTcl analysis state. The detail attribute of
    # the return object is 0 if the variable has not been changed and 1 if it has
    def checkVariable(self):
        url = "http://"+self.server+":"+self.rest+"/spectcl/treevariable/list"
        response = self.sendRequest(url)
        if response is None :
            return None 
        var_dict = json.loads(response.decode())
        return var_dict["detail"] 


    # it is possible to change the flag above 
    def setFlagVariable(self, name):
        url = "http://"+self.server+":"+self.rest+"/spectcl/treevariable/setchanged?name="+str(name)
        self.sendRequest(url)    


    # There are cases where it's important to fire Tcl traces associated with tree variables. If not supplied
    # any pattern it defaults to all variable names
    def traceVariable(self, pattern="*"):
        url = "http://"+self.server+":"+self.rest+"/spectcl/treevariable/firetraces?filter="+str(pattern)
        self.sendRequest(url)
        

    ############################################################
    # Accessing filter command
    ############################################################            

    # Creation of filters. Parameter has to be a list in input. The query parameters are:
    # name - name of the filter to create
    # gate - specifies the name of a gate that will determine which events are output by the filter
    # parameter - each occurrance of this query parameter specifies the name of a parameter that will be written to
    #             the file output
    def createFilter(self, name, gate, parameters):
        url = "http://"+self.server+":"+self.rest+"/spectcl/filter/new?name="+str(name)+"&gate="+str(gate)
        for i in parameters:
            url += "&parameter="+str(i)+"&"
        self.sendRequest(url)


    # Delete a filter. The only parameter is the name of the filter we want to remove
    def deleteFilter(self, name):
         url = "http://"+self.server+":"+self.rest+"/spectcl/filter/delete?name="+str(name)
         self.sendRequest(url)


    # Enable/disable filter
    def enableFilter(self, name):
         url = "http://"+self.server+":"+self.rest+"/spectcl/filter/enable?name="+str(name)
         self.sendRequest(url)


    def disableFilter(self, name):
         url = "http://"+self.server+":"+self.rest+"/spectcl/filter/disable?name="+str(name)
         self.sendRequest(url)            


    # List of filter. Each object will describe a single filter and contains the following attributes:
    # name - name of the filter
    # gate - name of the gate applied to the filter
    # file - name of the output file to which the filter writes
    # parameters - an array of parameters names written to the output file for each event that passes the gate
    # enabled - if the filter is enabled, this attribute has value "enabled" otherwise "disabled"
    # format - contains the format string i.e. xdr
    def listFilter(self, pattern="*"):
        url = "http://"+self.server+":"+self.rest+"/spectcl/filter/list?filter="+str(pattern)
        response = self.sendRequest(url)
        if response is None :
            return None 
        filt_dict = json.loads(response.decode())
        return filt_dict["detail"]   


    ############################################################
    # Integrate command
    ############################################################

    # There are three forms of the integrate command
    # 1) Integrate the interior of a gate on any spectrum
    # 2) Integrate a 1-d spectrum within some explicit area of interest provided in spectrum world coordinates
    # 3) Integrate a spectrum within an arbitrary polygonal region of interest. The interior of the polygon is defined
    #    using the same odd crossing algorithm to determine the interior of the SpecTcl contours
    # The output is an object with the following fields:
    # centroid - the world coordinates of the integration centroid. For a 1-d spectrum this is a single value; for a 2-d spectrum
    #            this is a 2 element array containing the x and y coordinates of the centroid
    # counts - contains the number of counts inside the region of interest
    # fwhm - full width at half maximum of the peak (assuming to be gaussian). For a 1-d spectrum this is a single value;
    #        for a 2-d spectrum this is a 2 element array containing the x and y coordinates of fwhm

    def integrateGate(self, name, gate):
        url = "http://"+self.server+":"+self.rest+"/spectcl/integrate?spectrum="+str(name)+"&gate="+str(gate)
        response = self.sendRequest(url)
        if response is None :
            return None 
        int_dict = json.loads(response.decode())
        return int_dict["detail"]  
     

    def integrate1D(self, name, low, high):
        url = "http://"+self.server+":"+self.rest+"/spectcl/integrate?spectrum="+str(name)+"&low="+str(low)+"&high="+str(high)
        response = self.sendRequest(url)
        if response is None :
            return None 
        int_dict = json.loads(response.decode())
        return int_dict["detail"] 


    # points are a list of lists
    def integrate2D(self, name, points):
        url = "http://"+self.server+":"+self.rest+"/spectcl/integrate?spectrum="+str(name)
        for point in points:
            url += "&xcoord="+str(point[0])+"&ycoord="+str(point[1])
        response = self.sendRequest(url)
        if response is None :
            return None 
        int_dict = json.loads(response.decode())
        return int_dict["detail"] 
        
    
    ############################################################
    # Create parameter command
    ############################################################

    # Provides support for creating a new parameter definition. A parameter definition makes a correspondence between a name and a slot in the CEvent pseudo array
    # that SpecTcl event processing pipeline fills in. There are both mandatory and optional parameters. The query parameters are:
    # name - name of the parameters (must be unique)
    # number - parameter id (must be unique). It specifies the slot in the CEvent object in which that parameter will be placed by the event processing pipeline
    # units - (optional) units of measure of the parameter
    # resolution - (optional) provides the number of bits of resolution the parameter has
    # low - (optional but...) - assumed low limit of the parameter values. This requires resolution, high, and units to be present
    # high - (optional but...) - assumed low limit of the parameter values. This requires resolution, low, and units to be present    
    # To be used in Python as self.createRawParameter("newpar", 1111, resolution=12, low=0, high=100, units="")
    def createRawParameter(self, name, number, **kwargs):
        url = "http://"+self.server+":"+self.rest+"/spectcl/rawparameter/new?name="+str(name)+"&number="+str(number)
        for key, value in kwargs.items():
            url += key+"="+value+"&"
        self.sendRequest(url)                        


    # delete a parameter by name or id
    def deleteRawParameter(self, par):
        url = "http://"+self.server+":"+self.rest+"/spectcl/rawparameter/delete?"
        if isinstance(type(par),str):
            url += "name="+str(par)
        else:
            url += "id="+str(par)
        self.sendRequest(url)


    # list parameter by name or id. The detail attribute of the returned JSON contains an array of objects. Each objects describes one parameter.
    # Some attributes of these objects are present in all elements but the presence or absence of others depends on how the parameter was defined.
    # The attributes are:
    # name - name of the parameter
    # id - id (CEvent slot) parameter
    # resolution - (optional)
    # low - (optional)
    # high - (optional)
    # units - (optional)
    def listRawParameter(self, par="*"):
        url = "http://"+self.server+":"+self.rest+"/spectcl/rawparameter/list?"
        if isinstance(type(par),str):
            url += "pattern="+str(par)
        else:
            url += "id="+str(par)
        response = self.sendRequest(url)
        if response is None :
            return None 
        int_dict = json.loads(response.decode())
        return int_dict["detail"]    


    ############################################################
    # Pseudo command
    ############################################################

    # It provides the ability to create a new parameter frome existing parameters. The new parameter is computed using a script that is passed to
    # the pseudo command when the parameter is generated (see docs for pseudo in SpecTcl command reference)

    # create pseudo command. The parameters should exist before creating the pseudoparameter. The body query parameter defines the Tcl script that will computer
    # the pseudo parameter that are passed to this script.
    def createPseudo(self, name, body, parameters):
        url = "http://"+self.server+":"+self.rest+"/spectcl/pseudo/create?name="+str(name)+"&body="+str(body)
        for i in parameters:
            url += "&parameter="+str(i)+"&"        
        self.sendRequest(url)


    # list pseudo parameters. The return value detail is array where each object describes one pseudo parameter. Each object has some or all the following
    # attributes:
    # name - name of the pseudo parameter
    # parameters - array of strings that are the names of the parameters the pseudo depends on
    # computation - the script that defines the computation that takes the parameters and from it creates the pseudo parameter.
    def listPseudo(self, par="*"):
        url = "http://"+self.server+":"+self.rest+"/spectcl/pseudo/list?pattern="+str(par)  
        response = self.sendRequest(url)
        if response is None :
            return None 
        ps_dict = json.loads(response.decode())
        return ps_dict["detail"] 


    ############################################################
    # sread command
    ############################################################

    # sread supports reading spectra from file. With the exception of filename all query parameters are optional:
    # filename - name of the file to read
    # format - format of the file (defaults to ascii)
    # snapshot - if true, the resulting spectrum will be a snapshot spectrum (default 1)
    # replace - if true, the spectrum read in will replace any spectrum that already exists with the name of the spectrum in the file (default 0)
    # bind - binds the spectrum into the displayer shared memory region (default 1)
    # To be used in Python as self.sread("file.name", format="", snapshot=1, replace=0, bind=1)
    def sread(self, name, **kwargs):
        url = "http://"+self.server+":"+self.rest+"/spectcl/sread?filename="+str(name)+"&"
        for key, value in kwargs.items():
            url += key+"="+value+"&"
        self.sendRequest(url)


    ############################################################
    # ringformat command
    ############################################################

    # set the major and minor version of the ringformat command
    def ringFormat(self, major, minor="0"):
        url = "http://"+self.server+":"+self.rest+"/spectcl/ringformat?major="+str(major)+"&minor="+str(minor)
        self.sendRequest(url)


    ############################################################
    # unbind command
    ############################################################

    # it allows to access the capabilities of the SpecTcl unbind command by name or id
    def unbindByName(self, names):
        url = "http://"+self.server+":"+self.rest+"/spectcl/unbind/byname?"
        for name in names:
            url += "name="+str(name)+"&"
        self.sendRequest(url)


    def unbindById(self, sids):
        url = "http://"+self.server+":"+self.rest+"/spectcl/unbind/byid?"
        for sid  in sids:
            url += "name="+str(name)+"&"
        self.sendRequest(url)


    def unbindAll(self):
        url = "http://"+self.server+":"+self.rest+"/spectcl/unbind/all"
        self.sendRequest(url)        


    ############################################################
    # ungate command
    ############################################################

    # this command removes any gate condition from a spectrum. Names is a list.
    def ungateSpectum(self, names):
        url = "http://"+self.server+":"+self.rest+"/spectcl/ungate?"
        for name in names:
            url += "name="+str(name)+"&"
        self.sendRequest(url)    


    ############################################################
    # swrite command
    ############################################################

    # swrite supports writing spectrum files on request from clients. The query parameters are:
    # filename - (mandatory) name of the file to write
    # spectrum - (mandatory, can be multiple) each occurrence of this query parameter names a spectrum to be written to a file. At least one must be specified
    # format - (optiona) format of the file (defaults to ascii)
    def sread(self, name, spectra, formats="ascii"):
        url = "http://"+self.server+":"+self.rest+"/spectcl/swrite?file="+str(name)+"&"
        for spectrum in spectra:
            url += "spectrum="+spectrum+"&"
        url += "format="+str(formats)
        self.sendRequest(url)


    ############################################################
    # start/stop analysis command
    ############################################################        

    # data analysis from the source can be started or stopped using the following queries:
    def startAnalysis(self):
        url = "http://"+self.server+":"+self.rest+"/analyze/start"
        self.sendRequest(url)


    def stopAnalysis(self):
        url = "http://"+self.server+":"+self.rest+"/analyze/stop"
        self.sendRequest(url)        


    ############################################################
    # root tree command
    ############################################################                

    # note that roottree is present only in a SpecTcl instance that has loaded the rootinterface package.
    # this interface allows you to create ROOT output trees. Note that writing ROOT trees impacts the performance of SpecTcl owing to the poor
    # performance of ROOT file output

    # create ROOT tree. The query parameters are:
    # tree - name of the tree
    # parameter - (multiple) each of these parameters specifies a patternthat can have glob wildcard characters. The parameters that will be booked in the tree
    #             is a set of parameters whose names match any of the patterns.
    # gate - (optional) if provided, the value of this parameter is the name of an existing gate that will determine which events are placed in the tree. Only those events
    #        that satisfy the gate will booked into the tree
    def createROOTtree(self, name, parameters, gate=""):
        url = "http://"+self.server+":"+self.rest+"/roottree/create?tree="+str(name)
        for i in parameters:
            url += "&parameter="+str(i)+"&"
        url += "gate="+str(gate)
        self.sendRequest(url)


    # delete a previously created tree
    def deleteROOTtree(self, name):
        url = "http://"+self.server+":"+self.rest+"/roottree/delete?tree="+str(name)
        self.sendRequest(url)


    # list all created trees. Provides a return value with the list of the root tree objects with names that match the optional pattern query.
    # The detail attribute will be an array of objects. Each objects describes one root tree and will have the following attributes:
    # tree - name of the tree to create
    # parameters - array of parameter name patterns that define the parameters to book into the tree
    # gate - name of the gate that must be satisfied to add an event to the tree
    def listROOTtree(self, pattern="*"):
        url = "http://"+self.server+":"+self.rest+"/roottree/list?filter="+str(pattern)
        response = self.sendRequest(url)
        if response is None :
            return None 
        root_dict = json.loads(response.decode())
        return root_dict["detail"] 


    ############################################################
    # traces command
    ############################################################                    

    # Traces are a mechanism to allow SpecTcl scripts to be informed of changes to parameters, spectrum, and gate dictionaries. Tracing is problematic for the
    # REST interface. Specifically, REST models is that the client makes a request, and the server fullfils it. Fully implementation of tracing would require the
    # server to initiate an operation in the client, rather than the other way around. Instead, SpecTcl REST establishes its own traces for all things that are traceable
    # in SpecTcl command extensions. Actual traces can be buffered for clients which then can poll for them and use the results of those polls client-side.
    # This requires:
    # 1) A client expresses an interest in traces and the server provides a token that associates that interest with its internal data structures.
    # 2) Periodically, the client can poll the server to get the traces that fired since the last poll.
    # 3) The client expresses that it is no longer interested in traces causing the server to remove any trace buffers and other data associated with that client.
    # When an interest is established, the client must specify a retention period for data in the trace buffer. As new traces are fired, any old data is removed from the trace buffer
    # associated with a client preventing zombie clients from growing without bounds.

    # start the trace service. This function informs the REST server that the client will be interested in trace data. The retention query parameter
    # is the minimum number of seconds the server should retain traced data on behalf of that specific client. On success, the detail attribute of the response
    # is an integer token that should be used in future calls involving the trace subsystem.
    def startTraces(self, seconds):
        url = "http://"+self.server+":"+self.rest+"/spectcl/trace/establish?retention="+str(seconds)
        response = self.sendRequest(url)
        if response is None :
            return None 
        trace_dict = json.loads(response.decode())
        return trace_dict["detail"]


    # stop the trace service. Once the application no longer requires trace information, or as it is cleaning up for exit, a request is made
    def stopTraces(self, token):
        url = "http://"+self.server+":"+self.rest+"/spectcl/trace/done?token="+str(token)
        self.sendRequest(url)


    # poll traces. The detail attribute of the returned JSON will be an object. Each attribute represents a trace type and each of those attributes will, in turn,
    # have a value that is an array:
    # parameter - the value of this attribute is an array. Each element of the array describes a parameter trace. Each element is the tcl list that contains the operation
    #             that fired the trace (add or delete) followed by the trace parameters that SpecTcl provides parameter -trace handlers
    # gate - the value of this attribute is an array. Each array element describes a gate trace. The array elements are a tcl list where the first element of the list is the
    #        gate trace operation add, delete, or change that describes the operation that fired the trace and the remainder are the parameters that are passed to the local trace handler
    # spectrum - the value of this attribute is an array. Each array element describes a spectrum trace. The elements are tcl lists. The first element of that list is the operation,
    #            add or delete, that fired the trace while the remainder are trace parameter(s) passed to the trace handler by SpecTcl
    # binding - the value of this attribute is an array. Each element of the array describes a change in shared memory binding status and is, itself, a three element array.
    #           the elements of the array are, in order, either the text add or remove indicating if the remaining elements describe a spectrum being added or removed from shared memory,
    #           the name of the affected spectrum and the xamine id, or binding slot, that was allocated to the spectrum or from which the spectrum was removed.

    def pollTraces(self, token):
        url = "http://"+self.server+":"+self.rest+"/spectcl/trace/fetch?token="+str(token)
        response = self.sendRequest(url)
        if response is None :
            return None 
        trace_dict = json.loads(response.decode())
        return trace_dict["detail"]

    
    ############################################################
    # general functions for communication and error handling
    ############################################################    
    
    def sendRequest(self, url):
        try:
            ### Bashur added #################
            self.logger.debug("REST GET %s", url)   # <-- add
            ##################################
            status, content = httplib2.Http().request(url, method="GET") # SpecTclREST only takes GET methods.
            #May have other bad keywords
            badKeyWords = ["bad parameter", "Invalid gate"]
            for kw in badKeyWords :
                if kw in str(content) :
                    self.logger.warning('sendRequest -- suspicious REST request status: %s', content)
                    return None
            return content
        except Exception :
            #Cannot pass the exception, full text dont convert to str so log custom error
            self.logger.error('sendRequest -- check Server/User/REST Port/Mirror Port')
            return None
    
    ''' # Bashir commented out
     #check if url is valid
    def checkSpecTclREST(self):
        url = "http://"+self.server+":"+self.rest+"/spectcl"
        try:
            response = httplib2.Http().request(url, method="GET") # SpecTclREST only takes GET methods.
            return True
        except Exception :
            return False
    '''

    def checkSpecTclREST(self):
        url = f"http://{self.server}:{self.rest}/spectcl/spectrum/list?filter=*"
        try:
            status, content = httplib2.Http().request(url, method="GET")
            data = json.loads(content.decode())
            ok = isinstance(data, dict) and data.get("status") == "OK"
            print(f"[PyREST] REST health {'OK' if ok else 'FAIL'} via {url}", flush=True)  # <-- add
            return ok
        except Exception:
            print(f"[PyREST] REST health FAIL via {url} (exception)", flush=True)          # <-- add
            return False


            
