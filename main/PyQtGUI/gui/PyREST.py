#!/usr/bin/env python
import json, httplib2
import threading
import urllib.parse

# Python class to interface SpecTcl REST plugin

_BAD_REST_KEYWORDS = ("bad parameter", "Invalid gate")


class PyREST:
    _HTTP_TIMEOUT = 5  # seconds; prevents GUI/worker thread hang on dead SpecTcl

    def __init__(self, loggerMain, server, rest):
        self.server = server
        self.rest = rest
        self.logger = loggerMain
        self._tls = threading.local()

    @property
    def _http(self):
        """One httplib2.Http per thread.

        httplib2.Http is not thread-safe; this client is used concurrently from
        the GUI thread and RestWorker's polling thread, which could interleave
        requests on a shared socket and corrupt responses."""
        http = getattr(self._tls, 'http', None)
        if http is None:
            http = httplib2.Http(timeout=self._HTTP_TIMEOUT)
            self._tls.http = http
        return http


    #### Bashir added so the REST client can switch to whatever they type in the Connect window #################
    def reconfigure(self, server, rest):
        self.server = str(server).strip()
        self.rest = str(rest).strip()
        # self.logger.info("PyREST reconfigured to http://%s:%s", self.server, self.rest)
    ##################################################################################

    def _build_url(self, endpoint: str, **params) -> str:
        query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        base = f"http://{self.server}:{self.rest}/{endpoint}"
        return f"{base}?{query}" if query else base

    @staticmethod
    def _q(value) -> str:
        """Percent-encode one query value. Spectrum/gate/parameter names may
        contain spaces, '&', '+', '#' — appending them raw corrupts the URL
        (hostile-names class, see thread_workers.lookup_spectrum_info)."""
        return urllib.parse.quote_plus(str(value))

    ########################################
    ## Parameter requests
    ########################################

    # get list of parameters in a dictionary form. List the SpecTcl parameters
    # and their properties.
    #  name
    #  id
    #  bins
    #  low
    #  high
    #  units
    def listParameter(self, pattern="*"):
        url = self._build_url("spectcl/parameter/list", filter=str(pattern))
        response = self.sendRequest(url)
        if response is None :
            return []
        detail = json.loads(response.decode()).get("detail", [])
        return detail if isinstance(detail, list) else []


    # edit parameter. Modifies the properties of a parameter.
    # name (mandatory - must be the name of a currently defined tree parameter)
    # bins
    # low
    # high
    # units
    # To be used in Python as self.editParameter("h", bins=10, low=10, high=20)
    def editParameter(self, name, **kwargs):
        url = self._build_url("spectcl/parameter/edit", name=str(name), **kwargs)
        self.sendRequest(url)


    # promote parameter. Promotes a simple parameter to a tree parameter.
    # name (mandatory)
    # bins (mandatory)
    # low (mandatory)
    # high (mandatory)
    # units (optional)
    def promoteParameter(self, name, bins, low, high, units=""):
        url = self._build_url("spectcl/parameter/promote", name=str(name), bins=str(bins), low=str(low), high=str(high), units=str(units))
        self.sendRequest(url)


    # create parameters. Provides direct access to the SpecTcl treeparameter
    # -create command.
    # name (mandatory)
    # bins (mandatory)
    # low (mandatory)
    # high (mandatory)
    # units (optional)
    def createParameter(self, name, bins, low, high, units=""):
        url = self._build_url("spectcl/parameter/create", name=str(name), bins=str(bins), low=str(low), high=str(high), units=str(units))
        self.sendRequest(url)


    # lists only the tree parameters that have been created by treeparameter -create command
    def listnewParameter(self):
        url = self._build_url("spectcl/parameter/listnew")
        response = self.sendRequest(url)
        if response is None :
            return []
        detail = json.loads(response.decode()).get("detail", [])
        return detail if isinstance(detail, list) else []


    # returns the state of a tree parameter check flag. The required query
    # parameter name is the name of the tree parameter to operate on.
    def checkParameter(self, name):
        url = self._build_url("spectcl/parameter/check", name=str(name))
        response = self.sendRequest(url)
        if response is None :
            return {}
        param_dict = json.loads(response.decode())
        return param_dict["detail"]


    # clears the check flag
    def uncheckParameter(self, name):
        url = self._build_url("spectcl/parameter/uncheck", name=str(name))
        response = self.sendRequest(url)
        if response is None :
            return {}
        param_dict = json.loads(response.decode())
        return param_dict["detail"]


    # Tree parameter implementation version. Returns a string
    def versionParameter(self, name):
        url = self._build_url("spectcl/parameter/version", name=str(name))
        response = self.sendRequest(url)
        if response is None :
            return {}
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
        url = self._build_url("spectcl/spectrum/list", filter=str(pattern))
        response = self.sendRequest(url)
        if response is None:
            return []
        detail = json.loads(response.decode()).get("detail", [])
        return detail if isinstance(detail, list) else []


    # delete spectrum. the name parameter provides the name of the spectrum to delete
    def deleteSpectrum(self, name):
        url = self._build_url("spectcl/spectrum/delete", name=str(name))
        self.sendRequest(url)


    # create new spectrum. params and axes (list of lists) are lists.
    #  name (mandatory)
    #  type (mandatory) spectrum type code i.e. 1 for 1-d spectrum, 2 for 2-d spectrum
    #  parameters (mandatory) parameter expressed as a space separated list
    #  axes (mandatory) space separated list of SpecTcl axis i.e. {0 1023 1024} {0 511 512}
    #  chantype - channel type code. defaults to long
    def createSpectrum(self, name, types, params, axes):
        url = self._build_url("spectcl/spectrum/create") + "?" + str(name) + "&type=" + str(types) + "&parameters="
        if int(types) == 2:
            url += "{"+str(params[0])+"} {"+str(params[1])+"}&axes={"+(axes[0])[0]+" "+(axes[0])[1]+" "+(axes[0])[2]+"} {"+(axes[1])[0]+" "+(axes[1])[1]+" "+(axes[1])[2]+"}"
        else:
            url += "{"+params[0]+"}&axes={"+(axes[0])[0]+" "+(axes[0])[1]+" "+(axes[0])[2]+"}"
        self.sendRequest(url)


    # clear spectrum. Clears the counts in a set of spectra
    def clearSpectrum(self, pattern=""):
        url = self._build_url("spectcl/spectrum/clear", filter=str(pattern))
        self.sendRequest(url)


    # contents spectrum. Returns the content of a spectrum.
    def contentSpectrum(self, name):
        url = self._build_url("spectcl/spectrum/contents", name=str(name))
        response = self.sendRequest(url)
        if response is None :
            return {}
        spectrum_dict = json.loads(response.decode())
        return spectrum_dict["detail"]


    ########################################
    ## Gate requests
    ########################################

    # get list of gates in a dictionary form. List the definitions of gates
    # whose names match a pattern with glob wildcards.
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
        url = self._build_url("spectcl/gate/list", filter=str(pattern))
        response = self.sendRequest(url)
        if response is None :
            return []
        detail = json.loads(response.decode()).get("detail", [])
        return detail if isinstance(detail, list) else []


    # delete gate. the name parameter provides the name of the gate to delete
    def deleteGate(self, name):
        url = self._build_url("spectcl/gate/delete", name=str(name))
        self.sendRequest(url)


    # edit/create gate. It creates or redefines an existing gate.
    def createGate(self, name, types, parameters, boundaries, maskval="*"):

        url = self._build_url("spectcl/gate/edit", name=str(name), type=str(types))
        if str(types) == "s": # slice
            url += "&parameter="+self._q(parameters[0])+"&low="+self._q(boundaries[0])+"&high="+self._q(boundaries[1])
        elif str(types) == "gs": # gamma slice
            for i in parameters:
                url += "&parameter="+self._q(i)
            url += "&low="+self._q(boundaries[0])+"&high="+self._q(boundaries[1])
        elif (str(types) == "c" or str(types) == "b"):  # contour or band
            url += "&xparameter="+self._q(parameters[0])+"&yparameter="+self._q(parameters[1])
            for point in boundaries:
                url += "&xcoord="+self._q(point["x"])+"&ycoord="+self._q(point["y"])
        elif (str(types) == "gc" or str(types) == "gb"):  # gamma contour or band
            for i in parameters:
                url +="&parameter="+self._q(i)
            for point in boundaries:
                url += "&xcoord="+self._q(point["x"])+"&ycoord="+self._q(point["y"])
        elif (str(types) == "em" or str(types) == "am" or str(types) == "nm"):  # bit mask
            url += "&parameter="+self._q(parameters)+"&value="+self._q(maskval)
        #want + gate here for c and b gates on m2 spectrum (for cutiepie)
        elif str(types) == "+":
            # Pass the type RAW and let _build_url's urlencode encode it once.
            # Pre-encoding it here was correct while the URLs were concatenated
            # by hand; once _build_url took over the encoding, the "%" in "%2B"
            # got encoded again and the server received the text "%2B".
            url = self._build_url("spectcl/gate/edit", name=str(name), type="+")
            for i in parameters:
                url +="&gate="+self._q(i)
        elif str(types) == "vs+":
            self.createVectorOrSlice(name, parameters[0], boundaries[0], boundaries[1])
            return
        elif str(types) == "vs*":
            self.createVectorAndSlice(name, parameters[0], boundaries[0], boundaries[1])
            return
        self.sendRequest(url)


    # (create1DGate / createMaskGate / listSource / unbindById removed 2026-07-08
    #  — dead code that referenced undefined names. Recreate from
    #  createGate/create2DGate patterns if ever needed.)

    def createVectorSlice(self, name, type, vector, low, high):
        """Create/edit a generic vector slice in SpecTcl: Parameters name
        -name of the new condition. type -type of the condition ('vs+' or
        'vs*' only) vector -name of a vector parameter."""
        if type == 'vs%2B':
            type = 'vs+'
        if type not in ['vs+', 'vs*'] :
            raise Exception(f'Invalid gate type: {type} must be either "vs+ or "vs*"')
        url = self._build_url("spectcl/gate/edit", name=name, type=type, parameter=vector, low=low, high=high)
        self.sendRequest(url)

    def createVectorAndSlice(self, name, vector, low, high):
        self.createVectorSlice(name, 'vs*', vector, low, high)
    def createVectorOrSlice(self, name, vector, low, high):
        self.createVectorSlice(name, 'vs+', vector, low, high)


    # Creates a simple 2d gate. This must be of type c/b or gc/gb.
    def create2DGate(self, name, types, parameters, boundaries):
        url = self._build_url("spectcl/gate/edit", name=str(name), type=str(types))
        if (str(types) == "c" or str(types) == "b"):  # contour or band
            url += "&xparameter="+self._q(parameters[0])+"&yparameter="+self._q(parameters[1])
            for point in boundaries:
                url += "&xcoord="+self._q(point[0])+"&ycoord="+self._q(point[1])
        elif (str(types) == "gc" or str(types) == "gb"):  # gamma contour or band
            for i in parameters:
                url +="&parameter="+self._q(i)
            for point in boundaries:
                url += "&xcoord="+self._q(point[0])+"&ycoord="+self._q(point[1])
        else:
            raise Exception("Only c/b and gc/gb types are allowed")
        self.sendRequest(url)




    # list the gates applied to a spectrum. The result is an object with
    # attributes status and detail.
    def applylistgate(self, pattern="*"):
        url = self._build_url("spectcl/apply/list", pattern=str(pattern))
        response = self.sendRequest(url)
        if response is None :
            return []
        detail = json.loads(response.decode()).get("detail", [])
        return detail if isinstance(detail, list) else []


    # gate application. Applies the gate to a spectrum.
    def applyGate(self, gate, spectrum):
        url = self._build_url("spectcl/apply/apply", gate=str(gate), spectrum=str(spectrum))
        self.sendRequest(url)


    ############################################################
    # Attaching data sources
    ############################################################

    # The SpecTcl attach command can be accessed using the REST plugin. The
    # query parameters are: type - options are file (to read data from a file)
    # or pipe (to read data from a program on the other end of a pipe) source
    # - the source string expected to attach the source type.
    #          for "pipe" full command string
    # size (optional) - sets the blocking factor for reads from the data source (default 8192)
    # format (optional) - sets the data format. Acceptable values are ring (default), nscl (NSCLDAQ before v10),
    #                     jumbo (fixed length buffers longer than 128K bytes from NSCLDAQ before v10), filter (XDR filter data)
    def attachSource(self, types, source, size="8192", formats="ring"):
        url = self._build_url("spectcl/attach/attach", type=str(types), source=str(source), size=str(size), format=str(formats))
        self.sendRequest(url)


    # starts the data analysis
    def startSource(self):
        url = self._build_url("spectcl/attach/start")
        self.sendRequest(url)




    ############################################################
    # Binding spectra to display memory
    ############################################################

    # bind all spectra to display memory
    def sbindall(self):
        url = self._build_url("spectcl/sbind/all")
        self.sendRequest(url)


    # bind all spectra named in all instances of the spectrum query parameter to display memory.
    # spectra is a list
    def sbindSpectrum(self, spectra):
        url = self._build_url("spectcl/sbind/sbind")
        for spectrum in spectra:
            url += "&spectrum=" + self._q(spectrum)
        self.sendRequest(url)


    # return information about the bindings of spectra that match the pattern query parameter interpreted as a glob spectrum name match string.
    # Each of the objects has the following fields:
    #  spectrumid
    #  name
    #  binding
    def listsbind(self, pattern=""):
        url = self._build_url("spectcl/sbind/list", filter=str(pattern))
        response = self.sendRequest(url)
        if response is None:
            return []
        detail = json.loads(response.decode()).get("detail", [])
        return detail if isinstance(detail, list) else []


    ############################################################
    # Fit command
    ############################################################

    # Provides access to the SpecTcl fit command. fit provides you with the
    # ability to fit spectra to arbitrary functions.

    # creates and fits a function to a spectrum. The query parameters are:
    # name - name associated to the fit spectrum - spectrum on which the fit
    # is computed low, high - the channel coordinates over which the fit is to
    # be computed type - type of fit to perform.
    def createFit(self, fitname, spectrum, low, high, fittype):
        url = self._build_url("spectcl/fit/create", name=str(fitname), spectrum=str(spectrum), low=str(low), high=str(high), type=str(fittype))
        self.sendRequest(url)


    # As spectra accumulate, fit data will be outdated. This allows the fit
    # information to be recomputed to match current data.
    def updateFit(self, pattern="*"):
        url = self._build_url("spectcl/fit/update", filter=str(pattern))
        response = self.sendRequest(url)
        if response is None :
            return {}
        fit_dict = json.loads(response.decode())
        return fit_dict["detail"]


    # deletes a fit
    def deleteFit(self, name):
        url = self._build_url("spectcl/fit/delete", name=str(name))
        self.sendRequest(url)


    # list of all created fits. The result is a detail attribute that is an
    # array.
    def listFit(self, pattern=""):
        url = self._build_url("spectcl/fit/list", filter=str(pattern))
        response = self.sendRequest(url)
        if response is None :
            return []
        detail = json.loads(response.decode()).get("detail", [])
        return detail if isinstance(detail, list) else []


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
        url = self._build_url("spectcl/fold/list", filter=str(pattern))
        response = self.sendRequest(url)
        if response is None :
            return []
        detail = json.loads(response.decode()).get("detail", [])
        return detail if isinstance(detail, list) else []


    # apply gamma gate to a list of spectra
    def applyFold(self, gate, spectra):
        url = self._build_url("spectcl/fold/apply", gate=str(gate))
        for spectrum in spectra:
            url += "&spectrum=" + self._q(spectrum)
        self.sendRequest(url)


    # unfolds a spectrum
    def removeFold(self, spectrum):
        url = self._build_url("spectcl/fold/remove", spectrum=str(spectrum))
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
        url = self._build_url("spectcl/channel/get", spectrum=str(name))
        for key, value in kwargs.items():
            url += "&" + key + "=" + self._q(value)
        response = self.sendRequest(url)
        if response is None :
            return {}
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
        url = self._build_url("spectcl/channel/set", spectrum=str(name))
        for key, value in kwargs.items():
            if (key == "xvalue" or key == "yvalue"):
                key = "value"
            url += "&" + key + "=" + self._q(value)
        self.sendRequest(url)


    ############################################################
    # Clear spectra
    ############################################################

    def spectrumClear(self, pattern="*"):
        url = self._build_url("spectcl/spectrum/zero", filter=str(pattern))
        self.sendRequest(url)


    def spectrumAllClear(self):
        self.spectrumClear("*")


    ############################################################
    # Projecting spectra
    ############################################################

    # Project an existing spectrum onto one of the axes creating a new
    # spectrum. The query parameters are: snapshot - boolean value.
    # contour (optional) - if specified, this is a contour that must have been displayable on the source spectrum. The projected spectrum
    #                      The projected spectrum is initially populated only with counts that are within that contour. Furthermore, if the
    #                      projected spectrum is not a snapshot spectrum, it is gated on that contour so that the projection remains faithful
    #                      as new data arrive.
    def createProjection(self, snapshot, source, newname, direction, contour=""):
        url = self._build_url("spectcl/project", snapshot=str(snapshot), source=str(source), newname=str(newname), direction=str(direction), contour=str(contour))
        self.sendRequest(url)

    ############################################################
    # Spectrum underflow and overflow statistics
    ############################################################

    # Returns the underflow and overflow statistics for the spectra whose names match the optional pattern query.
    # Each object has the attributes:
    # name
    # underflows - array of per-axis underflow counts (x, then y for 2-d spectra)
    # overflows - array of per-axis overflow counts (x, then y for 2-d spectra)
    # Always a list: SpecTcl answers an error with a string or an int in
    # "detail", which a caller iterating the objects would index as if it were
    # one of them. The other list-returning endpoints here make the same promise.
    def getSpectrumStats(self, pattern="*"):
        url = self._build_url("spectcl/specstats", filter=str(pattern))
        response = self.sendRequest(url)
        if response is None :
            return []
        detail = json.loads(response.decode()).get("detail", [])
        return detail if isinstance(detail, list) else []


    ############################################################
    # Accessing tree variable command
    ############################################################

    # variable list. The detail attribute is an array of objects.
    def listVariable(self):
        url = self._build_url("spectcl/treevariable/list")
        response = self.sendRequest(url)
        if response is None :
            return []
        detail = json.loads(response.decode()).get("detail", [])
        return detail if isinstance(detail, list) else []


    # Tree variable values can be changed. Note that the units are NOT optional
    def setVariable(self, name, value, units):
        url = self._build_url("spectcl/treevariable/set", name=str(name), value=str(value), units=str(units))
        self.sendRequest(url)


    # tree variables have a flag that indicates if they have been modified in
    # the life of the SpecTcl run. This flag is normally used to limit the
    # amount of information that must be saved in files that capture the
    # SpecTcl analysis state.
    def checkVariable(self):
        url = self._build_url("spectcl/treevariable/list")
        response = self.sendRequest(url)
        if response is None :
            return {}
        var_dict = json.loads(response.decode())
        return var_dict["detail"]


    # it is possible to change the flag above
    def setFlagVariable(self, name):
        url = self._build_url("spectcl/treevariable/setchanged", name=str(name))
        self.sendRequest(url)


    # There are cases where it's important to fire Tcl traces associated with tree variables. If not supplied
    # any pattern it defaults to all variable names
    def traceVariable(self, pattern="*"):
        url = self._build_url("spectcl/treevariable/firetraces", filter=str(pattern))
        self.sendRequest(url)


    ############################################################
    # Accessing filter command
    ############################################################

    # Creation of filters. Parameter has to be a list in input.
    def createFilter(self, name, gate, parameters):
        url = self._build_url("spectcl/filter/new", name=str(name), gate=str(gate))
        for i in parameters:
            url += "&parameter=" + self._q(i)
        self.sendRequest(url)


    # Delete a filter. The only parameter is the name of the filter we want to remove
    def deleteFilter(self, name):
         url = self._build_url("spectcl/filter/delete", name=str(name))
         self.sendRequest(url)


    # Enable/disable filter
    def enableFilter(self, name):
         url = self._build_url("spectcl/filter/enable", name=str(name))
         self.sendRequest(url)


    def disableFilter(self, name):
         url = self._build_url("spectcl/filter/disable", name=str(name))
         self.sendRequest(url)


    # List of filter. Each object will describe a single filter and contains the following attributes:
    # name - name of the filter
    # gate - name of the gate applied to the filter
    # file - name of the output file to which the filter writes
    # parameters - an array of parameters names written to the output file for each event that passes the gate
    # enabled - if the filter is enabled, this attribute has value "enabled" otherwise "disabled"
    # format - contains the format string i.e. xdr
    def listFilter(self, pattern="*"):
        url = self._build_url("spectcl/filter/list", filter=str(pattern))
        response = self.sendRequest(url)
        if response is None :
            return []
        detail = json.loads(response.decode()).get("detail", [])
        return detail if isinstance(detail, list) else []


    ############################################################
    # Integrate command
    ############################################################

    # Integrates the interior of a gate, an explicit 1-d area of interest,
    # or an arbitrary polygon. Returns centroid, counts and fwhm — scalars
    # for a 1-d spectrum, [x, y] pairs for a 2-d one.

    def integrateGate(self, name, gate):
        url = self._build_url("spectcl/integrate", spectrum=str(name), gate=str(gate))
        response = self.sendRequest(url)
        if response is None :
            return {}
        int_dict = json.loads(response.decode())
        return int_dict["detail"]


    def integrate1D(self, name, low, high):
        url = self._build_url("spectcl/integrate", spectrum=str(name), low=str(low), high=str(high))
        response = self.sendRequest(url)
        if response is None :
            return {}
        int_dict = json.loads(response.decode())
        return int_dict["detail"]


    # points are a list of lists
    def integrate2D(self, name, points):
        url = self._build_url("spectcl/integrate", spectrum=str(name))
        for point in points:
            # encoded like every other appended value; these are floats today,
            # so this is consistency rather than a live fix
            url += "&xcoord="+self._q(point[0])+"&ycoord="+self._q(point[1])
        response = self.sendRequest(url)
        if response is None :
            return {}
        int_dict = json.loads(response.decode())
        return int_dict["detail"]


    ############################################################
    # Create parameter command
    ############################################################

    # Provides support for creating a new parameter definition. A parameter
    # definition makes a correspondence between a name and a slot in the
    # CEvent pseudo array that SpecTcl event processing pipeline fills in.
    def createRawParameter(self, name, number, **kwargs):
        url = self._build_url("spectcl/rawparameter/new", name=str(name), number=str(number))
        for key, value in kwargs.items():
            url += "&" + key + "=" + self._q(value)
        self.sendRequest(url)


    # delete a parameter by name or id
    def deleteRawParameter(self, par):
        url = self._build_url("spectcl/rawparameter/delete")
        if isinstance(par, str):
            url += "?name="+str(par)
        else:
            url += "?id="+str(par)
        self.sendRequest(url)


    # list parameter by name or id. The detail attribute of the returned JSON
    # contains an array of objects.
    # resolution - (optional)
    # low - (optional)
    # high - (optional)
    # units - (optional)
    def listRawParameter(self, par="*"):
        url = self._build_url("spectcl/rawparameter/list")
        if isinstance(par, str):
            url += "?pattern="+str(par)
        else:
            url += "?id="+str(par)
        response = self.sendRequest(url)
        if response is None :
            return []
        detail = json.loads(response.decode()).get("detail", [])
        return detail if isinstance(detail, list) else []


    ############################################################
    # Pseudo command
    ############################################################

    # It provides the ability to create a new parameter frome existing parameters. The new parameter is computed using a script that is passed to
    # the pseudo command when the parameter is generated (see docs for pseudo in SpecTcl command reference)

    # create pseudo command. The parameters should exist before creating the
    # pseudoparameter.
    def createPseudo(self, name, body, parameters):
        url = self._build_url("spectcl/pseudo/create", name=str(name), body=str(body))
        for i in parameters:
            url += "&parameter=" + self._q(i)
        self.sendRequest(url)


    # list pseudo parameters. The return value detail is array where each
    # object describes one pseudo parameter.
    def listPseudo(self, par="*"):
        url = self._build_url("spectcl/pseudo/list", pattern=str(par))
        response = self.sendRequest(url)
        if response is None :
            return []
        detail = json.loads(response.decode()).get("detail", [])
        return detail if isinstance(detail, list) else []


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
        url = self._build_url("spectcl/sread", filename=str(name))
        for key, value in kwargs.items():
            url += "&" + key + "=" + self._q(value)
        self.sendRequest(url)


    ############################################################
    # ringformat command
    ############################################################

    # set the major and minor version of the ringformat command
    def ringFormat(self, major, minor="0"):
        url = self._build_url("spectcl/ringformat", major=str(major), minor=str(minor))
        self.sendRequest(url)


    ############################################################
    # unbind command
    ############################################################

    # it allows to access the capabilities of the SpecTcl unbind command by name or id
    def unbindByName(self, names):
        url = self._build_url("spectcl/unbind/byname")
        for name in names:
            url += "&name=" + self._q(name)
        self.sendRequest(url)




    def unbindAll(self):
        url = self._build_url("spectcl/unbind/all")
        self.sendRequest(url)


    ############################################################
    # ungate command
    ############################################################

    # this command removes any gate condition from a spectrum. Names is a list.
    def ungateSpectum(self, names):
        url = self._build_url("spectcl/ungate")
        for name in names:
            url += "&name=" + self._q(name)
        self.sendRequest(url)


    ############################################################
    # swrite command
    ############################################################

    # swrite supports writing spectrum files on request from clients. The
    # query parameters are: filename - (mandatory) name of the file to write
    # spectrum - (mandatory, can be multiple) each occurrence of this query
    # parameter names a spectrum to be written to a file.
    def swrite(self, name, spectra, formats="ascii"):
        url = self._build_url("spectcl/swrite", file=str(name))
        for spectrum in spectra:
            url += "&spectrum=" + self._q(spectrum)
        url += "&format=" + str(formats)
        self.sendRequest(url)


    ############################################################
    # start/stop analysis command
    ############################################################

    # data analysis from the source can be started or stopped using the following queries:
    def startAnalysis(self):
        url = self._build_url("analyze/start")
        self.sendRequest(url)


    def stopAnalysis(self):
        url = self._build_url("analyze/stop")
        self.sendRequest(url)


    ############################################################
    # root tree command
    ############################################################

    # note that roottree is present only in a SpecTcl instance that has loaded
    # the rootinterface package. this interface allows you to create ROOT
    # output trees.

    # create ROOT tree. The query parameters are: tree - name of the tree
    # parameter - (multiple) each of these parameters specifies a patternthat
    # can have glob wildcard characters.
    def createROOTtree(self, name, parameters, gate=""):
        url = self._build_url("roottree/create", tree=str(name))
        for i in parameters:
            url += "&parameter=" + self._q(i)
        url += "&gate=" + self._q(gate)
        self.sendRequest(url)


    # delete a previously created tree
    def deleteROOTtree(self, name):
        url = self._build_url("roottree/delete", tree=str(name))
        self.sendRequest(url)


    # list all created trees. Provides a return value with the list of the
    # root tree objects with names that match the optional pattern query.
    def listROOTtree(self, pattern="*"):
        url = self._build_url("roottree/list", filter=str(pattern))
        response = self.sendRequest(url)
        if response is None :
            return []
        detail = json.loads(response.decode()).get("detail", [])
        return detail if isinstance(detail, list) else []


    ############################################################
    # traces command
    ############################################################

    # Traces are a mechanism to allow SpecTcl scripts to be informed of
    # changes to parameters, spectrum, and gate dictionaries. Tracing is
    # problematic for the REST interface.

    # start the trace service. This function informs the REST server that the
    # client will be interested in trace data.
    def startTraces(self, seconds):
        url = self._build_url("spectcl/trace/establish", retention=str(seconds))
        response = self.sendRequest(url)
        if response is None :
            return {}
        trace_dict = json.loads(response.decode())
        return trace_dict["detail"]


    # stop the trace service. Once the application no longer requires trace information, or as it is cleaning up for exit, a request is made
    def stopTraces(self, token):
        url = self._build_url("spectcl/trace/done", token=str(token))
        self.sendRequest(url)


    # poll traces. The detail attribute of the returned JSON will be an
    # object.

    # Returns the detail object on success, or None when the poll produced no
    # usable answer: the request failed, or the body could not be parsed. The
    # two outcomes are kept apart because the caller retries a failed poll but
    # accepts an empty detail, which is the normal "nothing fired since the
    # last poll" reply.
    def pollTraces(self, token):
        url = self._build_url("spectcl/trace/fetch", token=str(token))
        response = self.sendRequest(url)
        if response is None:
            return None
        try:
            return json.loads(response.decode())["detail"]
        except (ValueError, KeyError, TypeError, AttributeError):
            self.logger.warning('pollTraces - could not parse the trace reply')
            return None


    ############################################################
    # general functions for communication and error handling
    ############################################################

    def sendRequest(self, url):
        try:
            self.logger.debug("REST GET %s", url)
            status, content = self._http.request(url, method="GET")
            decoded = content.decode()
            if any(kw in decoded for kw in _BAD_REST_KEYWORDS):
                self.logger.warning('sendRequest -- suspicious REST request status: %s', content)
                return None
            return content
        except Exception:
            self.logger.error('sendRequest -- check Server/User/REST Port/Mirror Port')
            return None

    def checkSpecTclREST(self):
        url = f"http://{self.server}:{self.rest}/spectcl/spectrum/list?filter=*"
        try:
            status, content = self._http.request(url, method="GET")
            data = json.loads(content.decode())
            ok = isinstance(data, dict) and data.get("status") == "OK"
            self.logger.info(f"[PyREST] REST health {'OK' if ok else 'FAIL'} via {url}")
            return ok
        except Exception:
            self.logger.warning(f"[PyREST] REST health FAIL via {url} (exception)")
            return False

    # size in bytes of SpecTcl's display shared memory (the region the mirror
    # copies); same endpoint the mirror client itself uses to size the mapping
    def shmemSize(self):
        url = self._build_url("spectcl/shmem/size")
        response = self.sendRequest(url)
        if response is None:
            return None
        try:
            return int(json.loads(response.decode()).get("detail"))
        except (ValueError, TypeError, json.JSONDecodeError):
            self.logger.warning('shmemSize - could not parse response')
            return None
