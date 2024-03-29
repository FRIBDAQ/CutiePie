#!/usr/bin/env python
import io
import sys, os
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

import fit_factory

class Pol2Fit:
    def __init__(self, p0, p1, p2):
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2

    # function defined by the user
    def pol2(self, x, p0, p1, p2):
        return p0+p1*x+p2*x*x

    # implementation of the fitting algorithm
    def start(self, x, y, xmin, xmax, fitpar, axis, fit_results):
        fitln = None
        if (fitpar[0] != 0.0):
            self.p0 = fitpar[0]
        else:
            self.p0 = 100
        if (fitpar[1] != 0.0):
            self.p1 = fitpar[1]
        else:
            self.p1 = 10
        if (fitpar[2] != 0.0):
            self.p2 = fitpar[2]
        else:
            self.p2 = 10
        p_init = [self.p0, self.p1, self.p2]
        popt, pcov = curve_fit(self.pol2, x, y, p0=p_init, maxfev=1000000)

        # plotting fit curve and printing results
        try:
            x_fit = np.linspace(x[0],x[-1], 10000)
            y_fit = self.pol2(x_fit, *popt)

            fitln, = axis.plot(x_fit,y_fit, 'r-')
            for i in range(len(popt)):
                s = 'Par['+str(i)+']: '+str(round(popt[i],3))+'+/-'+str(round(pcov[i][i],3))
                fit_results.append(s)
        except:
            pass
        return fitln

class Pol2FitBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, p0=100, p1=10, p2=10):
        if not self._instance:
            self._instance = Pol2Fit(p0, p1, p2)
        return self._instance
