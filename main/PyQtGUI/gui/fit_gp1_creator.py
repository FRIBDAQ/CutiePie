#!/usr/bin/env python

from fit_function import FitFunction
import numpy as np

def gauss(self, x, params):
    """Un-normalized Gaussian."""
    return params[0]*np.exp(-(x-params[1])**2 / (2*params[2]**2))

def pol1(self, x, params):
    """Linear function."""
    return params[0] + params[1]*x

class GPol1Fit(FitFunction):
    def __init__(self, amplitude, mean, standard_deviation, p0, p1, f):
        params = np.array([amplitude, mean, standard_deviation, p0, p1, f],
                          dtype=np.float64)
        super().__init__(params)
        
    # function defined by the user
    def model(self, x, params):
        """Gaussian + linear background."""
        frac = params[5]
        return frac*gauss(x, params[0:3]) + (1-frac)*pol1(x, params[3:5])

    def set_initial_parameters(self, x, y, params):
        super().set_initial_parameters(x, y, params)
        if (params[0] != 0.0):
            self.p_init[0] = params[0]
        else:
            self.p_init[0] = np.max(y)
        if (params[1] != 0.0):
            self.p_init[1] = params[1]
        else:
            self.p_init[1] = x[np.argmax(y)]
        if (params[2] != 0.0):
            self.p_init[2] = params[2]
        else:
            self.p_init[2] = np.std(x) # From width of fit range.
        if (params[3] != 0.0):
            self.p_init[3] = params[3]
        else:
            self.p_init[3] = min(y[0], y[-1])
        if (params[4] != 0.0):
            self.p_init[4] = params[4]
        else:
            self.p_init[4] = (y[-1] - y[0])/(x[-1] - x[0])
        if (params[5] != 0.0):
            self.p_init[5] = params[5]
        else:
            self.p_init[5] = 0.9

class GPol1FitBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, amplitude=1000, mean=100, standard_deviation=10,
                 p0=100, p1=10, f=0.9):
        if not self._instance:
            self._instance = GPol1Fit(amplitude, mean, standard_deviation,
                                      p0, p1, f)
        return self._instance
