#!/usr/bin/env python

from fit_function import FitFunction
import numpy as np

class GPol2Fit(FitFunction):
    def __init__(self, amplitude, mean, standard_deviation, p0, p1, p2, f):
        params = np.array([amplitude, mean, standard_deviation, p0, p1, p2, f],
                          dtype=np.float64)
        super().__init__(params)
        
    # function defined by the user
    def model(self, x, params):
        """Gaussian + quadratic background."""
        def gauss(x, params):
            return params[0]*np.exp(-(x-params[1])**2 / (2*params[2]**2))
        def pol2(x, params):
            return params[0] + params[1]*x + params[2]*x**2
        frac = params[6]
        return frac*gauss(x, params[0:3]) + (1-frac)*pol2(x, params[3:6])

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
            self.p_init[2] = abs(self.p_init[1])/10
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
            self.p_init[5] =0.0
        if (params[6] != 0.0):
            self.p_init[6] = params[6]
        else:
            self.p_init[6] = 0.9

class GPol2FitBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, amplitude=1000, mean=100, standard_deviation=10,
                 p0=100, p1=10, p2=1, f=0.9):
        if not self._instance:
            self._instance = GPol2Fit(amplitude, mean, standard_deviation,
                                      p0, p1, p2, f)
        return self._instance
