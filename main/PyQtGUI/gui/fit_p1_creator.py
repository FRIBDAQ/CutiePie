#!/usr/bin/env python

from fit_function import FitFunction
import numpy as np

class Pol1Fit(FitFunction):
    def __init__(self, p0, p1):
        params = [p0, p1]
        super().__init__(params)

    def model(self, x, params):
        """Linear function."""
        return params[0] + params[1]*x
    
    def set_initial_parameters(self, x, y, params):
        super().set_initial_parameters(x, y, params)
        if (params[0] != 0.0):
            self.p_init[0] = params[0]
        else:
            self.p_init[0] = min(y[0], y[-1])
        if (params[1] != 0.0):
            self.p_init[1] = params[1]
        else:
            self.p_init[1] = (y[-1] - y[0])/(x[-1] - x[0])
            
class Pol1FitBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, p0=100, p1=10):
        if not self._instance:
            self._instance = Pol1Fit(p0, p1)
        return self._instance
