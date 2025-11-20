#!/usr/bin/env python

from fit_function import FitFunction
import numpy as np

class Pol2Fit(FitFunction):
    def __init__(self, p0, p1, p2):
        params = np.array([p0, p1, p2], dtype=np.float64)
        super().__init__(params)

    def model(self, x, params):
        """Quadratic function."""
        return params[0] + params[1]*x + params[2]*x**2
    
    def set_initial_parameters(self, x, y, params):
        super().set_initial_parameters(x, y, params)

        # p0
        if params[0] is not None:
            self.p_init[0] = params[0]
        else:
            self.p_init[0] = min(y[0], y[-1])

        # p1
        if params[1] is not None:
            self.p_init[1] = params[1]
        else:
            self.p_init[1] = (y[-1] - y[0]) / (x[-1] - x[0])

        # p2
        if params[2] is not None:
            self.p_init[2] = params[2]
        else:
            self.p_init[2] = 0.001  # some small-ish number

            
class Pol2FitBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, p0=100, p1=10, p2=1):
        if not self._instance:
            self._instance = Pol2Fit(p0, p1, p2)
        return self._instance
