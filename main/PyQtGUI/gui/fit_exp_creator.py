#!/usr/bin/env python

from fit_function import FitFunction
import numpy as np

class ExpFit(FitFunction):
    def __init__(self, a, b, c):
        params = [a, b, c]
        super().__init__(params)

    def model(self, x, params):
        """Simple exponential with constant offset"""
        return params[0] + params[1]*np.exp(params[2]*(x-x[0]))

    def set_initial_parameters(self, x, y, params):
        super().set_initial_parameters(x, y, params)
        if (params[0] != 0.0):
            self.p_init[0] = params[0]
        else:
            self.p_init[0] = min(y[0], y[-1])
        if (params[1] != 0.0):
            self.p_init[1] = params[1]
        else:
            self.p_init[1] = max(y) - self.p_init[0]
        if (params[2] != 0.0):
            self.p_init[2] = params[2]
        else:
            self.p_init[2] = -1

class ExpFitBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, a=1, b=5, c=-1):
        if not self._instance:
            self._instance = ExpFit(a, b, c)
        return self._instance
