#!/usr/bin/env python

from fit_function import FitFunction
import numpy as np

class ExpFit(FitFunction):
    def __init__(self, a, b, c):
        params = np.array([a, b, c], dtype=np.float64)
        super().__init__(params)

    def model(self, x, params):
        """Simple exponential with constant offset"""
        return params[0] + params[1]*np.exp(params[2]*(x-x[0]))

    def set_initial_parameters(self, x, y, params):
        super().set_initial_parameters(x, y, params)
        if params[0] is not None:
            self.p_init[0] = float(params[0])
        else:
            self.p_init[0] = float(min(y[0], y[-1]))
        if params[1] is not None:
            self.p_init[1] = float(params[1])
        else:
            self.p_init[1] = float(max(y)) - self.p_init[0]
        if params[2] is not None:
            self.p_init[2] = float(params[2])
        else:
            self.p_init[2] = -1.0

class ExpFitBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, a=1, b=5, c=-1):
        if not self._instance:
            self._instance = ExpFit(a, b, c)
        return self._instance
