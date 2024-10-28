#!/usr/bin/env python

from fit_function import FitFunction
import numpy as np

class GausFit(FitFunction):
    def __init__(self, amplitude, mean, standard_deviation):
        params = np.array([amplitude, mean, standard_deviation], dtype=np.float64)
        super().__init__(params)

    # function defined by the user
    def model(self, x, params):
        """Unnormalized Gaussian."""
        return params[0]*np.exp(-(x-params[1])**2 / (2*params[2]**2))

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

class GausFitBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, amplitude=1000, mean=100, standard_deviation=10):
        if not self._instance:
            self._instance = GausFit(amplitude, mean, standard_deviation)
        return self._instance
