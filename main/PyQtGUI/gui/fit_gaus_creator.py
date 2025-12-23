#!/usr/bin/env python

from fit_function import FitFunction
import numpy as np

class GausFit(FitFunction):
    def __init__(self, amplitude, mean, standard_deviation):
        params = np.array([amplitude, mean, standard_deviation],
                          dtype=np.float64)
        super().__init__(params)

    def model(self, x, params):
        """Un-normalized Gaussian."""
        sigma = float(abs(params[2]))
        if sigma <= 1e-12:
            sigma = 1e-12
        return params[0] * np.exp(-(x - params[1])**2 / (2 * sigma**2))


    def set_initial_parameters(self, x, y, params):
        """
        Use None as 'not seeded', and treat 0.0 as a valid user seed.
        """
        # Make sure we can index like a list, even if it's a numpy array
        p = list(params)

        # amplitude
        if p[0] is not None:
            a0 = float(p[0])
        else:
            a0 = float(np.max(y))

        # mean
        if p[1] is not None:
            m0 = float(p[1])
        else:
            m0 = float(x[int(np.argmax(y))])

        # sigma
        if p[2] is not None:
            s0 = float(p[2])
        else:
            s0 = float(np.std(x))   # From width of fit range

        # Now call the base class with *numeric only* parameters
        clean_params = np.array([a0, m0, s0], dtype=np.float64)
        super().set_initial_parameters(x, y, clean_params)

class GausFitBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, amplitude=1000, mean=100, standard_deviation=10):
        if not self._instance:
            self._instance = GausFit(amplitude, mean, standard_deviation)
        return self._instance
