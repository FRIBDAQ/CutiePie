#!/usr/bin/env python

from fit_function import FitFunction
import numpy as np

class GPol1Fit(FitFunction):
    def __init__(self, amplitude, mean, standard_deviation, p0, p1, f):
        params = np.array([amplitude, mean, standard_deviation, p0, p1, f],
                          dtype=np.float64)
        super().__init__(params)
        
    # function defined by the user
    def model(self, x, params):
        """Gaussian + linear background."""
        def gauss(x, params):
            return params[0]*np.exp(-(x-params[1])**2 / (2*params[2]**2))
        def pol1(x, params):
            return params[0] + params[1]*x
        frac = params[5]
        return frac*gauss(x, params[0:3]) + (1-frac)*pol1(x, params[3:5])

    def set_initial_parameters(self, x, y, params):
        """
        Use None as 'not seeded', and treat 0.0 as a valid user seed.
        """
        # Make sure we can index like a list, even if it's a numpy array
        p = list(params)

        # amplitude (Gaussian)
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
            s0 = float(np.std(x))  # width of fit range

        # background p0 (offset)
        if p[3] is not None:
            b0 = float(p[3])
        else:
            b0 = float(min(y[0], y[-1]))

        # background p1 (slope)
        if p[4] is not None:
            b1 = float(p[4])
        else:
            b1 = float((y[-1] - y[0]) / (x[-1] - x[0]))

        # fraction f
        if p[5] is not None:
            f0 = float(p[5])
        else:
            f0 = 0.9

        clean_params = np.array([a0, m0, s0, b0, b1, f0], dtype=np.float64)
        super().set_initial_parameters(x, y, clean_params)


class GPol1FitBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, amplitude=1000, mean=100, standard_deviation=10,
                 p0=100, p1=10, f=0.9):
        if not self._instance:
            self._instance = GPol1Fit(amplitude, mean, standard_deviation,
                                      p0, p1, f)
        return self._instance
