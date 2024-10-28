#!/usr/bin/env python

from fit_function import FitFunction
import numpy as np

class SkelFit(FitFunction):
    def __init__(self, param_1, param_2, param_3):
        params = np.array([param_1, param_2, param_3], dtype=np.float64)
        super().__init__(params)        

    def model(self, params):
        """Users need to implement thier own fit functions."""
        raise RuntimeError("Implementation of a model function is up to you.")

    #def set_inital_parameters(self, x, y, params):
    #    """Uncomment and override this funciton to customize the parameter
    #    initialization for your fits. Otherwise use the default (read from
    #    fit panel).
    #    """
    #    # Custom initialization goes here.
    
class SkelFitBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, param_1=1, param_2=2, param_3=10, **_ignored):
        if not self._instance:
            self._instance = SkelFit(param_1, param_2, param_3)
        return self._instance
