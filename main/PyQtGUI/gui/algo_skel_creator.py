#!/usr/bin/env python
import sys, os
sys.path.append(os.getcwd())

import algo_factory  # noqa: F401 - keep import so the factory can discover this module
        
# Skeleton algo parameters
# param_1 = 5
# param_2 = 'test',
# param_3 = 100

class SkelAlgo:
    def __init__(self, param_1, param_2, param_3):
        self.param_1 = param_1
        self.param_2 = param_2
        self.param_3 = param_3        

    # implementation of the algorithm, the argument are mandatory even if not used
    def start(self, data, weigths, nclusters, axis, figure=None):
        print("Skeleton example. Implementation is all yours.")
        
class SkelAlgoBuilder:
    def __call__(self, param_1 = 5, param_2 = 'test', param_3 = 100, **_ignored):
        return SkelAlgo(param_1, param_2, param_3)
