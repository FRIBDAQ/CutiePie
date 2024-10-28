import io
import sys, os
sys.path.append(os.getcwd())
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

np.seterr(invalid='ignore') # suppress some 'invalid value' runtime warnings

class FitFunction:
    """Base class for curve fitting by Poisson MLE. It is up to the derived
    classes to implement the model function and (possibly) inital parameter 
    estimation.
    
    """
    def __init__(self, params):
        self.p_init = params # Initial guesses

    def model(self, x, params):
        """Function body. Must be implemented in derived classes."""
        raise NotImplementedError()

    def set_initial_parameters(self, x, y, params):
        """Defaults to reading from the fit panel when called."""
        self.p_init = params[0:len(self.p_init)]

    def neg_log_likelihood_p(self, params, x, y):
        """Poisson negative log-likelihood. The fit parameters must be first 
        parameter in the signiture as it is set by x0 in the minimize call.
        """
        pred = self.model(x, params)        
        if np.any(pred <= 0): # protect against log(x <= 0) 
            return np.inf
        return -np.sum(poisson.logpmf(y, pred))

    def start(self, x, y, xmin, xmax, params, axis, fit_results):
        """Perform the fit and show the results. Return the data to plot."""
        fitln = None # data to plot        
        self.set_initial_parameters(x, y, params)
        # Relax the tolerances to get good convergence with finite difference:
        result = minimize(self.neg_log_likelihood_p, x0=self.p_init, args=(x,y), method='bfgs', options={'maxiter': 1000000, 'gtol': 1e-3, 'eps': 1e-6})
        if not result.success:
            print(f"WARNING: fit did not terminate successfully:\n{result}")

        try:
            x_fit = np.linspace(x[0],x[-1], 10000)
            y_fit = self.model(x_fit, result.x)
            fitln, = axis.plot(x_fit,y_fit, 'r-')
            # Inverse Hessian is ~ Cov:
            for i in range(len(result.x)):
                s = 'Par['+str(i)+']: '+str(round(result.x[i],6))+'+/-'+str(round(result.hess_inv[i][i],6))
                fit_results.append(s)
        except:
            pass # can't plot, ignored
        
        return fitln
