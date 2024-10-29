import io
import sys, os
sys.path.append(os.getcwd())
import numpy as np
from scipy.optimize import minimize

np.seterr(invalid='ignore') # Suppress some 'invalid value' runtime warnings

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
        """Poisson negative log-likelihood. Fit parameters must be first 
        argument and are initially set by x0 in the `minimize` call. x, y 
        are passed as additional args.
        """
        # Make sure the model is implemented:
        try:
            pred = self.model(x, params)
        except NotImplementedError:
            print("ERROR: Model function is not defined!")
            return np.inf
        else:
            # Negative log likelihood (nll) or np.inf if data has values with
            # invalid logaritms. Note only terms which depend on the params
            # are kept in the nll. 
            if np.any(pred <= 0):
                return np.inf
            else:
                return -np.sum(y*np.log(pred) - pred)

    def start(self, x, y, xmin, xmax, params, axis, fit_results):
        """Perform the fit and show the results. Return the data to plot."""
        self.set_initial_parameters(x, y, params)
        # Use BFGS and higher-order Jacobian approx. BFGS provides appoximate
        # Hessian for extracting parameter uncertainties without an additional
        # step (as would be needed for e.g., simplex method):
        result = minimize(self.neg_log_likelihood_p,
                          x0=self.p_init,
                          args=(x,y),
                          method='bfgs',
                          jac='3-point')
        # Most often an issue with final precision on error estimates:
        if not result.success:
            print(f"WARNING: fit did not terminate successfully:\n{result}")

        fitln = None # Data to plot
        
        try:
            x_fit = np.linspace(x[0],x[-1], 10000)
            y_fit = self.model(x_fit, result.x)
            fitln, = axis.plot(x_fit,y_fit, 'r-')
            # Inverse Hessian is ~ Cov matrix:
            for i in range(len(result.x)):
                s = 'Par['+str(i)+']: '+str(round(result.x[i],6))+'+/-'+str(round(result.hess_inv[i][i],6))
                fit_results.append(s)
        except:
            pass # Can't plot, ignored
        
        return fitln
