import numpy as np
from scipy.optimize import minimize

# Invalid values are ordinary here: the optimiser walks through parameter sets
# that make the model produce nan, and the objective is defined for them. That
# is a property of fitting, not of the process, so the suppression is applied
# per fit in start() rather than to every numpy operation in the GUI.

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
            # Replace small values with some small number to ensure log(pred)
            # is valid. I chose to modify the likelihood this way over
            # returning, e.g, np.inf if any pred <= 0 because it prevents huge
            # jumps in the objective function value when the parameters are
            # close to values which give pred <= 0. Also the return value is
            # always defined.
            pred = np.maximum(pred, 1e-10)
            return -np.sum(y*np.log(pred) - pred)

    def start(self, x, y, xmin, xmax, params, axis, fit_results):
        """Perform the fit and show the results. Return the data to plot."""
        with np.errstate(invalid='ignore'):
            return self._start(x, y, xmin, xmax, params, axis, fit_results)

    def _start(self, x, y, xmin, xmax, params, axis, fit_results):
        self.set_initial_parameters(x, y, params)
        # Use BFGS and higher-order Jacobian approx. BFGS provides appoximate
        # Hessian for extracting parameter uncertainties without an additional
        # step (as would be needed for e.g., simplex method):
        result = minimize(self.neg_log_likelihood_p,
                          x0=self.p_init,
                          args=(x,y),
                          method='bfgs',
                          jac='3-point',
                          options={"gtol": 1e-3})
        # Most often an issue with final precision on error estimates:
        if not result.success:
            print(f"WARNING: fit did not terminate successfully:\n{result}")

        # Remember the fitted parameters so callers (and tests) can re-evaluate
        # the model at the solution.
        self._last_params = result.x

        # Pearson goodness-of-fit on the fitted bins. These models are fit by
        # Poisson MLE (no chi2 falls out of the optimizer), so compute one here
        # for reporting parity with the Alpha* fits and so Save Fit can carry it.
        chi2 = redchi = float('nan')
        ndof = 0
        try:
            pred = np.maximum(self.model(np.asarray(x, dtype=float), result.x),
                              1e-10)
            resid = np.asarray(y, dtype=float) - pred
            chi2 = float(np.sum(resid**2 / pred))
            ndof = int(len(x) - len(result.x))
            redchi = chi2 / max(ndof, 1)
            fit_results.append(
                f'[stats] chi-square={chi2:.3f} ; '
                f'reduced chi-square={redchi:.3f} ; ndof={ndof}')
        except Exception:
            pass  # goodness-of-fit is best-effort; never block the fit

        fitln = None # Data to plot

        try:
            x_fit = np.linspace(x[0],x[-1], 10000)
            y_fit = self.model(x_fit, result.x)
            fitln, = axis.plot(x_fit,y_fit, 'r-')
            # Inverse Hessian is ~ Cov matrix:
            for i in range(len(result.x)):
                s = 'Par['+str(i)+']: '+str(round(result.x[i],6))+'+/-'+str(round(np.sqrt(result.hess_inv[i][i]),6))
                fit_results.append(s)
        except Exception:
            pass # Can't plot, ignored

        if fitln is not None:
            fitln.chi2 = chi2
            fitln.redchi = redchi
            fitln.ndof = ndof

        return fitln
