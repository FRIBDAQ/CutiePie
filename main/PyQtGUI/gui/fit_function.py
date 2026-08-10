from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize


@dataclass
class FitRequest:
    """What a fit needs: data, the window, and the panel's seed parameters."""
    x: Sequence[float]
    y: Sequence[float]
    xmin: float
    xmax: float
    params: Sequence[float]


@dataclass
class FitResult:
    """What a fit produced. No widgets: the caller decides how to show it,
    which is what makes a plugin testable without Qt or a matplotlib axis."""
    # the optimiser's own verdict. A False here still leaves a usable curve —
    # scipy reports precision loss on a poor seed — so callers that want "did
    # this fit work" should look at the numbers, not only at this flag.
    converged: bool = False
    params: Sequence[float] = ()
    chi2: float = float("nan")
    redchi: float = float("nan")
    ndof: int = 0
    curve: Optional[Tuple[Sequence[float], Sequence[float]]] = None
    stats_line: Optional[str] = None
    param_lines: list = field(default_factory=list)
    message: Optional[str] = None

# Invalid values are ordinary here: the optimiser walks through parameter sets
# that make the model produce nan, and the objective is defined for them. That
# is a property of fitting, not of the process, so the suppression is applied
# per fit in start() rather than to every numpy operation in the GUI.

class FitFunction:
    """Base class for curve fitting by Poisson MLE. It is up to the derived
    classes to implement the model function and (possibly) inital parameter 
    estimation.
    
    """
    def __init__(self, params, **kwargs):
        self.p_init = params
        self._should_abort = None

    def model(self, x, params):
        """Function body. Must be implemented in derived classes."""
        raise NotImplementedError()

    def set_initial_parameters(self, x, y, params):
        """Defaults to reading from the fit panel when called."""
        self.p_init = params[0:len(self.p_init)]

    def neg_log_likelihood_p(self, params, x, y):
        """Poisson negative log-likelihood. Fit parameters must be first
        argument and are initially set by x0 in the `minimize` call."""
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
            # close to values which give pred <= 0.
            pred = np.maximum(pred, 1e-10)
            return -np.sum(y*np.log(pred) - pred)

    def run(self, request):
        """Fit and return a FitResult. Qt-free, so a plugin written against
        this seam can be tested without a widget or an axis."""
        with np.errstate(invalid='ignore'):
            return self._run(request)

    def start(self, x, y, xmin, xmax, params, axis, fit_results):
        """Back-compat shim: the seven-positional signature every existing
        plugin implements. Runs the fit through `run` and does the drawing and
        reporting the old contract expects, returning the plotted line."""
        res = self.run(FitRequest(x=x, y=y, xmin=xmin, xmax=xmax, params=params))
        if res.stats_line:
            fit_results.append(res.stats_line)
        fitln = None
        if res.curve is not None:
            try:
                fitln, = axis.plot(res.curve[0], res.curve[1], 'r-')
                # the parameter lines share the plot's try in the original, so
                # a pad that cannot be drawn on reports the stats and no more
                for line in res.param_lines:
                    fit_results.append(line)
            except Exception:
                fitln = None
        if fitln is not None:
            fitln.chi2 = res.chi2
            fitln.redchi = res.redchi
            fitln.ndof = res.ndof
        return fitln

    def _run(self, request):
        x, y, params = request.x, request.y, request.params
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
            stats_line = (f'[stats] chi-square={chi2:.3f} ; '
                          f'reduced chi-square={redchi:.3f} ; ndof={ndof}')
        except Exception:
            stats_line = None  # goodness-of-fit is best-effort; never block the fit

        curve = None
        param_lines = []
        try:
            x_fit = np.linspace(x[0], x[-1], 10000)
            curve = (x_fit, self.model(x_fit, result.x))
            # Inverse Hessian is ~ Cov matrix:
            for i in range(len(result.x)):
                param_lines.append(
                    'Par['+str(i)+']: '+str(round(result.x[i],6))+'+/-'
                    + str(round(np.sqrt(result.hess_inv[i][i]),6)))
        except Exception:
            curve = None
            param_lines = []

        return FitResult(converged=bool(result.success), params=result.x,
                         chi2=chi2, redchi=redchi, ndof=ndof, curve=curve,
                         stats_line=stats_line, param_lines=param_lines)
