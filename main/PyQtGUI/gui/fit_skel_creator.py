#!/usr/bin/env python
# fit_skel_creator.py
#
# Implements "Skeleton" fit as a double EMG (AlphaEMG2+bg) with a linear background.
# Works with your existing fit_factory and GUI:
#   - class name: SkelFit
#   - builder:    SkelFitBuilder
#   - entry point: SkelFit.start(x, y, xmin, xmax, fitpar, axis, fit_results)

import sys, os
sys.path.append(os.getcwd())

import numpy as np
from lmfit import Model, Parameters, fit_report
from scipy.special import erfcx

import fit_factory  # keep import so the factory can discover this module

def _emg2(x, A, mu, sigma, tau1, tau2, eta, b0, b1):
    """
    Numerically stable double EMG (no binwidth; plain y=f(x)) + linear background.
    """
    sigma = max(float(sigma), 1e-9)
    tau1  = max(float(tau1),  1e-9)
    tau2  = max(float(tau2),  1e-9)
    eta   = float(np.clip(eta, 0.0, 1.0))

    # Stable core + scaled erfc (erfcx)
    beta1 = ((x - mu)/sigma + (sigma/tau1))/np.sqrt(2.0)
    beta2 = ((x - mu)/sigma + (sigma/tau2))/np.sqrt(2.0)
    g     = np.exp(-((x - mu)**2)/(2.0*sigma**2))

    t1 = (1.0 - eta)/tau1 * g * erfcx(beta1)
    t2 = (       eta)/tau2 * g * erfcx(beta2)

    return 0.5 * A * (t1 + t2) + (b0 + b1*x)


class SkelFit:
    def __init__(self, param_1=1, param_2=2, param_3=10):
        # You can ignore these; kept so the factory signature matches the skeleton pattern.
        self.param_1 = param_1
        self.param_2 = param_2
        self.param_3 = param_3

    def start(self, x, y, xmin, xmax, fitpar, axis, fit_results):
        """
        Required by your framework.
        x, y         : arrays (already sliced by GUI)
        xmin, xmax   : numeric range (unused here; x,y are already in range)
        fitpar       : list of up to 8 GUI-provided initial numbers [A, mu, sigma, tau1, tau2, eta, b0, b1]
        axis         : Matplotlib axis to draw on
        fit_results  : QTextEdit to write the fit report
        Returns: a Line2D of the fitted curve (or None on failure)
        """
        # Keep only finite data
        m = np.isfinite(x) & np.isfinite(y)
        x = np.asarray(x)[m]
        y = np.asarray(y)[m]
        if x.size < 5:
            fit_results.setPlainText("Not enough data to fit.")
            return None

        # Unpack (tolerant to fewer entries)
        A0_ui, mu0_ui, sig0_ui, t10_ui, t20_ui, eta_ui, b0_ui, b1_ui = (
            list(fitpar) + [0]*(8-len(fitpar))
        )

        # Robust, data-driven initial guesses
        dx     = float(np.median(np.diff(x))) if x.size > 1 else 1.0
        span   = float(x.max() - x.min()) or 1.0
        mu0    = float(x[np.argmax(y)]) if np.all(np.isfinite(y)) else float((x.min()+x.max())/2.0)
        sigma0 = max(dx, span/50.0)
        tau10  = 0.5*sigma0
        tau20  = 1.5*sigma0
        A0     = float(np.trapz(np.clip(y,0,None), x))
        k      = max(3, y.size//20)
        b0_    = float(np.median(np.r_[y[:k], y[-k:]]))
        b1_    = 0.0

        # If user provided sane UI seeds, prefer them; else fall back to auto
        def pick(ui, auto): 
            return float(ui) if np.isfinite(ui) and ui != 0 else float(auto)

        pars = Parameters()
        pars.add('A',     value=max(pick(A0_ui, A0), 1.0), min=0)
        pars.add('mu',    value=pick(mu0_ui, mu0),  min=float(x.min()), max=float(x.max()))
        pars.add('sigma', value=max(pick(sig0_ui, sigma0), dx/4.0), min=dx/4.0, max=span)
        pars.add('tau1',  value=max(pick(t10_ui, tau10),  dx/10.0),  min=dx/10.0, max=10.0*sigma0)
        pars.add('tau2',  value=max(pick(t20_ui, tau20),  dx/10.0),  min=dx/10.0, max=20.0*sigma0)
        pars.add('eta',   value=np.clip(pick(eta_ui, 0.0), 0.0, 1.0), min=0.0, max=1.0)
        pars.add('b0',    value=pick(b0_ui, b0_))
        pars.add('b1',    value=pick(b1_ui, b1_))

        model = Model(_emg2, independent_vars=['x'])

        # Preflight: make sure the initial model is finite; if not, broaden widths
        ok = False
        for _ in range(5):
            y0 = model.eval(pars, x=x)
            if np.all(np.isfinite(y0)):
                ok = True
                break
            pars['sigma'].set(value=max(pars['sigma'].value*2.0, dx/2.0))
            pars['tau1'].set(value=max(pars['tau1'].value*2.0, dx/5.0))
            pars['tau2'].set(value=max(pars['tau2'].value*2.0, dx/5.0))
        if not ok:
            fit_results.setPlainText("Model non-finite at initial guess; aborting.")
            return None

        # First pass: simpler single-tail (eta fixed at 0) to stabilize
        eta_saved = pars['eta'].value
        pars['eta'].set(value=0.0, vary=False)
        res = model.fit(y, params=pars, x=x, method='least_squares')

        # Optional second pass: free eta (two tails) starting from result
        pars2 = res.params.copy()
        pars2['eta'].set(value=np.clip(0.3 if eta_saved == 0 else eta_saved, 0.0, 1.0),
                         min=0.0, max=1.0, vary=True)
        try:
            y0 = model.eval(pars2, x=x)
            if np.all(np.isfinite(y0)):
                res = model.fit(y, params=pars2, x=x, method='least_squares')
        except Exception:
            pass

        # Report
        try:
            fit_results.setPlainText(fit_report(res, show_correl=True))
        except Exception:
            fit_results.setPlainText(str(res))

        # Draw fitted curve
        xx = np.linspace(x.min(), x.max(), 2000)
        yy = model.eval(res.params, x=xx)
        (fitln,) = axis.plot(xx, yy, lw=2)
        # fitln.set_label("fit-_-")  # GUI will replace with an index label
        return fitln


class SkelFitBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, param_1=1, param_2=2, param_3=10, **_ignored):
        # Singleton pattern, mirroring your existing skeleton
        if not self._instance:
            self._instance = SkelFit(param_1, param_2, param_3)
        return self._instance
