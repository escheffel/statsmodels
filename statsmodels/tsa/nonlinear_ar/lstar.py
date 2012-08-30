from datetime import datetime

import numpy as np
from scipy import optimize
from scipy.stats import t, norm
from scipy.signal import lfilter
from numpy import (dot, identity, kron, log, zeros, pi, exp, eye, abs, empty,
                   zeros_like)
from numpy.linalg import inv, pinv

from statsmodels.tools.decorators import (cache_readonly,
        cache_writable, resettable_cache)
import statsmodels.base.model as base
import statsmodels.tsa.base.tsa_model as tsbase
import statsmodels.base.wrapper as wrap
from statsmodels.regression.linear_model import yule_walker, GLS
from statsmodels.tsa.tsatools import (lagmat, add_trend, add_lag,
        _ar_transparams, _ar_invtransparams, _ma_transparams,
        _ma_invtransparams)
from statsmodels.tsa.vector_ar import util
from statsmodels.tsa.ar_model import AR
from statsmodels.tools.numdiff import (approx_fprime, approx_fprime_cs,
        approx_hess_cs)
from statsmodels.tsa.base.datetools import _index_date



class LSTAR(tsbase.TimeSeriesModelNLS):
    
    __doc__ = tsbase._tsa_doc % {"model" : _lstar_model,
                    "params" : _lstar_params, "extra" : ""}

    def __init__(self, endog, thresh_data, thresh_delay=1, order=None,
                 variant='sep_reg', exog=None, dates=None, freq=None):
        super(TimeSeriesModelNLS, self).__init__(endog, exog, dates, freq)
        exog = self._data.exog # get it after it's gone through processing
        self._X = exog
        if order is None:
            import warnings
            warnings.warn("In the next release order will not be optional "
                    "in the model constructor.", FutureWarning)
        else:
            # order is a list of 2 lists, one for the upper and one for the lower regime
            self.k_upper = order[0]
            self.k_lower = order[1]
            # Get the maximum lag for the two separate regimes
            self.lag_upper = max(order[0])
            self.lag_lower = max(order[1])
            # Get the overall maximum lag for both regimes
            self.lag_both = max(self.lag_upper,self.lag_lower)
            # Attach more properties and get the max lag
            self.thresh_delay = thresh_delay
            self.max_lag = max(self.lag_both,self.thresh_delay)
        
        self.variant = variant
            
        if exog is not None:
            if exog.ndim == 1:
                exog = exog[:,None]
            k_exog = exog.shape[1]  # number of exog. variables excl. const
        else:
            k_exog = 0
        self.k_exog = k_exog
        
    def sumofsquares(self,params):
        '''
        This is the NonLinear model analogue to the Likelihood model's Likelihood function.
        Only here we want to minimize the residual-sum-of-squares function
        '''
        k_trend = self.k_trend
        k_upper = self.k_upper

        if self.variant == 'sep_reg':
            fitted_upper = np.dot(params[:k_trend+np.sum(k_upper)],X)
            
    def gen_two_X(self):
        '''
        Given the info supplied in variable `order`, generate two separate
        X data arrays for the two separate regimes.
        Careful: no constants are being added here yet.
        '''
        k_upper = self.k_upper
        k_lower = self.k_lower
        max_upper = self.lag_upper
        max_lower = self.lag_lower
        max_both = self.lag_both
        max_lag = self.max_lag
        XX = self._X
        upper_clag = k_upper[0]
        X_upper = add_lag(XX[:,0],lags=upper_clag)[(max_lag-upper_clag):,1:]
        lower_clag = k_lower[0]
        X_lower = add_lag(XX[:,0],lags=lower_clag)[(max_lag-lower_clag):,1:]
        for i1,lago in enumerate(k_upper[1:]):
            if lago > upper_clag: upper_clag = lago
            X_upper = np.hstack((X_upper,
                                 add_lag(XX[(max_lag-upper_clag):,i1+1],lags=upper_clag)[:,1:]))
            upper_clag = lago
        for i1,lago in enumerate(k_lower[1:]):
            if lago > lower_clag: lower_clag = lago
            X_lower = np.hstack((X_lower,
                                 add_lag(XX[(max_lag-lower_clag):,i1+1],lags=lower_clag)[:,1:]))
        self.X_upper = X_upper
        self.X_lower = X_lower
    
    def regime_weight(self,smooth_param=None,thresh_func=None,thresh_param=None,delay_param=1):
        #Use pure Python list here
        if type(thresh_var) != type([]):
            thresh_var = [x for x in thresh_var]
        thresh_var_d = thresh_var[delay_param:]
        thresh_std = np.std(thresh_var_d)
        lambdat = 1-1/(1+np.exp(-thresh_param*(thresh_var_d-thresh_param)/thresh_std))
        return lambdat
    
    def jacobian(self,params):
        pass
    
    def hessian(self,params):
        pass

    def _fit_start_params(self, order, method):
        '''
        Method call for getting initial starting values, may be based on simple linear AR models
        '''
        if method != 'css-mle': # use Hannan-Rissanen to get start params
            start_params = self._fit_start_params_hr(order)
        else: # use CSS to get start params
            func = lambda params: -self.loglike_css(params)
            #start_params = [.1]*(k_ar+k_ma+k_exog) # different one for k?
            start_params = self._fit_start_params_hr(order)
            if self.transparams:
                start_params = self._invtransparams(start_params)
            bounds = [(None,)*2]*sum(order)
            mlefit = optimize.fmin_l_bfgs_b(func, start_params,
                        approx_grad=True, m=12, pgtol=1e-7, factr=1e3,
                        bounds = bounds, iprint=-1)
            start_params = self._transparams(mlefit[0])
        return start_params
    
    def fit(self, order=None, start_params=None, trend='c', method = "css-mle",
            transparams=True, solver=None, maxiter=35, full_output=1,
            disp=5, callback=None, **kwargs):
        """
        Fits ARMA(p,q) model using exact maximum likelihood via Kalman filter.

        Parameters
        ----------
        start_params : array-like, optional
            Starting parameters for ARMA(p,q).  If None, the default is given
            by ARMA._fit_start_params.  See there for more information.
        transparams : bool, optional
            Whehter or not to transform the parameters to ensure stationarity.
            Uses the transformation suggested in Jones (1980).  If False,
            no checking for stationarity or invertibility is done.
        method : str {'css-mle','mle','css'}
            This is the loglikelihood to maximize.  If "css-mle", the
            conditional sum of squares likelihood is maximized and its values
            are used as starting values for the computation of the exact
            likelihood via the Kalman filter.  If "mle", the exact likelihood
            is maximized via the Kalman Filter.  If "css" the conditional sum
            of squares likelihood is maximized.  All three methods use
            `start_params` as starting parameters.  See above for more
            information.
        trend : str {'c','nc'}
            Whehter to include a constant or not.  'c' includes constant,
            'nc' no constant.
        solver : str or None, optional
            Solver to be used.  The default is 'l_bfgs' (limited memory Broyden-
            Fletcher-Goldfarb-Shanno).  Other choices are 'bfgs', 'newton'
            (Newton-Raphson), 'nm' (Nelder-Mead), 'cg' - (conjugate gradient),
            'ncg' (non-conjugate gradient), and 'powell'.
            The limited memory BFGS uses m=30 to approximate the Hessian,
            projected gradient tolerance of 1e-7 and factr = 1e3.  These
            cannot currently be changed for l_bfgs.  See notes for more
            information.
        maxiter : int, optional
            The maximum number of function evaluations. Default is 35.
        tol : float
            The convergence tolerance.  Default is 1e-08.
        full_output : bool, optional
            If True, all output from solver will be available in
            the Results object's mle_retvals attribute.  Output is dependent
            on the solver.  See Notes for more information.
        disp : bool, optional
            If True, convergence information is printed.  For the default
            l_bfgs_b solver, disp controls the frequency of the output during
            the iterations. disp < 0 means no output in this case.
        callback : function, optional
            Called after each iteration as callback(xk) where xk is the current
            parameter vector.
        kwargs
            See Notes for keyword arguments that can be passed to fit.

        Returns
        -------
        `statsmodels.tsa.arima.ARMAResults` class

        See also
        --------
        statsmodels.model.LikelihoodModel.fit for more information
        on using the solvers.

        Notes
        ------
        If fit by 'mle', it is assumed for the Kalman Filter that the initial
        unkown state is zero, and that the inital variance is
        P = dot(inv(identity(m**2)-kron(T,T)),dot(R,R.T).ravel('F')).reshape(r,
        r, order = 'F')

        The below is the docstring from
        `statsmodels.LikelihoodModel.fit`
        """
        if order is not None:
            import warnings
            warnings.warn("The order argument to fit is deprecated. "
                    "Please use the model constructor argument order. "
                    "This will overwrite any order given in the model "
                    "constructor.", FutureWarning)

            # get model order and constants
            self.k_ar = k_ar = int(order[0])
            self.k_ma = k_ma = int(order[1])
            self.k_lags = max(k_ar,k_ma+1)
        else:
            try:
                assert hasattr(self, "k_ar")
                assert hasattr(self, "k_ma")
            except:
                raise ValueError("Please give order to the model constructor "
                        "before calling fit.")
            k_ar = self.k_ar
            k_ma = self.k_ma

        # enforce invertibility
        self.transparams = transparams

        self.method = method.lower()

        endog, exog = self.endog, self.exog
        k_exog = self.k_exog
        self.nobs = len(endog) # this is overwritten if method is 'css'

        # (re)set trend and handle exogenous variables
        # always pass original exog
        k_trend, exog = _make_arma_exog(endog, self._data.exog, trend)

        self.k_trend = k_trend
        self.exog = exog    # overwrites original exog from __init__

        # (re)set names for this model
        self.exog_names = _make_arma_names(self._data, k_trend, (k_ar, k_ma))
        k = k_trend + k_exog


        # choose objective function
        method = method.lower()
        # adjust nobs for css
        if method == 'css':
            self.nobs = len(self.endog) - k_ar
        loglike = lambda params: -self.loglike(params)

        if start_params is not None:
            start_params = np.asarray(start_params)

        else: # estimate starting parameters
            start_params = self._fit_start_params((k_ar,k_ma,k), method)

        if transparams: # transform initial parameters to ensure invertibility
            start_params = self._invtransparams(start_params)

        if solver is None:  # use default limited memory bfgs
            bounds = [(None,)*2]*(k_ar+k_ma+k)
            mlefit = optimize.fmin_l_bfgs_b(loglike, start_params,
                    approx_grad=True, m=12, pgtol=1e-8, factr=1e2,
                    bounds=bounds, iprint=disp)
            self.mlefit = mlefit
            params = mlefit[0]

        else:   # call the solver from LikelihoodModel
            mlefit = super(ARMA, self).fit(start_params, method=solver,
                        maxiter=maxiter, full_output=full_output, disp=disp,
                        callback = callback, **kwargs)
            self.mlefit = mlefit
            params = mlefit.params

        if transparams: # transform parameters back
            params = self._transparams(params)

        self.transparams = False # set to false so methods don't expect transf.

        normalized_cov_params = None #TODO: fix this
        armafit = ARMAResults(self, params, normalized_cov_params)
        return ARMAResultsWrapper(armafit)
    
    def _stackX(self, k_ar, trend):
        """
        Private method to build the RHS matrix for estimation.

        Columns are trend terms then lags.
        """
        endog = self.endog
        X = lagmat(endog, maxlag=k_ar, trim='both')
        k_trend = util.get_trendorder(trend)
        if k_trend:
            X = add_trend(X, prepend=True, trend=trend)
        self.k_trend = k_trend
        return X

    fit.__doc__ += base.NonLinearLeastSquaresModel.fit.__doc__


if __name__ == "__main__":
    import numpy as np
    import statsmodels.api as sm