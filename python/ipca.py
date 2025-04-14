import numpy as np
import pandas as pd
import progressbar

class IPCA():
    def __init__(self, n_factors=1, intercept=False):
        # parameter input validation
        if not isinstance(n_factors, int) or n_factors < 1:
            raise ValueError('n_factors must be an int greater / equal 1.')
        if not isinstance(intercept, bool):
            raise NotImplementedError('intercept must be  boolean')
        
        # save constructor parameters as object attributes
        params = locals()
        for k, v in params.items():
            if k != 'self':
                setattr(self, k, v)
    
    def fit(self, X, y, indices=None):
        """ 
        X : Matrix of characteristics with entity-time pair index.
        y : Dependent variable with entity-time index corresponding to X.
        """

        # prepare input by checking indices and cleaning data
        X, y, indices, metad = _prep_input(X, y, indices)
        N, L, T = metad["N"], metad["L"], metad["T"]
        
        # check that enough characteristics are provided for the requested number of factors
        if np.size(X, axis=1) < self.n_factors:
            raise ValueError('Number of factors requested (n_factors) cannot exceed number of features.')

        # store data
        self.X, self.y, self.indices = X, y, indices

        # build characteristics-weighted portfolio needed for the optimization of dynamic betas
        Q, W, val_obs = _build_portfolio(X, y, indices, metad)
        self.Q, self.W, self.val_obs = Q, W, val_obs
        self.metad = metad

        # run IPCA
        Gamma, Factors = self._fit_ipca(X=X, y=y, indices=indices, Q=Q,
                                        W=W, val_obs=val_obs)

########## helper functions ##########

def _prep_input(X, y=None, indices=None): 
    """
    Prepares different input types to consistent schema.
    """   
    # parameter input validation
    if X is None:
        raise ValueError('Must pass panel input data.')
    else:
        # remove panel rows containing missing observations
        non_nan_ind = ~np.any(np.isnan(X), axis=1)
        X = X[non_nan_ind]
        if y is not None:
            y = y[non_nan_ind]

    # check compatability of entity-time indices between X and y, break out indices from data
    if isinstance(X, pd.DataFrame) and not isinstance(y, pd.Series):
        indices = X.index
        chars = X.columns
        X = X.values
    elif not isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
        indices = y.index
        y = y.values
        chars = np.arange(X.shape[1])
    elif isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
        Xind = X.index
        chars = X.columns
        yind = y.index
        X = X.values
        y = y.values
        if not np.array_equal(Xind, yind):
            raise ValueError("If indices are provided with both X and y, they must be the same.")
        indices = Xind
    else:
        chars = np.arange(X.shape[1])

    if indices is None:
        raise ValueError("Entity-time indices must be provided either separately or as a MultiIndex with X/y")

    # extract numpy array and labels from multiindex
    if isinstance(indices, pd.MultiIndex):
        indices = indices.to_frame().values
    ids = np.unique(indices[:, 0])
    dates = np.unique(indices[:, 1])
    indices[:,0] = np.unique(indices[:,0], return_inverse=True)[1]
    indices[:,1] = np.unique(indices[:,1], return_inverse=True)[1]

    # init data dimensions
    T = np.size(dates, axis=0)
    N = np.size(ids, axis=0)
    L = np.size(chars, axis=0)

    # prep metadata
    metad = {}
    metad["dates"] = dates
    metad["ids"] = ids
    metad["chars"] = chars
    metad["T"] = T
    metad["N"] = N
    metad["L"] = L

    return X, y, indices, metad

def _build_portfolio(X, y, indices, metad):
    """
    Converts a stacked panel of data where each row corresponds to an
    observation (i, t) into a tensor of dimensions (N, L, T) where N is the
    number of unique entities, L is the number of characteristics and T is
    the number of unique dates.

    IPCA needs returns interacted with instruments (Q) to solve the optimization problem for dynamic betas.

    --- RETURNS ---
    Q : matrix of dimensions (L, T), containing the characteristics-weighted portfolios
    W : matrix of dimensions (L, L, T)
    val_obs : matrix of dimension (T), containting the number of non missing observations at each point in time
    """
    N, L, T = metad["N"], metad["L"], metad["T"]

    print(f"Panel dimensions:\n"
            f"  Number of unique entities (N): {N}\n"
            f"  Number of unique dates (T): {T}\n"
            f"  Number of characteristics used as instruments (L): {L}")
    
    # show progress
    bar = progressbar.ProgressBar(maxval=T,
                                  widgets=[progressbar.Bar('=', '[', ']'),
                                           ' ', progressbar.Percentage()])
    bar.start()

    # initialize portfolio outputs based on given dimensions
    W = np.full((L, L, T), np.nan)
    val_obs = np.full((T), np.nan)

    if y is not None:
        Q = np.full((L, T), np.nan)
        
        # for each time t
        for t in range(T):
            ixt = (indices[:, 1] == t) # select all observations
            val_obs[t] = np.sum(ixt) # store number of observations

            """
            Q : Each element Q_l_t represents a weighted average of returns at time t,
            for a portfolio whose weights are determined by the value of the assets' characteristic l,
            normalized by the number of non-missing observations of time t.
            If the first two characteristics l are e.g. value and capital,
            then the first rows Q_l are time series of returns managed on the basis of these.
            """
            Q[:, t] = X[ixt, :].T.dot(y[ixt])/val_obs[t]
            W[:, :, t] = X[ixt, :].T.dot(X[ixt, :])/val_obs[t]
            bar.update(t)
    # if dependent variable y is None, build the portfolio info for ind vars (?)
    else:
        Q = None
        for t in range(T):
            ixt = (indices[:, 1] == t)
            val_obs[t] = np.sum(ixt)
            W[:, :, t] = X[ixt, :].T.dot(X[ixt, :])/val_obs[t]
            bar.update(t)
    
    bar.finish()

    # return portfolio data
    return Q, W, val_obs

def _fit_ipca(self, X, y, indices, Q, W, val_obs):
    """
    Fits the regressor to the data using alternating least squares.

    --- RETURNS ---
    Gamma : array-like with dimensions (L, n_factors)
    Factors : array_like with dimensions (n_factors, T)
    """

    ALS_inputs = (Q, W, val_obs)
    ALS_fit = self._ALS_fit_portfolio
