import numpy as np
import pandas as pd

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

        print("Number of unique entities (N): ", N)
        print("Number of unique dates (T): ", T)
        print("Number of characteristics used as instruments (L): ", L)
        
        # check that enough characeteristics are provided for the requested number of factors
        if np.size(X, axis=1) < self.n_factors:
            raise ValueError('The number of factors requested (n_factors) exceeds number of features.')

########## helper functions ##########

def _prep_input(X, y=None, indices=None):    
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