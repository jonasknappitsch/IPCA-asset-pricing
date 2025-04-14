'''
IPCA algorithm and applications:
Kelly, Bryan T. and Pruitt, Seth and Su, Yinan, Characteristics Are Covariances: A Unified Model of Risk and Return. JFE Forthcoming. (2018)

Original Python implementation:
Liz Chen at AQR Capital Management (2019)

Modified and extended by:
Jonas Knappitsch at Vienna University of Economics and Business (2025)
'''
import pandas as pd
import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as ssla

# datasets
from statsmodels.datasets import grunfeld
import wrds
import openassetpricing as oap

# visualization
import matplotlib.pyplot as plt

class IPCA(object):
    def __init__(self, Z, R=None, X=None, K=0, gFac=None):
        '''
        [Dimensions]
            N: the number of assets
            T: the number of time periods
            L: the number of characteristics
            K: the number of latent factors
            M: the number of pre-specified factors (plus anomaly term)

        [Inputs]
            Z (dict(T) of df(NxL)): characteristics; can be rank-demeaned
            R (dict(T) of srs(N); not needed for managed-ptf-only version): asset returns
            X (df(LxT); only needed for managed-ptf-only version): managed portfolio returns
            K (const; optional): number of latent factors
            gFac (df(MxT); optional): Anomaly term ([1,...,1]), or Pre-Specified Factors (i.e. returns of HML, SMB, etc.)

        * IPCA can be run with only K > 0 or only gFac
        * IMPORTANT: this structure and the code supposes that lagging has already occurred.
          i.e. If t is March 2003, monthly data, then R[t] are the returnss realized at the end of March 2003 during March 2003,
          and Z[t] are the characteristics known at the end of February 2003.

        [Transformed Inputs]
            N_valid (srs(T)): number of nonmissing obs each period, where a missing obs is any asset with missing return or any missing characteristic
            X (df(LxT)): managed portfolio returns: X[t] = Z[t][valid].T * R[t][valid] / N_valid[t]
            W (dict(T) of df(LxL)): characteristic second moments: W[t] = Z[t][valid].T * Z[t][valid].T / N_valid(t)

        [Outputs]
        calculated in run_ipca method:
            Gamma (df(Lx(K+M))): gamma estimate (fGamma for latent, gGamma for pre-specified)
            Fac (df((K+M)xT)): factor return estimate (fFac for latent, gFac for pre-specified)
            Lambd (srs(K+M)): mean of Fac (fLambd for latent, gLambd for pre-specified)
        calculated in fit method:
            fitvals (dict(4) of dict(T) of srs(N)): fitted values of asset returns; 4 versions: {constant risk price, dynamic risk price} x {assets, managed-ptfs}
            r2 (srs(4)): r-squared of the four versions of fitted values against actual values
        '''
        # type of model
        self.X_only = True if R is None else False # managed-ptf-only version
        self.has_latent = True if K else False
        self.has_prespec = True if (gFac is not None and len(gFac) > 0) else False

        # inputs
        self.Z, self.R, self.X = Z, R, X
        self.times, self.charas = sorted(Z.keys()), Z[list(Z.keys())[0]].columns
        self.gFac = gFac if self.has_prespec else pd.DataFrame(columns=self.times)
        self.gLambd = self.gFac.mean(axis=1)
        self.fIdx, self.gIdx = list(map(str, range(1, K+1))), list(self.gFac.index)
        self.K, self.M, self.L, self.T = K, len(self.gIdx), len(self.charas), len(self.times)

        # transformation inputs
        self.N_valid = pd.Series(index=self.times)
        if not self.X_only:
            self.X = pd.DataFrame(index=self.charas, columns=self.times)
        self.W = {t: pd.DataFrame(index=self.charas, columns=self.charas) for t in self.times}
        for t in self.times:
            is_valid = pd.DataFrame({'z':self.Z[t].notnull().all(axis=1),'r':self.R[t].notnull()}).all(axis=1) # not valid if ret or any charas are missing
            z_valid = self.Z[t].loc[is_valid.values,:]
            r_valid = self.R[t].loc[is_valid.values]
            self.N_valid[t] = (1. * is_valid).sum()
            if not self.X_only:
                self.X[t] = z_valid.T.dot(r_valid) / self.N_valid[t]
            self.W[t] = z_valid.T.dot(z_valid) / self.N_valid[t]

        # outputs
        self.Gamma, self.fGamma, self.gGamma = None, None, None
        self.Fac, self.fFac = None, None
        self.Lambd, self.fLambd = None, None
        self.fitvals, self.r2 = {}, pd.Series()

    def run_ipca(self, fit=True, dispIters=False, MinTol=1e-6, MaxIter=5000):
        '''
        Computes Gamma, Fac and Lambd

        [Inputs]
        fit (bool): whether to compute fitted returns and r-squared after params are estimated
        dispIters (bool): whether to display results of each iteration
        MinTol (float): tolerance for convergence
        MaxIter (int): max number of iterations

        [Outputs]
        Gamma (df(Lx(K+M))): gamma estimate (fGamma for latent, gGamma for pre-specified)
        Fac (df((K+M)xT)): factor return estimate (fFac for latent, gFac for pre-specified)
        Lambd (srs(K+M)): mean of Fac (fLambd for latent, gLambd for pre-specified)

        * When characteristics are rank-demeaned and returns are used in units (ie 0.01 is a 1% return),
          1e-6 tends to be a good convergence criterion.
          This is because the convergence of the algorithm mostly comes from GammaBeta being stable,
          and 1e-6 is small given that GammaBeta is always rotated to be orthonormal.
        '''
        # initial guess
        Gamma0 = GammaDelta0 = pd.DataFrame(0., index=self.charas, columns=self.gIdx)
        if self.has_latent:
            svU, svS, svV = ssla.svds(self.X.values, self.K)
            svU, svS, svV = np.fliplr(svU), svS[::-1], np.flipud(svV) # reverse order to match MATLAB svds output
            fFac0 = pd.DataFrame(np.diag(svS).dot(svV), index=self.fIdx, columns=self.times) # first K PC of X
            GammaBeta0 = pd.DataFrame(svU, index=self.charas, columns=self.fIdx) # first K eigvec of X
            GammaBeta0, fFac0 = _sign_convention(GammaBeta0, fFac0)
            Gamma0 = pd.concat([GammaBeta0, GammaDelta0], axis=1)

        # ALS estimate
        tol, iter = float('inf'), 0
        while iter < MaxIter and tol > MinTol:
            Gamma1, fFac1 = self._ipca_als_estimation(Gamma0)
            tol_Gamma = abs(Gamma1 - Gamma0).values.max()
            tol_fFac = abs(fFac1 - fFac0).values.max() if self.has_latent else None
            tol = max(tol_Gamma, tol_fFac)

            if dispIters:
                print('iter {}: tolGamma = {} and tolFac = {}'.format(iter, tol_Gamma, tol_fFac))

            Gamma0, fFac0 = Gamma1, fFac1
            iter += 1

        self.Gamma, self.fGamma, self.gGamma = Gamma1, Gamma1[self.fIdx], Gamma1[self.gIdx]
        if self.has_prespec:
            self.Fac = pd.concat([fFac1, self.gFac])
        else:
            self.Fac = fFac1
        self.fFac = fFac1
        self.Lambd, self.fLambd = self.Fac.mean(axis=1), self.fFac.mean(axis=1)

        if fit: # default to automatically compute fitted values
            self.fit()

    def _ipca_als_estimation(self, Gamma0):
        '''
        Runs one iteration of the alternating least squares estimation process

        [Inputs]
        Gamma0 (df(Lx(K+M))): previous iteration's Gamma estimate

        [Outputs]
        Gamma1 (df(Lx(K+M))): current iteration's Gamma estimate
        fFac1 (df(KxT)): current iteration's latent Factor estimate

        * Imposes identification assumption on Gamma1 and fFac1:
          Gamma1 is orthonormal matrix and fFac1 orthogonal with positive mean (taken across times)

        '''
        # 1. estimate latent factor
        fFac1 = pd.DataFrame(index=self.fIdx, columns=self.times)
        if self.has_latent:
            GammaBeta0, GammaDelta0 = Gamma0[self.fIdx], Gamma0[self.gIdx]
            for t in self.times:
                numer = GammaBeta0.T.dot(self.X[t])
                if self.has_prespec:
                    numer -= GammaBeta0.T.dot(self.W[t]).dot(GammaDelta0).dot(self.gFac[t])
                denom = GammaBeta0.T.dot(self.W[t]).dot(GammaBeta0)
                fFac1[t] = pd.Series(_mldivide(denom, numer), index=self.fIdx)

        # 2. estimate gamma
        vec_len = self.L * (self.K + self.M)
        numer, denom = np.zeros(vec_len), np.zeros((vec_len, vec_len))
        for t in self.times:
            if self.has_prespec:
                Fac = pd.concat([fFac1[t], self.gFac[t]])
            else:
                Fac = fFac1[t]
            FacOutProd = np.outer(Fac, Fac)
            numer += np.kron(self.X[t], Fac) * self.N_valid[t]
            denom += np.kron(self.W[t], FacOutProd) * self.N_valid[t] # this line takes most of the time
        Gamma1_tmp = np.reshape(_mldivide(denom, numer), (self.L, self.K + self.M))
        Gamma1 = pd.DataFrame(Gamma1_tmp, index=self.charas, columns=self.fIdx + self.gIdx)

        # 3. identification assumption
        if self.has_latent: # GammaBeta orthonormal and fFac1 orthogonal
            GammaBeta1, GammaDelta1 = Gamma1[self.fIdx], Gamma1[self.gIdx]

            R1 = sla.cholesky(GammaBeta1.T.dot(GammaBeta1))
            R2, _, _ = sla.svd(R1.dot(fFac1).dot(fFac1.T).dot(R1.T))
            GammaBeta1 = pd.DataFrame(_mrdivide(GammaBeta1, R1).dot(R2), index=self.charas, columns=self.fIdx)
            fFac1 = pd.DataFrame(_mldivide(R2, R1.dot(fFac1)), index=self.fIdx, columns=self.times)
            GammaBeta1, fFac1 = _sign_convention(GammaBeta1, fFac1)

            if self.has_prespec: # orthogonality between GammaBeta and GammaDelta
                GammaDelta1 = (np.identity(self.L) - GammaBeta1.dot(GammaBeta1.T)).dot(GammaDelta1)
                fFac1 += GammaBeta1.T.dot(GammaDelta1).dot(self.gFac) # (K x M reg coef) * gFac
                GammaBeta1, fFac1 = _sign_convention(GammaBeta1, fFac1)

            Gamma1 = pd.concat([GammaBeta1, GammaDelta1], axis=1)
        return Gamma1, fFac1

    def fit(self):
        '''
        Computes fitted values and their associated r-squared

        [Inputs]
        Assumes the run_ipca was already run

        [Outputs]
        fitvals (dict(4) of dict(T) of srs(N)): fitted values of asset returns; 4 versions: (constant vs dynamic risk prices) x (assets vs managed-ptfs)
        r2 (srs(4)): r-squared of the four versions of fitted values against actual values

        * Dynamic Risk Price -> F
          Constant Risk Price -> Lambda
        '''
        if not self.X_only:
            self.fitvals['R_DRP'] = {t: self.Z[t].dot(self.Gamma).dot(self.Fac[t]) for t in self.times}
            self.fitvals['R_CRP'] = {t: self.Z[t].dot(self.Gamma).dot(self.Lambd) for t in self.times}
            self.r2['R_Tot'] = _calc_r2(self.R, self.fitvals['R_DRP'])
            self.r2['R_Prd'] = _calc_r2(self.R, self.fitvals['R_CRP'])

        self.fitvals['X_DRP'] = {t: self.W[t].dot(self.Gamma).dot(self.Fac[t]) for t in self.times}
        self.fitvals['X_CRP'] = {t: self.W[t].dot(self.Gamma).dot(self.Lambd) for t in self.times}
        self.r2['X_Tot'] = _calc_r2(self.X, self.fitvals['X_DRP'])
        self.r2['X_Prd'] = _calc_r2(self.X, self.fitvals['X_CRP'])

    def visualize_factors(self):
        factors = self.Fac.T  # shape: T x K

        # plot time-series of latent factors
        factors.plot(figsize=(10, 4), title='Estimated IPCA Latent Factors')
        plt.xlabel("Time")
        plt.ylabel("Factor Value")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

########## HELPER FUNCTIONS ##########

# matrix left/right division (following MATLAB function naming)
_mldivide = lambda denom, numer: sla.lstsq(np.array(denom), np.array(numer))[0]
_mrdivide = lambda numer, denom: (sla.lstsq(np.array(denom).T, np.array(numer).T)[0]).T

def _sign_convention(gamma, fac):
    '''
    sign the latent factors to have positive mean, and sign gamma accordingly
    '''
    sign_conv = fac.mean(axis=1).apply(lambda x: 1 if x >= 0 else -1)
    return gamma.mul(sign_conv.values, axis=1), fac.mul(sign_conv.values, axis=0)

def _calc_r2(r_act, r_fit):
    '''
    compute r2 of fitted values vs actual
    '''
    sumsq = lambda x: x.dot(x)
    sse = sum(sumsq(r_act[t] - r_fit[t]) for t in r_fit.keys())
    sst = sum(sumsq(r_act[t]) for t in r_fit.keys())
    return 1. - sse / sst

def download_data(dataset="grunfeld"):
    if(dataset == "grunfeld"):
        data = grunfeld.load_pandas().data

        ########## preprocessing ##########

        # convert date
        data.year = data.year.astype(np.int64)

        # establish unique IDs
        N = len(np.unique(data.firm))
        ID = dict(zip(np.unique(data.firm).tolist(), np.arange(1, N+1)+5))
        data.firm = data.firm.apply(lambda x: ID[x])

        # rearrange ordering
        data = data[['firm', 'year', 'invest', 'value', 'capital']]

        # prepare pre-specified factors test vars
        PSF1 = np.random.randn(len(np.unique(data.loc[:, 'year'])), 1)
        PSF1 = PSF1.reshape((1, -1))
        PSF2 = np.random.randn(len(np.unique(data.loc[:, 'year'])), 2)
        PSF2 = PSF2.reshape((2, -1))

        # set entity-time index and prepare independent variables (value, capital) and dependent variable (invest, analogous to return)
        data = data.set_index(['firm', 'year'])
        X = data.drop('invest', axis=1)
        y = data['invest']
        
        # lag x by 1 period as required:
        X_lagged = X.groupby('firm').shift(1).dropna()
        y_aligned = y.loc[X_lagged.index]

        # convert to time-indexed dict(T) as required:
        # Z (dict(T) of df(NxL)): characteristics; can be rank-demeaned
        # R (dict(T) of srs(N); not needed for managed-ptf-only version): asset returns
        Z = {t: df.droplevel('year') for t, df in X.groupby('year')}
        R = {t: s.droplevel('year') for t, s in y.groupby('year')}
    elif(dataset == "openassetpricing"):

        # https://github.com/mk0417/open-asset-pricing-download/blob/master/examples/ML_portfolio_example.ipynb
        openap = oap.OpenAP(202408)

        # requires WRDS authentication
        wrds_conn = wrds.Connection()

        crsp = wrds_conn.raw_sql(
            """select a.permno, a.date, a.ret*100 as ret
                                from crsp.msf a
                                join crsp.msenames b 
                                on a.permno = b.permno
                                and a.date >= b.namedt
                                and a.date <= b.nameendt
                                where b.shrcd in (10, 11, 12) 
                                and b.exchcd in (1, 2, 3)""",
            date_cols=["date"],
        )

        firm_characteristics = openap.dl_all_signals('pandas') 
        
    elif(dataset == "crsp"):
        raise NotImplementedError('Dataset not yet supported.')
    else:
        raise NotImplementedError('No valid dataset selected.')
    return(Z, R)