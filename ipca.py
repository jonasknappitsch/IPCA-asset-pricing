'''
IPCA algorithm and applications:
Kelly, Bryan T. and Pruitt, Seth and Su, Yinan, Characteristics Are Covariances:
A Unified Model of Risk and Return. JFE. (2018)
https://www.dropbox.com/scl/fo/309bktmb7pc6oihtn1cpe/ALKrhMV_GCh9HrDUuAfGaDI?dl=0&e=1&preview=IPCA_empirical_GB.m&rlkey=rg99gls2dr8q4got2bq00cdyx

Base python implementation:
Liz Chen at AQR Capital Management (2019)
https://www.dropbox.com/scl/fi/ecm4mlm1d27ka71gwrgsq/ipca_public.py?rlkey=40icghiic36vgns0n552zpzdh&e=2&dl=0

Modified and extended by:
Jonas Knappitsch at Vienna University of Economics and Business (2025)
'''

import time
import pandas as pd
import numpy as np
import scipy.linalg as sla # optimization
import scipy.sparse.linalg as ssla # optimization
from joblib import Parallel, delayed # parallelization
import matplotlib.pyplot as plt # visualization
import seaborn as sns # visualization

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
            # validates that there are no assets with missing returns or characteristics
            is_valid = pd.DataFrame({
                'z':self.Z[t].notnull().all(axis=1),
                'r':self.R[t].notnull()}).all(axis=1) 
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

    def run_ipca(self, fit=True, dispIters=False, parallel=True, MinTol=1e-6, MaxIter=5000):
        '''
        Computes Gamma, Fac and Lambd

        [Inputs]
        fit (bool): whether to compute fitted returns and r-squared after params are estimated
        dispIters (bool): whether to display results of each iteration
        parallel (bool): whether to use parallelized estimation (can reduce time per iteration)
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
        # initial guess starting from PCA SVD
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

        start_time = time.time()

        while iter < MaxIter and tol > MinTol:
            Gamma1, fFac1 = self._ipca_als_estimation(Gamma0,parallel)
            tol_Gamma = abs(Gamma1 - Gamma0).values.max()
            tol_fFac = abs(fFac1 - fFac0).values.max() if self.has_latent else None
            tol = max(x for x in [tol_Gamma, tol_fFac] if x is not None)

            if dispIters:
                print('iter {}: tolGamma = {} and tolFac = {}'.format(iter, tol_Gamma, tol_fFac))

            Gamma0, fFac0 = Gamma1, fFac1
            iter += 1

        end_time = time.time()

        print(f"ALS total time: {end_time - start_time:.2f} seconds")

        self.Gamma, self.fGamma, self.gGamma = Gamma1, Gamma1[self.fIdx], Gamma1[self.gIdx]
        
        if self.has_latent and self.has_prespec:
            self.Fac = pd.concat([fFac1, self.gFac])
        elif self.has_latent and not self.has_prespec:
            self.Fac = fFac1
        elif self.has_prespec and not self.has_latent:
            self.Fac = self.gFac
        else:
            raise ValueError("Both (latent) fFac and (pre-specified) gFac are missing.")
        self.fFac = fFac1

        self.Lambd, self.fLambd = self.Fac.mean(axis=1), self.fFac.mean(axis=1)

        if fit: # default to automatically compute fitted values
            self.fit()

    def _ipca_als_estimation(self, Gamma0, parallel):
        '''
        Runs one iteration of the alternating least squares estimation process

        [Inputs]
        Gamma0 (df(Lx(K+M))): previous iteration's Gamma estimate
        parallel (bool): whether to use parallelized estimation (can reduce time per iteration for larger datasets)

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

        if(parallel):
            # Parallelized Estimation
            # helper function for parallel execution of timestamps
            def compute_time_contribution(t):
                if self.has_latent and self.has_prespec:
                    Fac = pd.concat([fFac1[t], self.gFac[t]])
                elif self.has_latent and not self.has_prespec:
                    Fac = fFac1[t]
                elif self.has_prespec and not self.has_latent:
                    Fac = self.gFac[t]
                else:
                    raise ValueError("Both (latent) fFac and (pre-specified) gFac are missing at time t.")
                FacOutProd = np.outer(Fac, Fac)
                numer_t = np.kron(self.X[t], Fac) * self.N_valid[t]
                denom_t = np.kron(self.W[t], FacOutProd) * self.N_valid[t] # this line takes most of the time
                return numer_t, denom_t
            # n_jobs=-1 ensures all CPU cores available are used
            results = Parallel(n_jobs=-1, prefer="threads")(
                delayed(compute_time_contribution)(t) for t in self.times
            )
            for numer_t, denom_t in results:
                numer += numer_t
                denom += denom_t
        else:
            # Non-Parallelized Estimation
            for t in self.times:
                if self.has_latent and self.has_prespec:
                    Fac = pd.concat([fFac1[t], self.gFac[t]])
                elif self.has_latent and not self.has_prespec:
                    Fac = fFac1[t]
                elif self.has_prespec and not self.has_latent:
                    Fac = self.gFac[t]
                else:
                    raise ValueError("Both (latent) fFac and (pre-specified) gFac are missing at time t.")
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
                fFac1 += GammaBeta1.T.dot(GammaDelta1).dot(self.gFac) # (K x M reg coef) * gFac
                GammaDelta1 = (np.identity(self.L) - GammaBeta1.dot(GammaBeta1.T)).dot(GammaDelta1) # orthagonalize GammaDelta AFTER rotating F_New estimate
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

    ##### VISUALIZATIONS #####

    def visualize_factors(self, save_path=None):
        '''
        Plots time-series of all latent factor return estimates

        [Inputs]
        save_path (string): if given, stores the plot instead of showing it
        '''
        factors = self.Fac.T  # shape: T x K

        # plot time-series of latent factors
        factors.plot(figsize=(10, 6), title='IPCA Latent Factors')
        plt.xlabel("Time")
        plt.ylabel("Factor Value")
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()
    
    def visualize_gamma_heatmap(self, save_path=None):
        '''
        Plots heatmap of Gamma loadings of all latent factors

        [Inputs]
        save_path (string): if given, stores the plot instead of showing it
        '''
        gamma = self.Gamma

        # plot heatmap of gamma loadings
        plt.figure(figsize=(12, 20))
        ax = sns.heatmap(gamma, cmap="vlag", center=0, annot=True, fmt=".2f", cbar=True,linewidths=0.5)
        plt.title("IPCA Gamma Loadings")
        plt.xlabel("Latent Factors")
        plt.ylabel("Characteristics")

        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='right', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()

    def visualize_gamma_barplot(self, sorted=False, save_path=None):
        '''
        Plots barplot of Gamma loadings for each latent factor
        
        [Inputs]
        sorted (bool): whether to sort by loading values or not
        save_path (string): if given, stores the plot instead of showing it
        '''
        gamma = self.fGamma  # latent gamma only
        for i, factor_id in enumerate(gamma.columns):
            loadings = gamma[factor_id].sort_values() if sorted else gamma[factor_id]
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(loadings.index, loadings.values, color='gray')
            ax.set_title(f"Factor {i + 1}", fontsize=14)
            ax.axhline(0, color='black', linewidth=0.8)
            ax.set_xticks(range(len(loadings)))
            ax.set_xticklabels(loadings.index, rotation=90)
            ax.set_ylabel("Loading")
            ax.set_ylim(min(-0.8, loadings.min()*1.1), max(0.8, loadings.max()*1.1))
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300)
                plt.close()
            else:
                plt.show()

    ##### HYPOTHESIS TESTS #####
     
    def test_anomaly_alpha(self, B=1000):
        '''
        Tests whether loosening the alpha restriction (unrestricted model)
        improves the model fit as compared to the restricted model (alpha = 0).
        Null hypothesis states that characteristics are unassociated with alphas.
        Caveat: Computationally intensive due to bootstrapping B IPCAs

        [Inputs]
        Requires IPCA instance initialized with anomaly gFac

        [Outputs]
        Wald-type significance statistic
        p-value significance statistic
        '''
        assert self.has_latent and not self.has_prespec, "Test valid only in latent factor-only model."

        # extract estimated Gamma_alpha
        Gamma_alpha = self.gGamma.values
        W_stat = np.sum(Gamma_alpha ** 2)
        # compute residuals for unrestricted model
        d_hat = {t: self.R[t] - self.Z[t].dot(self.Gamma).dot(self.Fac[t]) for t in self.times}
        W_tilde_b = []

        for b in range(B):
            print("STARTING ", b+1, " out of ", B)
            # resample residuals
            d_b = {t: pd.Series(np.random.choice(d_hat[t].values, size=len(d_hat[t]), replace=True),
                        index=d_hat[t].index) for t in self.times}
            # bootstrap pseudo returns under null (alpha=0)
            pseudo_returns = {t: self.Z[t].dot(self.fGamma).dot(self.fFac[t]) + d_b[t] for t in self.times}
            # re-estimate IPCA (restricted: gGamma forced to 0)
            ipca_b = IPCA(Z=self.Z, R=pseudo_returns, K=self.K)
            ipca_b.run_ipca(fit=False)
            Gamma_alpha_b = ipca_b.gGamma.values
            W_tilde_b.append(np.sum(Gamma_alpha_b ** 2))
        p_val = np.mean(np.array(W_tilde_b) > W_stat)
        return {"W_stat": W_stat, "p_value": p_val}

    def test_observable_factors(self, B=1000):
        '''
        Tests explanatory power of observable factors beyond the baseline of IPCA factors
        '''
        
        # TODO
    
    def test_gamma_significance(self, B=1000):
        '''
        Tests the significance of individual characteristics' contribution while controlling
        for all other characteristics. Generates bootstrap samples under the null-hypothesis
        that characteristic l has no effect on loadings.
        Caveat: Highly computationally intensive due to dimensionality of bootstrapping IPCAs (L x B)

        [Inputs]
        B number of bootstrap samples per chara
        
        [Outputs]
        Wald-type significance statistic
        p-value significance statistic
        '''
        assert self.has_latent and not self.has_prespec, "Test valid only in latent factor-only model."
        
        results = []
        
        for l, char_name in enumerate(self.charas):
            # compute W_stat = gamma.T @ gamma
            gamma_vec = self.fGamma.iloc[l, :].values
            W_stat = gamma_vec @ gamma_vec
            # zero out the l-th row
            Gamma_tilde = self.fGamma.copy()
            Gamma_tilde.iloc[l, :] = 0
            # compute residuals under the unrestricted model
            d_hat = {t: self.X[t] - self.Z[t].dot(self.fGamma).dot(self.fFac[t]) for t in self.times}
            # bootstrap
            W_tilde_b = []
            for b in range(B):
                print("Starting bootstrap ", b+1, "/", B, " for chara ", l+1, "/",len(self.charas))
                d_b = {t: d_hat[t].sample(frac=1, replace=True) for t in self.times}
                X_b = {t: self.Z[t].dot(Gamma_tilde).dot(self.fFac[t]) + d_b[t] for t in self.times}
                # No need to convert X_b to DataFrame; pass as-is
                ipca_b = IPCA(Z=self.Z, R=self.R, X=X_b, K=self.K)
                ipca_b.run_ipca(fit=False)
                gamma_vec_b = ipca_b.fGamma.iloc[l, :].values
                W_tilde_b.append(gamma_vec_b @ gamma_vec_b)
            # compute p-value
            p_val = np.mean(np.array(W_tilde_b) > W_stat)
            results.append((char_name, W_stat, p_val))
        return pd.DataFrame(results, columns=["Characteristic", "W_stat", "p_value"]).set_index("Characteristic")

##### HELPER FUNCTIONS #####

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