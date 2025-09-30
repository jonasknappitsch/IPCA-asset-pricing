import os
import numpy as np
import pandas as pd
import requests
import logging
from ipca import IPCA
import wrds # datasets
import matplotlib.pyplot as plt # visualization
import seaborn as sns # visualization

######################################
#####       PREREQUISITES        #####
######################################

# 1. WRDS Connection for Return and Factor Data
wrds_conn = wrds.Connection()

# 2. Stock Dataset, provided at data/fnw/characteristics_data_feb2017.csv"
"""
provided by https://sethpruitt.net/2019/12/01/characteristics-are-covariances/
download link http://dropbox.com/scl/fo/309bktmb7pc6oihtn1cpe/ANxVbKAJ2J5VN0HyhkusWSo/characteristics_data_feb2017.csv?rlkey=rg99gls2dr8q4got2bq00cdyx&e=1&dl=1
"""

# 3. Bond Dataset, provided at data/kpbonds/corp_jkp_mergedv2.csv"
"""
provided by https://sethpruitt.net/2022/03/29/reconciling-trace-bond-returns/
download link https://www.dropbox.com/scl/fo/0na6wktek6ydredni1697/APpJBFtYGIrQt_rds6vbk1w?e=2&preview=corp_jkp_mergedv2.csv&rlkey=j7kyimwnmzs62pzlsv0hus8av&dl=0
"""

######################################
##### IPCA Application to Stocks #####
######################################

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.info("##### IPCA Application to Stocks #####")

##### Download Data #####

"""
Dataset Freyberger et al. (2017) "Dissecting Characteristics Nonparametrically"
as used in Kelly et al. (2019) "Characteristics are Covariances"
provided by https://sethpruitt.net/2019/12/01/characteristics-are-covariances/
download link http://dropbox.com/scl/fo/309bktmb7pc6oihtn1cpe/ANxVbKAJ2J5VN0HyhkusWSo/characteristics_data_feb2017.csv?rlkey=rg99gls2dr8q4got2bq00cdyx&e=1&dl=1
"""

# download dataset if not exists
dataset="fnw"
if not os.path.isfile(f'data/{dataset}/characteristics_data_feb2017.csv'):
    print("Dataset not found locally. Downloading dataset (approx. 750 MB) ...")
    source = "http://dropbox.com/scl/fo/309bktmb7pc6oihtn1cpe/ANxVbKAJ2J5VN0HyhkusWSo/characteristics_data_feb2017.csv?rlkey=rg99gls2dr8q4got2bq00cdyx&e=1&dl=1"
    response = requests.get(source)
    response.raise_for_status()  # raise error if download fails
    with open(f'data/{dataset}/characteristics_data_feb2017.csv', "wb") as f:
        f.write(response.content)
    print("Download complete.")

signals = pd.read_csv(f'data/{dataset}/characteristics_data_feb2017.csv', delimiter=',')

# drop metadata and non-needed columns
signals = signals.drop(columns=['Unnamed: 0', 'yy', 'mm','q10', 'q20', 'q50', 'prc'])

# set date from end of month to 28th to allow join with rf rate
signals['date'] = pd.to_datetime(signals['date'])
signals['date'] = signals['date'].apply(lambda d: d.replace(day=28)) 

# download Fama-French risk-free rate to compute excess returns
rf = wrds_conn.raw_sql("""
                SELECT date, rf
                FROM ff.factors_monthly
                """, date_cols=["date"])

# change ff date from 1st day of month to 28th to allow join
rf['date'] = rf['date'].apply(lambda d: d.replace(day=28))

# merge crsp returns with rf and compute excess returns
data = signals.merge(rf, on='date', how='left')
data['excess_ret'] = data['ret'] - data['rf']

# rename variables based on Freyberger Neuhierl Weber (2017) to adhere to naming of Kelly Pruitt Su (2019)
rename_map = {
    "at": "assets",
    "beme": "bm",
    "free_cf": "freecf",
    "idio_vol": "idiovol",
    "lme": "mktcap",
    "lturnover": "turn",
    "rel_to_high_price": "w52h",
    "cum_return_12_2": "mom",
    "cum_return_12_7": "intmom",
    "cum_return_1_0": "strev",
    "cum_return_36_13": "ltrev",
    "sga2m": "sga2s",
    "spread_mean": "bidask"
}
data = data.rename(columns=rename_map)

# set entity-time multi-index
data = data.set_index(['permno', 'date'])

# retrieve signal names
non_signal_cols = ['excess_ret','ret','rf']
signal_names = [col for col in data.columns if col not in non_signal_cols]

##### Preprocess Data #####

# 1. filter by date
start_year = 1962
end_year = 2014

processed_data = data[
    (data.index.get_level_values('date').year >= start_year) &
    (data.index.get_level_values('date').year <= end_year)]

# 2. remove rows where return is null
processed_data = processed_data[processed_data['excess_ret'].notnull()]

# 3. remove rows where all signals are missing (e.g. due to lagging)
processed_data = processed_data.dropna(subset=signal_names, how='all')

# 4. standardize by performing rank-normalization among non-missing observations
for col in signal_names:
        processed_data[col] = processed_data.groupby(level='date')[col].transform(
            lambda x: ((x.rank(method='average', na_option='keep') - 1) / (x.count() - 1)) - 0.5
        )

# 5. impute missing values with median, which equals 0 after standardization
processed_data[signal_names] = processed_data[signal_names].fillna(0)

# 6. add constant
processed_data["const"] = 1.0
signal_names.append("const")

##### Estimate and Evaluate IPCA Model #####

# construct Z and R as required by ipca class (convert pd.Float64 to np.float32, drop date from index)
Z = {t: df[signal_names].astype(np.float32).droplevel("date") for t, df in processed_data.groupby("date")}
R = {t: s["excess_ret"].astype(np.float32).droplevel("date") for t, s in processed_data.groupby("date")}

# IPCA K=5 Without Anomalies
K = 5
model_stocks = IPCA(Z, R=R, K=K)
model_stocks.run_ipca(dispIters=True)

print("R2: ", model_stocks.r2)
model_stocks.visualize_factors()
model_stocks.visualize_gamma_heatmap()
print("Factor Expected Returns (monthly) \n: ",model_stocks.Fac.T.mean())
print("Factor Sharpe Ratios (annualized): \n", (model_stocks.Fac.T.mean()/model_stocks.Fac.T.std()) * (12**0.5))

# IPCA K=5 With Anomalies
K = 5
gFac = pd.DataFrame(1., index=sorted(R.keys()), columns=['anomaly']).T
model_stocks_unrestricted = IPCA(Z, R=R, K=K, gFac=gFac)
model_stocks_unrestricted.run_ipca(dispIters=True)

print("R2: ", model_stocks_unrestricted.r2)
model_stocks_unrestricted.visualize_factors()
model_stocks_unrestricted.visualize_gamma_heatmap()
print("Factor Expected Returns (monthly) \n: ",model_stocks_unrestricted.Fac.T.mean())
print("Factor Sharpe Ratios (annualized): \n", (model_stocks_unrestricted.Fac.T.mean()/model_stocks_unrestricted.Fac.T.std()) * (12**0.5))

##### Instrumenting Traditional Factor Models With IPCA Pre-Specified Factors (PSF) #####

# change start date as Fama-French Five Factors exist only from 1964, change Z and R correspondingly
start_year = 1964 
end_year = 2014

data_new = processed_data[
(data.index.get_level_values('date').year >= start_year) &
(data.index.get_level_values('date').year <= end_year)]

Z_new = {t: df[signal_names].astype(np.float32).droplevel("date") for t, df in data_new.groupby("date")}
R_new = {t: s["excess_ret"].astype(np.float32).droplevel("date") for t, s in data_new.groupby("date")}

gFac_Ks = [1,3,4,5,6] # define which FF Factor Models to use (cf. Thesis)
IPCAs = [] # stores resulting models for gFac_Ks

def load_observable_factors(gFac_K,R):
    """
    Downloads observable factors from WRDS Fama-French library for various K specifications.
    Returns gFac (df(M x T)) for given observable factor specification gFac_K.
    
    gFac_K: int, number of factors (1=CAPM, 3=FF3, 4=FFC4, 5=FF5, 6=FFC6)
    R: dictionary of excess returns with date keys (used to align gFac time index)
    """
    # download full factor set from Fama-French
    ff_all = wrds_conn.raw_sql("""
        SELECT date, mktrf, smb, hml, umd, rmw, cma
        FROM ff.fivefactors_monthly
    """, date_cols=["date"])
    # change day to 28th to allow merge with returns
    ff_all['date'] = ff_all['date'].apply(lambda d: d.replace(day=28))
    ff_all.set_index('date', inplace=True)
    ff_all = ff_all.reindex(sorted(R.keys()))  # match with excess returns R index
    # prepare columns for each pre-specified factor
    if gFac_K == 1:
        cols = ['mktrf'] # CAPM
    elif gFac_K == 3:
        cols = ['mktrf', 'smb', 'hml']  # FF3 (CAPM + smb + hml)
    elif gFac_K == 4:
        cols = ['mktrf', 'smb', 'hml', 'umd']  # FFC4 (FF3 + momentum "umd")
    elif gFac_K == 5:
        cols = ['mktrf', 'smb', 'hml', 'rmw', 'cma']  # FF5 (FF3 + rmw + cma)
    elif gFac_K == 6:
        cols = ['mktrf', 'smb', 'hml', 'rmw', 'cma', 'umd']  # FFC6 (FF5 + momentum "umd")
    else:
        raise ValueError(f"Unsupported gFac_K={gFac_K}. Choose from [1,3,4,5,6].")
    gFac = ff_all[cols].T
    gFac.index = cols  # ensures proper row names
    print("gFac: ", gFac)
    assert not gFac.isnull().values.any(), "gFac contains NaNs. Check date alignment with R."
    return gFac

for gFac_K in gFac_Ks:
    gFac = load_observable_factors(gFac_K, R_new)
    model = IPCA(Z_new, R=R_new, gFac=gFac)
    model.run_ipca(dispIters=True)
    IPCAs.append(model)

# evaluate instrumented FFC6 model results
print("R2: ", IPCAs[4].r2)
IPCAs[4].visualize_factors()
IPCAs[4].visualize_gamma_heatmap()

######################################
##### IPCA Application to Bonds #####
######################################

logging.info("##### IPCA Application to Bonds #####")

##### Download Data #####

"""
Dataset Kelly and Pruitt (2022) "Reconciling TRACE bond returns"
provided by https://sethpruitt.net/2022/03/29/reconciling-trace-bond-returns/
download link https://www.dropbox.com/scl/fo/0na6wktek6ydredni1697/APpJBFtYGIrQt_rds6vbk1w?e=2&preview=corp_jkp_mergedv2.csv&rlkey=j7kyimwnmzs62pzlsv0hus8av&dl=0
"""

# download dataset if not exists
dataset="kpbonds"
if not os.path.isfile(f'data/{dataset}/corp_jkp_mergedv2.csv'):
    print("Dataset not found locally. Downloading dataset (approx. 350 MB) ...")
    source = "https://www.dropbox.com/scl/fi/np6omyvdct7ezt288qgjv/corp_jkp_mergedv2.csv?rlkey=g0429wg9r1dap5u33z3w2326y&st=g7i6ec7h&dl=1"
    response = requests.get(source)
    response.raise_for_status()  # raise error if download fails
    with open(f'data/{dataset}/corp_jkp_mergedv2.csv', "wb") as f:
        f.write(response.content)
    print("Download complete.")

signals = pd.read_csv(f'data/{dataset}/corp_jkp_mergedv2.csv', delimiter=',')

# set correct dtypes
signals["dates"] = pd.to_datetime(signals["dates"].astype(str) + "28", format="%Y%m%d")

# keep only observations from August 2003 onwards as per Kelly and Pruitt (2022) "Reconciling TRACE bond returns"
signals = signals[signals["dates"] >= "2003-08-01"]

# rename variables for consistency
rename_map = {
    "cusip": "permno", # use cusip as permno-column for model consistency; permno is then used as unique asset identifier (as for stocks)
    "dates": "date",
    "nextretexc": "excess_ret" # use nextretexc to predict excess_ret at t+1 by signals at t (lagging)
}
signals = signals.rename(columns=rename_map)

# retrieve signal names as per Kelly et al. (2023) "Modeling Corporate Bond Returns"
signal_names = [
    "age",              # 1. Bond age
    "coupon",           # 2. Coupon
    "amtout",           # 3. Face value
    "be_me",            # 4. Book-to-price
    "debt_ebitda",      # 5. Debt-to-EBITDA
    "duration",         # 6. Duration
    "ret_6_1",          # 7. Momentum 6m equity
    "ni_me",            # 8. Earnings-to-price
    "me",               # 9. Equity market cap
    "rvol_21d",         # 10. Equity volatility
    "totaldebt",        # 11. Firm total debt
    "mom6",             # 12. Momentum 6m bond
    "ret_6_1_ind",      # 13. Industry momentum (proxy via industry return)
    "mom6xrtg",         # 14. Momentum × ratings
    "at_be",            # 15. Book leverage
    "market_lev",       # 16. Market leverage
    "turn_vol",         # 17. Turnover volatility
    "spread",           # 18. Spread
    "oper_lvg",         # 19. Operating leverage
    "gp_at",            # 20. Profitability
    "chg_gp_at",        # 21. Profitability change
    "rtg",              # 22. Rating
    "D2D",              # 23. Distance-to-default
    "skew",             # 24. Bond skewness
    "mom6mspread",      # 25. Momentum 6m log(Spread)
    "spr_to_d2d",       # 26. Spread-to-D2D
    "volatility",       # 27. Bond volatility
    "VaR",              # 28. Value-at-Risk
    "vixbeta"           # 29. VIX beta
]

# drop metadata and non-needed columns, keeping only signal names and excess returns
data = signals[[col for col in signals.columns if col in signal_names or col in ["permno","date","excess_ret"]]]

# set entity-time multi-index
data = data.set_index(['permno', 'date'])

##### Preprocess Data #####

# 1. filter by date
start_year = 2003
end_year = 2020

processed_data = data[
    (data.index.get_level_values('date').year >= start_year) &
    (data.index.get_level_values('date').year <= end_year)]

# 2. remove rows where return is null
processed_data = processed_data[processed_data['excess_ret'].notnull()]

# 3. remove rows where all signals are missing (e.g. due to lagging)
processed_data = processed_data.dropna(subset=signal_names, how='all')

# 4. standardize by performing rank-normalization among non-missing observations
for col in signal_names:
        processed_data[col] = processed_data.groupby(level='date')[col].transform(
            lambda x: ((x.rank(method='average', na_option='keep') - 1) / (x.count() - 1)) - 0.5
        )

# 5. impute missing values with median, which equals 0 after standardization
processed_data[signal_names] = processed_data[signal_names].fillna(0)

# 6. add constant
processed_data["const"] = 1.0
signal_names.append("const")

##### Estimate and Analyze IPCA Model #####

# construct Z and R as required by ipca class (convert pd.Float64 to np.float32, drop date from index)
Z = {t: df[signal_names].astype(np.float32).droplevel("date") for t, df in processed_data.groupby("date")}
R = {t: s["excess_ret"].astype(np.float32).droplevel("date") for t, s in processed_data.groupby("date")}

# IPCA K=5 Without Anomalies
K = 5
model_bonds = IPCA(Z, R=R, K=K)
model_bonds.run_ipca(dispIters=True)

print("R2: ", model_bonds.r2)
model_bonds.visualize_factors()
model_bonds.visualize_gamma_heatmap()
print("Factor Expected Returns (monthly) \n: ",model_bonds.Fac.T.mean())
print("Factor Sharpe Ratios (annualized): \n", (model_bonds.Fac.T.mean()/model_bonds.Fac.T.std()) * (12**0.5))

# IPCA K=5 With Anomalies
K = 5
gFac = pd.DataFrame(1., index=sorted(R.keys()), columns=['anomaly']).T
model_bonds_unrestricted = IPCA(Z, R=R, K=K, gFac=gFac)
model_bonds_unrestricted.run_ipca(dispIters=True)

print("R2: ", model_bonds_unrestricted.r2)
model_bonds_unrestricted.visualize_factors()
model_bonds_unrestricted.visualize_gamma_heatmap()
print("Factor Expected Returns (monthly) \n: ",model_bonds_unrestricted.Fac.T.mean())
print("Factor Sharpe Ratios (annualized): \n", (model_bonds_unrestricted.Fac.T.mean()/model_bonds_unrestricted.Fac.T.std()) * (12**0.5))

######################################
#####  Common Factor Structure  #####
######################################

logging.info("##### Common Factor Structure #####")

##### find cross-asset-class correlation matrix

# transpose to get factors as columns
fac_stocks = model_stocks.Fac.T
fac_bonds  = model_bonds.Fac.T

# find common date period
common_idx = fac_stocks.index.intersection(fac_bonds.index)
fac_stocks = fac_stocks.loc[common_idx]
fac_bonds  = fac_bonds.loc[common_idx]

# compute full cross-correlation matrix (5x5)
cross_corr = pd.DataFrame(
    index=fac_bonds.columns, columns=fac_stocks.columns, dtype=float
)
for b in fac_bonds.columns:
    for s in fac_stocks.columns:
        cross_corr.loc[b, s] = fac_bonds[b].corr(fac_stocks[s])

# Plot heatmap
plt.figure(figsize=(8,6))
sns.heatmap(cross_corr.astype(float), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation")
plt.ylabel("Bond Factors")
plt.xlabel("Stock Factors")
plt.tight_layout()
plt.show()