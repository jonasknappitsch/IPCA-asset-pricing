import numpy as np
import pandas as pd
import pickle
from ipca import IPCA

# datasets
from statsmodels.datasets import grunfeld
import wrds
import openassetpricing as oap

# visualization
import matplotlib.pyplot as plt

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
        
        # lag x by 1 period if required:
        # X_lagged = X.groupby('firm').shift(1).dropna()
        # y_aligned = y.loc[X_lagged.index]

        # convert to time-indexed dict(T) as required:
        # Z (dict(T) of df(NxL)): characteristics; can be rank-demeaned
        # R (dict(T) of srs(N); not needed for managed-ptf-only version): asset returns
        Z = {t: df.droplevel('year') for t, df in X.groupby('year')}
        R = {t: s.droplevel('year') for t, s in y.groupby('year')}

    elif(dataset == "openap"):
        # https://github.com/mk0417/open-asset-pricing-download/blob/master/examples/ML_portfolio_example.ipynb
        
        # download WRDS CRSP return data
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

        # download all Chen-Zimmermann predictors
        openap = oap.OpenAP(202408)
        openap_signals = openap.dl_all_signals('pandas') # download all signals
        
        # openap_signals = openap.dl_signal('pandas',['AM','Beta','BM','CF','Mom12m']) # download certain signals of openap.dl_signal_doc('pandas')

        # get signal names
        signal_names = [col for col in openap_signals.columns if col not in ["permno", "yyyymm"]]

        # convert signals from float64 to float32 for performance and memory reasons
        openap_signals[signal_names] = openap_signals[signal_names].astype('float32')

        # lag to ensure return at t is predicted by signals at t+1, assume signal is available for trading end of month (28th)
        openap_signals["date"] = pd.to_datetime(openap_signals["yyyymm"].astype(str) + "28", format="%Y%m%d") + pd.DateOffset(months=1)

        # keep original signal date stored for clarity and arrange order
        openap_signals = openap_signals.rename(columns={"yyyymm": "signals_date"})
        cols = ["permno","date","signals_date"] + signal_names
        openap_signals = openap_signals[cols]

        # change crsp date day to 28th to allow join
        crsp["date"] = crsp["date"].apply(lambda d: d.replace(day=28))

        # change crsp permno dtype to int32 for memory alignment
        crsp["permno"] = crsp["permno"].astype("int32")

        # merge data by left join return on signals
        data = openap_signals.merge(crsp[["permno", "date", "ret"]], on=["permno", "date"], how="left")
    else:
        raise NotImplementedError('No valid dataset selected.')
    
    try:
        with open('raw_data.pkl', 'wb') as outp:
            pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)
        print("Raw data saved to raw_data.pkl")
    except:
        print("Couldn't save raw data.")

    return(data)

def preprocessing(data, signal_names):
    """
    how to deal with missing values [mean imputation, etc.]?
    zero mean, unit standard deviation?
    """
    # remove observations before 1963 (ca. 393739 observations, cf Chen and McCoy 2024)
    processed_data = data[data["date"].dt.year >= 1963]

    # remove observations where return is null (ca. 1271699 observations)
    processed_data = processed_data[processed_data["ret"].notnull()]

    # TODO filter for minimum observations per firm?

    ### standardization
    
    # standardize signals cross-sectionally at each date to achieve zero mean and unit std
    # TODO check why processed_data["AM"].max() is not exactly 0.5 but slightly lower, alternative try scipy ranking
    for col in signal_names:
        processed_data[col] = processed_data.groupby("date")[col].transform(
            lambda x: (x.rank(method="average") - 1) / (len(x) - 1) - 0.5
        )

    # replace NAs with 0
    processed_data[signal_names] = processed_data[signal_names].fillna(0)
    
    try:
        with open('processed_data.pkl', 'wb') as outp:
            pickle.dump(processed_data, outp, pickle.HIGHEST_PROTOCOL)
        print("Processed data saved to processed_data.pkl")
    except:
        print("Couldn't save processed data.")
    
    return(processed_data)

def save_data(IPCAs):
    try:
        with open('result_data.pkl', 'wb') as outp:
            pickle.dump(IPCAs, outp, pickle.HIGHEST_PROTOCOL)
        print("Result data saved to result_data.pkl")
    except:
        print("Couldn't export result data.")
    

if __name__ == '__main__':
    
    download_input = input("Do you want to download new data (y)?\n")

    if(download_input == "y"):
        dataset_input = input("Select your desired dataset [grunfeld | openap]:\n")
        download_data(dataset_input) # load your data here

    try:
        with open('raw_data.pkl', 'rb') as inp:
            data = pickle.load(inp)
    except:
        print("Couldn't find suitable raw data.")

    signal_names = [col for col in data.columns if col not in ["permno", "date","signals_date","ret"]] # TODO find more dynamic solution
    
    data = preprocessing(data, signal_names)

    data.set_index(["permno", "date"], inplace=True) # TODO move to correct (earlier) position

    
    # construct Z and R as required by ipca (convert pd.Float64 to np.float32, drop date from index)
    # TODO check whether np.float32 conversion makes sense earlier. conversion is necessary
    # as otherwise characteristics happen to become pd.Float64 at some point
    Z = {t: df[signal_names].astype(np.float32).droplevel("date") for t, df in data.groupby("date")}
    R = {t: s["ret"].astype(np.float32).droplevel("date") for t, s in data.groupby("date")}
    
    # IPCA: no anomaly
    K = 5 # specify K

    Ks = [1,2,3]
    IPCAs = []

    for K in Ks:
        model = IPCA(Z, R=R, K=K)
        model.run_ipca(dispIters=True)
        IPCAs.append(model)

    save_data(IPCAs)

    """
    print(ipca_0.r2)
    print(ipca_0.Gamma)
    print(ipca_0.Fac)
    ipca_0.visualize_factors()
    ipca_0.visualize_gamma()

    # IPCA: with anomaly
    gFac = pd.DataFrame(1., index=sorted(R.keys()), columns=['anomaly']).T
    ipca_1 = IPCA(Z, R=R, K=K, gFac=gFac)
    ipca_1.run_ipca(dispIters=True)

    # IPCA: with anomaly and a pre-specified factor
    gFac = pd.DataFrame(1., index=sorted(R.keys()), columns=['anomaly'])
    gFac['mkt'] = pd.Series({key:R[key].mean() for key in gFac.index}) # say we include the equally weighted market
    gFac = gFac.T
    ipca_2 = IPCA(Z, R=R, K=K, gFac=gFac)
    ipca_2.run_ipca(dispIters=True)
    """
    
    ########## compare results ##########
    """
    tested: this code reaches same results on grunfield data
    as Kelly's python implementation "https://github.com/bkelly-lab/ipca.git"
    for
    - Gamma
    - Factors
    but different results for
    - R2 (asset, portfolio)
    """

    """
    print(ipca_0.r2)
    print(ipca_0.Gamma)
    # print(ipca_2.Gamma)
    print(ipca_0.Fac)
    # ipca_2.visualize_factors()
    """
