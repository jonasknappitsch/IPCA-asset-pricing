import numpy as np
import pandas as pd
import pickle
from ipca import IPCA
import wrds # datasets
import matplotlib.pyplot as plt # visualization

def download_data(dataset="gukellyxiu"):
    if(dataset == "grunfeld"):
        from statsmodels.datasets import grunfeld
        data = grunfeld.load_pandas().data

        # convert date
        data.year = data.year.astype(np.int64)

        # establish unique IDs
        N = len(np.unique(data.firm))
        ID = dict(zip(np.unique(data.firm).tolist(), np.arange(1, N+1)+5))
        data.firm = data.firm.apply(lambda x: ID[x])

        # rearrange and rename to proper format
        data = data[['firm', 'year', 'invest', 'value', 'capital']]
        data = data.rename(columns={'year': 'date'})
        data = data.rename(columns={'invest': 'ret'}) # technically no return, but for consistency
        
        # set entity-time index and signals
        data = data.set_index(['firm', 'date'])
        signal_names = ['value','capital']

    elif(dataset == "openap"):
        '''
        Source: Chen and Zimmermann (2021) "Open Source Cross-Sectional Asset Pricing"
        https://github.com/mk0417/open-asset-pricing-download/

        212 Predictors
        '''

        # download all Chen-Zimmermann predictors
        import openassetpricing as oap
        openap = oap.OpenAP(202408)
        signals = openap.dl_all_signals('pandas') # download all signals
        # openap_signals = openap.dl_signal('pandas',['AM','Beta','BM','CF','Mom12m']) # download certain signals of openap.dl_signal_doc('pandas')

        # lag to ensure return at t is predicted by signals at t+1 and assume signal is available for trading end of month (28th) to allow merge with returns
        signals['date'] = pd.to_datetime(signals['yyyymm'].astype(str) + '28', format='%Y%m%d') + pd.DateOffset(months=1)
        # keep original (non-lagged) signal date stored for clarity
        signals = signals.rename(columns={'yyyymm': 'signals_date'})

        # get signal names
        signal_names = [col for col in signals.columns if col not in ['permno', 'date','signals_date']]

        # convert int64/float64 to int32/float32 for performance and memory reasons
        signals['permno'] = signals['permno'].astype('int32')
        signals[signal_names] = signals[signal_names].astype('float32')

        # arrange order
        cols = ['permno','date','signals_date'] + signal_names
        signals = signals[cols]

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

        # change crsp date day to 28th to allow join with signals
        crsp['date'] = crsp['date'].apply(lambda d: d.replace(day=28))

        # change crsp permno dtype to int32 for memory alignment # TODO harmonize memory conversions
        crsp['permno'] = crsp['permno'].astype('int32')

        # merge data by left join return on signals
        data = signals.merge(crsp[['permno', 'date', 'ret']], on=['permno', 'date'], how='left')
        # set entity-time index
        data = data.set_index(['permno','date'])

    elif(dataset=="gukellyxiu"):
        '''
        Source: Gu, Kelly and Xiu (2020) "Empirical Asset Pricing via Machine Learning"
        https://dachxiu.chicagobooth.edu
        '''
        
        try:
            signals = pd.read_csv('data/gukellyxiu/raw_data.csv', delimiter=',')
            print("Data loaded successfully.")
        except FileNotFoundError:
            print("Couldn't find suitable data.")

        # lag to ensure return at t is predicted by signals at t+1 and assume signal is available for trading end of month (28th) to allow merge with returns
        signals['date'] = pd.to_datetime(signals['DATE'].astype(str), format='%Y%m%d') + pd.DateOffset(months=1)
        signals['date'] = signals['date'].apply(lambda d: d.replace(day=28))

        # keep original (non-lagged) signal date stored for clarity
        signals = signals.rename(columns={'DATE': 'signals_date'})

        # get signal names
        signal_names = [col for col in signals.columns if col not in ['permno', 'date','signals_date','sic2']]

        # convert int64/float64 to int32/float32 for performance and memory reasons
        signals['permno'] = signals['permno'].astype('int32')
        signals[signal_names] = signals[signal_names].astype('float32')

        # arrange order
        cols = ['permno','date','signals_date','sic2'] + signal_names
        signals = signals[cols]

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

        # change crsp date day to 28th to allow join with signals
        crsp['date'] = crsp['date'].apply(lambda d: d.replace(day=28))

        # change crsp permno dtype to int32 for memory alignment # TODO harmonize memory conversions
        crsp['permno'] = crsp['permno'].astype('int32')

        # merge data by left join return on signals
        data = signals.merge(crsp[['permno', 'date', 'ret']], on=['permno', 'date'], how='left')
        # set entity-time index
        data = data.set_index(['permno','date'])

    else:
        raise NotImplementedError('No valid dataset selected.')
    
    try:
        with open('raw_data.pkl', 'wb') as outp:
            pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)
        print("Raw data saved to raw_data.pkl")
    except:
        print("Couldn't save raw data.")

    return(data, signal_names)

def preprocessing(data, signal_names):
    """
    how to deal with missing values [mean imputation, etc.]?
    zero mean, unit standard deviation?
    """
    print("Preprocessing data...")
    # filter by date (remove observations before 1963/1980, cf. Chen and McCoy 2024 # TODO)
    start_year = 1957
    end_year = 2016

    processed_data = data[
    (data.index.get_level_values('date').year >= start_year) &
    (data.index.get_level_values('date').year <= end_year)
]

    # remove observations where return is null (# TODO check whether this is duplicate with ipca __init__ is_valid)
    processed_data = processed_data[processed_data['ret'].notnull()]

    # TODO filter for minimum observations per firm?
    
    # standardize signals cross-sectionally at each date to achieve zero mean and unit std
    # TODO check why processed_data["AM"].max() is not exactly 0.5 but slightly lower, alternative try scipy ranking
    for col in signal_names:
        processed_data[col] = processed_data.groupby('date')[col].transform(
            lambda x: (x.rank(method='average') - 1) / (len(x) - 1) - 0.5
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

    if(download_input.lower() == "y"):
        dataset_input = input("Select your desired dataset [grunfeld | openap | gukellyxiu]:\n")
        data, signal_names = download_data(dataset_input) # load your data here
    else:
        try:
            with open('raw_data.pkl', 'rb') as inp:
                data = pickle.load(inp)
            signal_names = [col for col in data.columns if col not in ["permno", "date","signals_date","ret","sic2"]] # TODO find more dynamic solution
        except:
            print("Couldn't find suitable raw data.")    
    
    data = preprocessing(data, signal_names)
    
    # construct Z and R as required by ipca (convert pd.Float64 to np.float32, drop date from index)
    # TODO check whether np.float32 conversion makes sense earlier. conversion is necessary
    # as otherwise characteristics happen to become pd.Float64 at some point
    Z = {t: df[signal_names].astype(np.float32).droplevel("date") for t, df in data.groupby("date")}
    R = {t: s["ret"].astype(np.float32).droplevel("date") for t, s in data.groupby("date")}
    
    # IPCA: no anomaly
    Ks = [1,2,3,4,5]
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
