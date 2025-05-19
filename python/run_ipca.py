import numpy as np
import pandas as pd
import pickle
from ipca import IPCA
import wrds # datasets
import matplotlib.pyplot as plt # visualization

def download_data(dataset="gukellyxiu"):
    if(dataset=="gukellyxiu"):
        '''
        Source: Gu, Kelly and Xiu (2020) "Empirical Asset Pricing via Machine Learning"
        https://dachxiu.chicagobooth.edu
        https://dachxiu.chicagobooth.edu/download/datashare.zip
        '''
        
        try:
            signals = pd.read_csv('data/gukellyxiu.csv', delimiter=',')
            print("Signal data loaded successfully.")
        except FileNotFoundError:
            print("Couldn't find suitable data. Please provide 'data/gukellyxiu.csv'")

        # set entity-time multi-index (permno, date)
        signals = signals.rename(columns={'DATE': 'date'})
        signals["date"] = pd.to_datetime(signals["date"].astype(str), format="%Y%m%d")
        signals = signals.set_index(['permno', 'date'])

        # construct signal_names but exclude any non-signal columns
        non_signal_cols = ['sic2']
        signal_names = [col for col in signals.columns if col not in non_signal_cols]

        # lag signals to ensure return at t is predicted by previous signals
        use_frequency_lag = False # whether to lag according to signal frequency t=[1,4,6] as per Gu Kelly Xiu (2020), or uniformly by t=1
        
        if use_frequency_lag:
            # advanced lag based on signal frequency
            characteristics_table = pd.read_csv('data/characteristics_table.csv', delimiter=',')
            lag_frequency = {'Monthly': 1, 'Quarterly': 4, 'Annual': 6}
            lag_map = characteristics_table.set_index('Acronym')['Frequency'].map(lag_frequency).to_dict() # lag map as per Gu Kelly Xiu (2020)
            
            lagged_signals = pd.DataFrame(index=signals.index)
            for col in signal_names:
                lag = lag_map.get(col, 1)
                lagged_signals[col] = signals[col].groupby(level="permno").shift(lag)
        else:
            # simple uniform lag (e.g. 1 month)
            lagged_signals = signals[signal_names].groupby(level="permno").shift(1)

        # set date from end of month to 28th to allow join with returns
        lagged_signals = lagged_signals.reset_index()
        lagged_signals['date'] = lagged_signals['date'].apply(lambda d: d.replace(day=28))
        lagged_signals = lagged_signals.set_index(['permno', 'date'])

        # download WRDS CRSP return data
        wrds_conn = wrds.Connection()
        crsp = wrds_conn.raw_sql("""
                        SELECT a.permno, a.date, a.ret*100 as ret
                        FROM crsp.msf AS a
                        JOIN crsp.msenames AS c 
                            ON a.permno = c.permno
                            AND a.date >= c.namedt
                            AND a.date <= c.nameendt
                        WHERE c.shrcd in (10, 11, 12) 
                            AND c.exchcd in (1, 2, 3)
            """, date_cols=["date"])

        # change crsp date day to 28th to allow join
        crsp['date'] = crsp['date'].apply(lambda d: d.replace(day=28))

        # change crsp permno dtype to int32 for memory alignment # TODO harmonize memory conversions
        crsp['permno'] = crsp['permno'].astype('int32')

        # download Fama-French risk-free rate for excess returns
        rf = wrds_conn.raw_sql("""
                        SELECT date, rf
                        FROM ff.factors_monthly
                        """, date_cols=["date"])

        # change ff date from 1st of month to 28th of previous month to allow join
        rf['date'] = rf['date'] - pd.DateOffset(days=1)
        rf['date'] = rf['date'].apply(lambda d: d.replace(day=28))

        # merge crsp returns with rf and compute excess returns
        crsp = crsp.merge(rf, on='date', how='left')
        crsp['excess_ret'] = crsp['ret'] - crsp['rf']

        # merge data by left join return on signals
        data = lagged_signals.merge(crsp[['permno', 'date', 'ret', 'rf','excess_ret']], on=['permno', 'date'], how='left')
        
        # set entity-time multi-index
        data = data.set_index(['permno','date'])

    elif(dataset=="other"):
        # if user wants to use any other dataset, implement here...
        raise NotImplementedError('Please implement support for other dataset first.')
    else:
        raise NotImplementedError('No valid dataset selected.')
    
    try:
        with open('data/raw_data.pkl', 'wb') as outp:
            pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)
        print("Raw data saved to data/raw_data.pkl")
    except:
        print("Couldn't save raw data.")

    return(data, signal_names)

def preprocessing(data, signal_names):
    print("Preprocessing data...")

    # 1. filter by date pursuant to Gu Kelly Xiu 2020 (TODO consider removing observations before 1963/1980, cf. Chen and McCoy 2024)
    start_year = 1957
    end_year = 2016 # TODO maybe extend to 2021?

    processed_data = data[
    (data.index.get_level_values('date').year >= start_year) &
    (data.index.get_level_values('date').year <= end_year)]

    # 2. remove rows where return is null # TODO check if this is desired, or eg linear interpolation
    processed_data = processed_data[processed_data['excess_ret'].notnull()]
    
    # 3. remove rows where all signals are missing (e.g. due to lagging)
    processed_data = processed_data.dropna(subset=signal_names, how='all')

    # 4. standardize by performing rank-normalization among non-missing observations
    for col in signal_names:
        # rank characteristics cross-sectionally by date while ignoring NAs
        ranks = processed_data[col].groupby(level='date').transform(
            lambda x: x.rank(method='average', na_option='keep')
        )
        # get # of non-missing observations per date
        counts = processed_data[col].groupby(level='date').transform(
            lambda x: x.notnull().sum()
        )
        # map into [-0.5, 0.5] interval among non-missing observations
        processed_data[col] = (ranks / counts) - 0.5

    # 5. impute missing values with median, which equals 0 after standardization
    processed_data[signal_names] = processed_data[signal_names].fillna(0)

    try:
        with open('data/processed_data.pkl', 'wb') as outp:
            pickle.dump(processed_data, outp, pickle.HIGHEST_PROTOCOL)
        print("Processed data saved to data/processed_data.pkl")
    except:
        print("Couldn't save processed data.")
    
    return(processed_data)

def outline_data(data):
    # Total unique assets
    unique_assets = data.index.get_level_values("permno").nunique()
    print('Total unique assets: ', unique_assets)

    # Average stocks per month 
    avg_stocks_per_month = (data.reset_index()
                            .groupby("date")["permno"]
                            .nunique()
                            .mean()
                            )
    print('Average stocks per month: ', round(avg_stocks_per_month))

def save_data(IPCAs):
    try:
        with open('data/result_data.pkl', 'wb') as outp:
            pickle.dump(IPCAs, outp, pickle.HIGHEST_PROTOCOL)
        print("Result data saved to data/result_data.pkl")
    except:
        print("Couldn't export result data.")
    

if __name__ == '__main__':
    
    download_input = input("Do you want to download new data (y)?\n")
    if(download_input.lower() == "y"):
        dataset_input = input("Select your desired dataset [gukellyxiu | other]:\n")
        data, signal_names = download_data(dataset_input) # load your data here
    else:
        try:
            with open('data/raw_data.pkl', 'rb') as inp:
                data = pickle.load(inp)
            print("Using previous raw data from data/raw_data.pkl.")
            signal_names = [col for col in data.columns if col not in ["permno", "date","signals_date","ret","rf","excess_ret","sic2"]] # TODO find more dynamic solution
        except:
            print("Couldn't find suitable raw data.")    
    
    data = preprocessing(data, signal_names)
    
    outline_data(data)

    # construct Z and R as required by ipca (convert pd.Float64 to np.float32, drop date from index)
    # TODO check whether np.float32 conversion makes sense earlier. conversion is necessary
    # as otherwise characteristics happen to become pd.Float64 at some point
    Z = {t: df[signal_names].astype(np.float32).droplevel("date") for t, df in data.groupby("date")}
    R = {t: s["excess_ret"].astype(np.float32).droplevel("date") for t, s in data.groupby("date")}
    
    # IPCA: no anomaly
    Ks = [1,2,3,4,5]
    IPCAs = []

    for K in Ks:
        model = IPCA(Z, R=R, K=K)
        model.run_ipca(dispIters=True)
        IPCAs.append(model)

    save_data(IPCAs)    

    print(IPCAs[0].r2)
    print(IPCAs[0].Gamma)
    print(IPCAs[0].Fac)
    IPCAs[0].visualize_factors()
    IPCAs[0].visualize_gamma_heatmap()

    """
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
    