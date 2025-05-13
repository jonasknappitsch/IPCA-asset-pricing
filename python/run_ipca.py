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
        data = signals.merge(crsp[['permno', 'date', 'ret', 'rf','excess_ret']], on=['permno', 'date'], how='left')
        # set entity-time index
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
    processed_data = processed_data[processed_data['excess_ret'].notnull()]

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
    
    outline_data(data, signal_names)
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
    IPCAs[0].visualize_gamma()


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
    