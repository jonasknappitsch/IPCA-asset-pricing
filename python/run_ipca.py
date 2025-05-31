import numpy as np
import pandas as pd
import pickle
from ipca import IPCA
import wrds # datasets
import matplotlib.pyplot as plt # visualization
from tabulate import tabulate # print formatted 

def download_data(dataset="fnw"):
    if(dataset=="fnw"):
        '''
        Source: Freyberger, Neuhierl and Weber (2017) "Dissecting Characteristics Nonparametrically"
        As used in: Kelly, Pruitt and Su (2019) "Characteristics are Covariances"
        Provided by:
        - https://sethpruitt.net/
        - https://sethpruitt.net/research/
        '''

        try:
            signals = pd.read_csv('data/fnw.csv', delimiter=',')
            print("Signal data loaded successfully.")
        except FileNotFoundError:
            print("Couldn't find suitable data. Please provide 'data/fnw.csv'")

        # drop metadata and non-needed columns
        signals = signals.drop(columns=['Unnamed: 0', 'yy', 'mm','q10', 'q20', 'q50', 'prc'])

        signals['date'] = pd.to_datetime(signals['date'])
        
        # signals['date'] = signals['date'].apply(lambda d: d.replace(day=28)) # TODO only needed in case of merge

        # rename ret to use excess_ret (TODO check whether ret in fnw is already excess_ret)
        signals = signals.rename(columns={'ret': 'excess_ret'})

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
        signals = signals.rename(columns=rename_map)

        # set entity-time multi-index
        signals = signals.set_index(['permno', 'date'])

        # retrieve signal names
        non_signal_cols = ['excess_ret']
        signal_names = [col for col in signals.columns if col not in non_signal_cols]

        data = signals

    elif(dataset=="gukellyxiu"):
        '''
        Source: Gu, Kelly and Xiu (2020) "Empirical Asset Pricing via Machine Learning"
        Provided by:
        - https://dachxiu.chicagobooth.edu
        - https://dachxiu.chicagobooth.edu/download/datashare.zip
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
        use_frequency_lag = True # whether to lag according to signal frequency t=[1,4,6] as per Gu Kelly Xiu (2020), or uniformly by t=1
        
        if use_frequency_lag:
            # advanced lag based on signal frequency
            characteristics_table = pd.read_csv('data/characteristics_table.csv', delimiter=',')
            lag_frequency = {'Monthly': 1, 'Quarterly': 4, 'Annual': 6} # TODO check 1,4,6 (pref) or 1,5,7
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
        # TODO check if rf should scaled somehow (eg /100, *100), s.t. SELECT date, rf / 100 AS rf
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
    else:
        raise NotImplementedError('Selected dataset is not supported. Please implement first.')
    
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
    start_year = 1957 # default: 1957
    end_year = 2016 # default: 2016, TODO could be extended until 2021

    processed_data = data[
    (data.index.get_level_values('date').year >= start_year) &
    (data.index.get_level_values('date').year <= end_year)]

    # 2. remove rows where return is null # TODO check if this is desired, or eg linear interpolation
    processed_data = processed_data[processed_data['excess_ret'].notnull()]
    
    # 3. remove rows where all signals are missing (e.g. due to lagging)
    processed_data = processed_data.dropna(subset=signal_names, how='all')
    
    # 4. standardize by performing rank-normalization among non-missing (caveat) observations
    
    for col in signal_names:
        processed_data[col] = processed_data.groupby(level='date')[col].transform(
            lambda x: ((x.rank(method='average', na_option='keep') - 1) / (x.count() - 1)) - 0.5
        )
    
    '''for col in signal_names:
        # rank characteristics cross-sectionally by date while ignoring NAs
        ranks = processed_data[col].groupby(level='date').transform(
            lambda x: x.rank(method='average', na_option='keep')
        )
        # get # of non-missing observations per date
        counts = processed_data[col].groupby(level='date').transform(
            lambda x: x.notnull().sum()
        )
        # map into [-0.5, 0.5] interval among non-missing observations
        processed_data[col] = (ranks / counts) - 0.5'''
        
    # 5. impute missing values with median, which equals 0 after standardization
    processed_data[signal_names] = processed_data[signal_names].fillna(0)

    # 6. add constant
    processed_data["const"] = 1.0
    signal_names.append("const")

    try:
        with open('data/processed_data.pkl', 'wb') as outp:
            pickle.dump(processed_data, outp, pickle.HIGHEST_PROTOCOL)
        print("Processed data saved to data/processed_data.pkl")
    except:
        print("Couldn't save processed data.")
    
    return(processed_data, signal_names)

def outline_data(data, signal_names):
    summary = {}

    total_observations = len(data)
    total_assets = data.index.get_level_values("permno").nunique() # total unique assets
    total_months = data.index.get_level_values('date').nunique() # total months covered
    avg_assets_per_month = total_observations/total_months # average assets per month
    start_date = data.index.get_level_values('date').min().strftime('%Y-%m')
    end_date = data.index.get_level_values('date').max().strftime('%Y-%m')

    summary['Number of assets'] = total_assets
    summary['Number of months'] = total_months
    summary['Total observations'] = total_observations
    summary['Average assets per month'] = avg_assets_per_month
    summary['Start date'] = start_date
    summary['End date'] = end_date
    summary['Number of characteristics'] = len(signal_names)

    outline = pd.DataFrame.from_dict(summary, orient='index', columns=['Value'])

    print("\n=== Data Outline ===")
    print(tabulate(outline, headers=['Metric', 'Value'], tablefmt='github', floatfmt='.2f'))

def save_data(IPCAs, name):
    try:
        filename = f"data/result_data_{name}.pkl"
        with open(filename, 'wb') as outp:
            pickle.dump(IPCAs, outp, pickle.HIGHEST_PROTOCOL)
        print(f"Saved result data to {filename}")
    except:
        print("Couldn't export result data.")

def evaluate_IPCAs(IPCAs, name):
    results = []
    for model in IPCAs:
        K = model.K
        results.append({
                    "K": K,
                    "R2_Total": round(float(model.r2.get("R_Tot", float("nan"))),4),
                    "R2_Pred": round(float(model.r2.get("R_Prd", float("nan"))),4),
                    "xR2_Total": round(float(model.r2.get("X_Tot", float("nan"))),4),
                    "xR2_Pred": round(float(model.r2.get("X_Prd", float("nan"))),4),
                })
        filename = f"data/results_{name}"
        model.visualize_factors(save_path=f"{filename}_factors_K{K}.png")
        model.visualize_gamma_heatmap(save_path=f"{filename}_gamma_heatmap_K{K}.png")
        
    df = pd.DataFrame(results)
    filename = f"data/results_{name}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved R2 results to {filename}")
    print("\n=== R2 Results Outline ===")
    print(tabulate(df, headers="keys", tablefmt="github", showindex=False))

if __name__ == '__main__':
    
    download_input = input("Do you want to download new data [y (default) | n]? ") or "y"
    if(download_input.lower() == "y"):
        dataset_input = input("Choose dataset [fnw (default) | gukellyxiu | other]: ").strip() or "fnw"
        data, signal_names = download_data(dataset_input) # load your data here
        data, signal_names = preprocessing(data, signal_names)
    else:
        try:
            with open('data/processed_data.pkl', 'rb') as inp:
                data = pickle.load(inp)
            print("Using previous processed data from data/processed_data.pkl.")
            signal_names = [col for col in data.columns if col not in ["permno", "date","signals_date","ret","rf","excess_ret","sic2"]] # TODO find more dynamic solution
        except:
            print("Couldn't find suitable processed data as input.")    
    
    outline_data(data, signal_names)

    # construct Z and R as required by ipca (convert pd.Float64 to np.float32, drop date from index)
    # TODO check whether np.float32 conversion makes sense earlier. conversion is necessary
    # as otherwise characteristics happen to become pd.Float64 at some point
    Z = {t: df[signal_names].astype(np.float32).droplevel("date") for t, df in data.groupby("date")}
    R = {t: s["excess_ret"].astype(np.float32).droplevel("date") for t, s in data.groupby("date")}
    
    # IPCA: no anomaly
    
    Ks = [1,2,3,4,5,6]
    IPCAs = []

    for K in Ks:
        model = IPCA(Z, R=R, K=K)
        model.run_ipca(dispIters=True)
        IPCAs.append(model)

    save_data(IPCAs, name="no_anomaly")
    evaluate_IPCAs(IPCAs,name="no_anomaly")
    

    # IPCA: with anomaly
    Ks = [1,2,3,4,5,6]
    IPCAs = []

    gFac = pd.DataFrame(1., index=sorted(R.keys()), columns=['anomaly']).T

    for K in Ks:
        model = IPCA(Z, R=R, K=K, gFac=gFac)
        model.run_ipca(dispIters=True)
        IPCAs.append(model)

    save_data(IPCAs,name="anomaly")    
    evaluate_IPCAs(IPCAs,name="anomaly")
    
    ##### IPCA: with pre-specified factor (PSF) MKT as per CAPM #####
    # TODO ipca doesn't seem to work with only gFac when no fFac is passed
    '''
    Ks = [1]
    IPCAs = []

    # retrieve MKT from ff.factors_monthly
    wrds_conn = wrds.Connection()
    mkt = wrds_conn.raw_sql("""
        SELECT date, mktrf / 100 AS mktrf
        FROM ff.factors_monthly
    """, date_cols=["date"])

    # align to date structure (e.g., day=28)
    mkt['date'] = mkt['date'] - pd.DateOffset(days=1)
    mkt['date'] = mkt['date'].apply(lambda d: d.replace(day=28))
    mkt.set_index('date', inplace=True)

    # define gFac based on CAPM MKT as PSF
    gFac = mkt.reindex(sorted(R.keys()))  # R is your dictionary of excess returns
    gFac = gFac.T  # transpose: rows = factor(s), cols = time
    gFac.index = ['mkt']  # name the factor

    for K in Ks:
        model = IPCA(Z, R=R, K=K, gFac=gFac)
        model.run_ipca(dispIters=True)
        IPCAs.append(model)

    save_data(IPCAs,name="PSF_CAPM")    
    evaluate_IPCAs(IPCAs,name="PSF_CAPM")
    '''

    """
    # IPCA: no anomaly
    IPCAs[0].r2
    IPCAs[0].Gamma
    IPCAs[0].Fac
    IPCAs[0].visualize_factors()
    IPCAs[0].visualize_gamma_heatmap()   

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
    