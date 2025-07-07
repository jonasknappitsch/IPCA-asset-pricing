import os
import numpy as np
import pandas as pd
import pickle
from ipca import IPCA
import wrds # datasets
import urllib.request
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
            signals = pd.read_csv(f'data/{dataset}/fnw.csv', delimiter=',')
            print("Signal data loaded successfully.")
        except FileNotFoundError:
            print(f"Couldn't find suitable data. Please provide 'data/{dataset}/fnw.csv'")

        # drop metadata and non-needed columns
        signals = signals.drop(columns=['Unnamed: 0', 'yy', 'mm','q10', 'q20', 'q50', 'prc'])

        signals['date'] = pd.to_datetime(signals['date'])
        
        # set date from end of month to 28th to allow join with rf rate
        signals['date'] = signals['date'].apply(lambda d: d.replace(day=28)) 
        
        # download Fama-French risk-free rate to compute excess returns
        wrds_conn = wrds.Connection()
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
        
        print(data)

    elif(dataset=="oap"):
        '''
        Source: Chen and Zimmermann (2021) "Open Source Cross-Sectional Asset Pricing"
        Provided by:
        - https://www.openassetpricing.com
        - https://github.com/mk0417/open-asset-pricing-download
        '''

        try:
            import openassetpricing as oap
            openap = oap.OpenAP(202408)
            signals = openap.dl_all_signals('pandas') # download all signals
        except:
            print("Couldn't download openassetpricing data. Please check specification.")
        
        print("Signals: ", signals)
        # lag to ensure return at t is predicted by signals at t-1, assume signal is available for trading end of month (28th)
        lagged_signals = signals.copy()
        lagged_signals["date"] = pd.to_datetime(signals["yyyymm"].astype(str) + "28", format="%Y%m%d") + pd.DateOffset(months=1)
        lagged_signals = lagged_signals.set_index(['permno', 'date'])
        print("Lagged signals: ", lagged_signals)

        # drop metadata and non-needed columns
        lagged_signals = lagged_signals.drop(columns=['yyyymm'])

        # construct signal_names but exclude any non-signal columns
        non_signal_cols = ['']
        signal_names = [col for col in lagged_signals.columns if col not in non_signal_cols]

        # download WRDS CRSP return data
        wrds_conn = wrds.Connection()
        crsp = wrds_conn.raw_sql("""
                        SELECT a.permno, a.date, a.ret
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

        # change ff date from 1st day of month to 28th to allow join
        rf['date'] = rf['date'].apply(lambda d: d.replace(day=28))

        # merge crsp returns with rf and compute excess returns
        crsp = crsp.merge(rf, on='date', how='left')
        crsp['excess_ret'] = crsp['ret'] - crsp['rf']

        # merge data by left join return on signals
        data = lagged_signals.merge(crsp[['permno', 'date', 'ret', 'rf','excess_ret']], on=['permno', 'date'], how='left')
        
        # set entity-time multi-index
        data = data.set_index(['permno','date'])
        print(data)

    elif(dataset=="gukellyxiu"):
        '''
        Source: Gu, Kelly and Xiu (2020) "Empirical Asset Pricing via Machine Learning"
        Provided by:
        - https://dachxiu.chicagobooth.edu
        - https://dachxiu.chicagobooth.edu/download/datashare.zip
        '''
        
        try:
            signals = pd.read_csv(f'data/{dataset}/gkx.csv', delimiter=',')
            print("Signal data loaded successfully.")
        except FileNotFoundError:
            print(f"Couldn't find suitable data. Please provide 'data/{dataset}/gkx.csv'")

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
            characteristics_table = pd.read_csv(f'data/{dataset}/characteristics_table_gkx.csv', delimiter=',')
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
                        SELECT a.permno, a.date, a.ret as ret
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

        # change ff date from 1st day of month to 28th to allow join
        rf['date'] = rf['date'].apply(lambda d: d.replace(day=28))

        # merge crsp returns with rf and compute excess returns
        crsp = crsp.merge(rf, on='date', how='left')
        crsp['excess_ret'] = crsp['ret'] - crsp['rf']

        # merge data by left join return on signals
        data = lagged_signals.merge(crsp[['permno', 'date', 'ret', 'rf','excess_ret']], on=['permno', 'date'], how='left')
        
        # set entity-time multi-index
        data = data.set_index(['permno','date'])
    elif(dataset=="osb"):
        '''
        Source: Dickerson, Nozawa and Robotti (2024) "Factor Investing with Delays"
        - https://openbondassetpricing.com/machine-learning-data/
        '''

        # 341 factor machine learning dataset
        # https://openbondassetpricing.com/wp-content/uploads/2024/10/OSBAP_ML_Panel_Oct_2024.zip
        with open(f'data/{dataset}/OSBAP_ML_Panel_Oct_2024.pkl', 'rb') as inp:
            data = pickle.load(inp)

        # replace _ from predictors for consistency with Dickerson et al. (2024)
        data.columns = data.columns.str.replace('_', '', regex=False)

        # select 58 factors as per Dickerson et al. 2024
        signal_names = ["seas25an", "MomSeason06YrPlus", "dVolPut", "seas610an", "dVolCall", "AnnouncementReturn",
                        "ret31", "TrendFactor", "seas11na", "rmax5rvol21d", "rmax521d", "IndRetBig", "ret10",
                        "GrAdExp", "empgr1", "AssetGrowth", "colgr1a", "coagr1a", "ppeinvgr1a", "invgr1a",
                        "noagr1a", "invgr1", "nfnagr1a", "atgr1", "cowcgr1a", "fnlgr1a", "value", "empvalue",
                        "impliedspread", "mom3mspread", "ratingxspread", "strucvalue", "swapspread", "dts",
                        "26sprtod2d", "18spread", "yield", "yldtoworst", "25mom6mspread", "bondprice", "EBM",
                        "eqnetisat", "8nime", "eqnpome", "nime", "eqdur", "turnovervar126d", "corr1260d", "rskew21d",
                        "iskewcapm21d", "28VaR", "iskewhxz421d", "iskewff321d", "betadown252d", "27volatility",
                        "CoskewACX", "kurt", "spreadvol"]

        # create excess_ret and permno column as per model definition for consistency
        # permno is overriden here by bond ID as an exception
        data["excess_ret"] = data["rfretxrf"] # TODO check which ret to use [ensretxrf / rfretxrf / ...]
        data["permno_backup"] = data["permno"]
        data["permno"] = data["ID"]

        # set entity-time multi-index
        data = data.set_index(['permno', 'date'])        
        
        # drop metadata and non-needed columns by keeping only signal_names and excess_ret
        data = data[[col for col in data.columns if col in signal_names or col == "excess_ret"]]

        # lag to ensure return at t is predicted by signals at t-1 TODO check if signals are lagged already
        data[signal_names] = data[signal_names].groupby(level='permno').shift(1)
        
        print(data)
    elif(dataset=="kpp"):
        '''
        Source: Kelly, Palhares and Pruitt (2023) "Modeling Corporate Bond Returns"
        - https://sethpruitt.net/2020/10/28/modeling-corporate-bond-returns/
        '''
        raise NotImplementedError('Selected dataset is not supported. Please implement first.')
    else:
        raise NotImplementedError('Selected dataset is not supported. Please implement first.')
    
    try:
        with open(f'data/{dataset}/raw_data.pkl', 'wb') as outp:
            pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)
        print(f"Raw data saved to data/{dataset}/raw_data.pkl")
    except:
        print("Couldn't save raw data.")

    return(data, signal_names)

def preprocessing(data, dataset, signal_names):
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
        with open(f'data/{dataset}/processed_data.pkl', 'wb') as outp:
            pickle.dump(processed_data, outp, pickle.HIGHEST_PROTOCOL)
        print(f"Processed data saved to data/{dataset}/processed_data.pkl")
    except:
        print("Couldn't save processed data.")
    
    return(processed_data, signal_names)

def load_observable_factors(gFac_K,R):

    """
    Downloads observable factors from WRDS Fama-French library for various K specifications.
    Returns gFac (df(M x T)) for given observable factor specification gFac_K.
    
    gFac_K: int, number of factors (1=CAPM, 3=FF3, 4=FFC4, 5=FF5, 6=FFC6)
    R: dictionary of excess returns with date keys (used to align gFac time index)
    """
    import wrds
    wrds_conn = wrds.Connection()

    # download full factor set from Fama-French
    ff_all = wrds_conn.raw_sql("""
        SELECT date, mktrf, smb, hml, umd, rmw, cma
        FROM ff.fivefactors_monthly
    """, date_cols=["date"])

    ff_all['date'] = ff_all['date'].apply(lambda d: d.replace(day=28))
    ff_all.set_index('date', inplace=True)

    ff_all = ff_all.reindex(sorted(R.keys()))  # match with excess returns R index

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

    summary = pd.DataFrame({
        'min': data.min(),
        'max': data.max(),
        'mean': data.mean(),
        'median': data.median(),
        'std': data.std(),
        'non_missing': data.notna().sum(),
        'missing': data.isna().sum(),
        'unique': data.nunique()
    })

    print("\n=== Variable Summary ===")
    print(tabulate(summary, headers=['Variable', 'min','max','mean','median','std','non-missing','missing','unique'], tablefmt='github', floatfmt='.2f'))

def save_data(IPCAs, dataset, name):
    try:
        filename = f"data/{dataset}/result_data_{name}.pkl"
        with open(filename, 'wb') as outp:
            pickle.dump(IPCAs, outp, pickle.HIGHEST_PROTOCOL)
        print(f"Saved result data to {filename}")
    except:
        print("Couldn't export result data.")

def evaluate_IPCAs(IPCAs, dataset, name):
    results = []
    for model in IPCAs:
        K = model.K
        if(K == 0): # if no latent factor K is specified, use K of gFac instead
            K = model.gFac.shape[0]
        results.append({
                    "K": K,
                    "R2_Total": round(float(model.r2.get("R_Tot", float("nan"))),4),
                    "R2_Pred": round(float(model.r2.get("R_Prd", float("nan"))),4),
                    "xR2_Total": round(float(model.r2.get("X_Tot", float("nan"))),4),
                    "xR2_Pred": round(float(model.r2.get("X_Prd", float("nan"))),4),
                })
        filename = f"data/{dataset}/results_{name}"
        model.visualize_factors(save_path=f"{filename}_factors_K{K}.png")
        model.visualize_gamma_heatmap(save_path=f"{filename}_gamma_heatmap_K{K}.png")
        
    df = pd.DataFrame(results)
    csv_filename = f"{filename}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Saved R2 results to {csv_filename}")
    print("\n=== R2 Results Outline ===")
    print(tabulate(df, headers="keys", tablefmt="github", showindex=False))

if __name__ == '__main__':
    
    '''
    Data is always stored under 'data/{dataset}'.
    Program will look for previously downloaded 'processed_data.pkl', otherwise download new data.
    '''
    dataset = input("Choose dataset [ fnw (default) | oap | gkx | osb ]: ").strip() or "fnw"

    if(os.path.exists(f'data/{dataset}/processed_data.pkl')):
        download_input = input(f"Previous data found for dataset {dataset}. Continue with previous data? [y (default) | n] ") or "y"
        if(download_input.lower() == "y"):
            with open(f'data/{dataset}/processed_data.pkl', 'rb') as inp:
                data = pickle.load(inp)
            print(f"Using previous processed data from data/{dataset}/processed_data.pkl.")
            signal_names = [col for col in data.columns if col not in ["permno", "date","signals_date","ret","rf","excess_ret","sic2"]] # TODO find more dynamic solution
        else:
            print(f"Downloading new data for dataset {dataset}.")
            data, signal_names = download_data(dataset) # load your data here
            data, signal_names = preprocessing(data, dataset, signal_names)
    else:
        print(f"Downloading data for dataset {dataset}.")
        data, signal_names = download_data(dataset) # load your data here
        data, signal_names = preprocessing(data, dataset, signal_names)
    
    print(data.dtypes[data.dtypes!="float64"])
    outline_data(data, signal_names)

    # construct Z and R as required by ipca (convert pd.Float64 to np.float32, drop date from index)
    # TODO check whether np.float32 conversion makes sense earlier. conversion is necessary
    # as otherwise characteristics happen to become pd.Float64 at some point
    Z = {t: df[signal_names].astype(np.float32).droplevel("date") for t, df in data.groupby("date")}
    R = {t: s["excess_ret"].astype(np.float32).droplevel("date") for t, s in data.groupby("date")}
    
    ##### IPCA: no anomaly #####
    Ks = [1,2,3,4,5,6]
    IPCAs = []

    for K in Ks:
        model = IPCA(Z, R=R, K=K)
        model.run_ipca(dispIters=True)
        IPCAs.append(model)

    save_data(IPCAs, dataset, name="no_anomaly")
    evaluate_IPCAs(IPCAs, dataset, name="no_anomaly")
    
    ##### IPCA: with anomaly #####
    Ks = [1,2,3,4,5,6]
    IPCAs = []

    gFac = pd.DataFrame(1., index=sorted(R.keys()), columns=['anomaly']).T

    for K in Ks:
        model = IPCA(Z, R=R, K=K, gFac=gFac)
        model.run_ipca(dispIters=True)
        IPCAs.append(model)

    save_data(IPCAs,dataset,name="anomaly")    
    evaluate_IPCAs(IPCAs,dataset,name="anomaly")
    
    ##### IPCA: with pre-specified factors (PSF) - instrumented #####
    
    Ks = []
    gFac_Ks = [1,3,4,5,6] # define which FF Factor Models to use
    IPCAs = []

    # change start date as Fama-French Five Factors exist only from 1964, change Z and R correspondingly
    start_year = 1964 
    end_year = 2016 # default: 2016
    
    data_new = data[
    (data.index.get_level_values('date').year >= start_year) &
    (data.index.get_level_values('date').year <= end_year)]

    Z_new = {t: df[signal_names].astype(np.float32).droplevel("date") for t, df in data_new.groupby("date")}
    R_new = {t: s["excess_ret"].astype(np.float32).droplevel("date") for t, s in data_new.groupby("date")}

    for gFac_K in gFac_Ks:
        gFac = load_observable_factors(gFac_K, R_new)
        model = IPCA(Z_new, R=R_new, gFac=gFac)
        model.run_ipca(dispIters=True)
        IPCAs.append(model)

    save_data(IPCAs,dataset,name="PSF_instrumented")    
    evaluate_IPCAs(IPCAs,dataset,name="PSF_instrumented")

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
    