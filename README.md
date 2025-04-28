# IPCA-asset-pricing
IPCA application to asset pricing based on Kelly, Pruitt and Su (2018 and 2020).

**Supported Datasets:**
- Gu, Kelly and Xiu (2020) [RECOMMENDED]
- Open Asset Pricing by Chen and Zimmermann (2020)
- Grunfeld (1950)

## How to use
1. Install **requirements** from requirements.txt
    ``` 
    pip install -r /path/to/requirements.txt
    ```
2. Run **run_ipca.py** to
    - download data (caveat: requires WRDS credentials)
    - preprocess data
    - generate IPCA instances from IPCA.py (caveat: takes time until ALS convergence)
    ``` 
    python run_ipca.py
    ```
3. **Test IPCA results** (example usage)
    ``` python
    print(IPCAs[0].r2)
    print(IPCAs[0].Gamma)
    print(IPCAs[0].Fac)
    IPCAs[0].visualize_factors()
    IPCAs[0].visualize_gamma()
    ```
4. **Data** at every step (raw data, preprocessed data and result data) is stored as pickle objects and can be loaded for later use.
    ``` python
    with open('result_data.pkl', 'rb') as inp:
        IPCAs = pickle.load(inp)
    ```